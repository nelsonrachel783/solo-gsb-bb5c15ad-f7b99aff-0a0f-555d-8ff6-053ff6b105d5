"""预告片片段规划核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

从 1～20 个候选片段中选出词区间互不重叠、总时长不超 budget_ms 的子集，
使每个必选说话人（1～3 个，重复项按首次出现去重）都至少出现一次；
先最大化总评分，同分最小化总时长，再按原时间顺序的候选序号序列
字典序裁决。

候选片段须为单一说话人的连续词区间（start_word / end_word 为 0-based、
含端点的词序号），时长取末词 end_ms 减首词 start_ms。

算法（排序后缀 DP + 带备忘的递归重建）：

* 候选按 (start_word, end_word, 输入序号) 排序；nxt[p] 为排序后第一个
  首词序号大于 p 末词序号的位置（二分），即与 p 词区间不重叠的最早后继。
  词序列时间有序，词序号顺序即时间顺序，互不重叠的选中集合在排序后的
  相对顺序就是“原时间顺序”；
* h[p][need]：从排序位置 p 起（含）选取、使必选说话人覆盖集合**包含**
  need（bitmask，去重后至多 3 人、至多 8 态）的 Pareto 前沿
  {总时长: 最大总评分}，被支配项（时长不短且评分不高）已剪除。转移合并
  “跳过 p”（h[p+1][need]）与“选 p 后接 h[nxt[p]][need & ~bit_p]”；
* 最优值：h[0][full] 中时长不超预算者先取最大评分、再取最小时长；
* 重建：从 (0, full, 最优评分, 最优时长) 递归——每步用前沿精确命中
  (时长, 评分) 判断“跳过 / 选取”哪个分支仍能达成目标，两边都可行时
  取候选序号序列字典序较小者；lru_cache 保证每状态只算一次。被支配的
  中间状态不可能属于最优解（否则全局解可被同覆盖、更短或更高分的状态
  替换），故只查 Pareto 前沿不丢最优性；固定前缀下序列比较只取决于
  各自后缀，局部 min 具有最优子结构。
"""

from bisect import bisect_right
from functools import lru_cache

from .errors import DomainError
from .segmentation import validate_words


def _fail(code, message, **extra):
    raise DomainError(code, message, **extra)


def plan_trailer(words, candidates, budget_ms, required_speakers):
    """选出最优预告片片段组合，返回 (clips, total_score, total_duration_ms)。

    参数 words 中每个对象需具备属性：start_ms / end_ms / text / speaker
    （沿用同一套词序与时间校验）；candidates 每项需具备属性
    start_word / end_word / score；required_speakers 为 1～3 个非空说话人
    标识（重复项按首次出现去重，语义不变）。

    clips 按原时间顺序排列，每项为 dict：index / candidate_index / speaker /
    score / start_word / end_word / source_start_ms / source_end_ms /
    duration_ms / trailer_start_ms / trailer_end_ms；trailer 时刻从 0 起
    按选中顺序累加时长。候选词序号越界、跨说话人，或无可行组合时抛出
    DomainError（HTTP 422）。
    """
    validate_words(words)
    n = len(words)
    m = len(candidates)

    # run_end[i]：从 i 开始的同一说话人连续区间的右开边界
    run_end = [0] * n
    boundary = n
    for i in range(n - 1, -1, -1):
        if i == n - 1 or words[i].speaker != words[i + 1].speaker:
            boundary = i + 1
        run_end[i] = boundary

    # ---- 候选校验：词序号区间合法、单一说话人连续区间 ----
    for ci, c in enumerate(candidates):
        s, e = c.start_word, c.end_word
        if not 0 <= s <= e < n:
            _fail(
                "candidate_index_out_of_range",
                f"候选片段 {ci} 的词序号区间 [{s}, {e}] 不合法，"
                f"须满足 0 <= start_word <= end_word < {n}",
                candidate_index=ci,
                start_word=s,
                end_word=e,
            )
        if e >= run_end[s]:
            _fail(
                "candidate_not_single_speaker",
                f"候选片段 {ci} 的词序号区间 [{s}, {e}] 跨越多个说话人，"
                "须为单一说话人的连续词区间",
                candidate_index=ci,
                start_word=s,
                end_word=e,
            )

    # ---- 必选说话人 bitmask（按首次出现去重）----
    required = list(dict.fromkeys(required_speakers))
    bit_of_speaker = {spk: 1 << i for i, spk in enumerate(required)}
    full = (1 << len(required)) - 1

    # ---- 候选按词序排序，预处理时长 / 评分 / 说话人 bit / 后继指针 ----
    order = sorted(
        range(m),
        key=lambda i: (candidates[i].start_word, candidates[i].end_word, i),
    )
    starts = [candidates[i].start_word for i in order]
    ends = [candidates[i].end_word for i in order]
    durs = [words[ends[p]].end_ms - words[starts[p]].start_ms for p in range(m)]
    scores = [candidates[order[p]].score for p in range(m)]
    bits = [bit_of_speaker.get(words[starts[p]].speaker, 0) for p in range(m)]
    nxt = [bisect_right(starts, ends[p]) for p in range(m)]

    # ---- 后缀 DP：h[p][need] = Pareto 前沿 {总时长: 最大总评分} ----
    h = [[None] * (full + 1) for _ in range(m + 1)]
    for need in range(full + 1):
        h[m][need] = {0: 0} if need == 0 else {}
    for p in range(m - 1, -1, -1):
        row = h[p]
        skip_row = h[p + 1]
        take_row = h[nxt[p]]
        dc, sc, bc = durs[p], scores[p], bits[p]
        for need in range(full + 1):
            merged = dict(skip_row[need])  # 跳过 p
            src = take_row[need & ~bc]     # 选 p 后的可行后缀
            for d, s in src.items():
                nd, ns = d + dc, s + sc
                if merged.get(nd, -1) < ns:
                    merged[nd] = ns
            # 剪除被支配项：时长升序扫描，仅保留评分严格更高者
            frontier = {}
            best_so_far = -1
            for d in sorted(merged):
                s = merged[d]
                if s > best_so_far:
                    frontier[d] = s
                    best_so_far = s
            row[need] = frontier

    # ---- 最优值：预算内先最大化总评分，同分最小化总时长 ----
    best = None  # (总评分, 总时长)
    for d, s in h[0][full].items():
        if d > budget_ms:
            continue
        if best is None or s > best[0] or (s == best[0] and d < best[1]):
            best = (s, d)
    if best is None:
        _fail(
            "infeasible_trailer_plan",
            "不存在满足全部约束的预告片片段组合",
            candidate_count=m,
            budget_ms=budget_ms,
            required_speakers=list(required_speakers),
        )
    best_score, best_dur = best

    # ---- 递归重建字典序最小的候选序号序列（原时间顺序）----
    @lru_cache(maxsize=None)
    def reconstruct(p, need, score_target, dur_target):
        """从排序位置 p 起恰好达成目标的字典序最小原始候选序号元组；不可行为 None。"""
        if p == m:
            return () if need == 0 and score_target == 0 and dur_target == 0 else None
        take_seq = skip_seq = None
        rest_need = need & ~bits[p]
        # 前沿命中即保证该分支存在恰达目标的选法（被支配状态不属于最优解）
        if h[nxt[p]][rest_need].get(dur_target - durs[p]) == score_target - scores[p]:
            sub = reconstruct(
                nxt[p], rest_need, score_target - scores[p], dur_target - durs[p]
            )
            if sub is not None:
                take_seq = (order[p],) + sub
        if h[p + 1][need].get(dur_target) == score_target:
            skip_seq = reconstruct(p + 1, need, score_target, dur_target)
        if take_seq is None:
            return skip_seq
        if skip_seq is None:
            return take_seq
        return min(take_seq, skip_seq)

    chosen = reconstruct(0, full, best_score, best_dur)

    # ---- 输出：源时刻 + 从 0 累加的预告片时刻 ----
    clips = []
    trailer_t = 0
    for idx, ci in enumerate(chosen):
        c = candidates[ci]
        s, e = c.start_word, c.end_word
        source_start = words[s].start_ms
        source_end = words[e].end_ms
        duration = source_end - source_start
        clips.append(
            {
                "index": idx,
                "candidate_index": ci,
                "speaker": words[s].speaker,
                "score": c.score,
                "start_word": s,
                "end_word": e,
                "source_start_ms": source_start,
                "source_end_ms": source_end,
                "duration_ms": duration,
                "trailer_start_ms": trailer_t,
                "trailer_end_ms": trailer_t + duration,
            }
        )
        trailer_t += duration
    return clips, best_score, trailer_t
