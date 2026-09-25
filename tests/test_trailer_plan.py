"""预告片片段规划核心逻辑单元测试 + 暴力枚举对拍（纯标准库）。

暴力枚举全部候选子集，校验词区间互不重叠、总时长不超预算、必选说话人
全覆盖，按 (总评分降序, 总时长升序, 原时间顺序的候选序号序列字典序)
取最优，与 plan_trailer 对比。
"""

import random
import unittest
from types import SimpleNamespace

from app.errors import DomainError
from app.trailer_plan import plan_trailer


def W(start, end, text="字", speaker="A"):
    return SimpleNamespace(start_ms=start, end_ms=end, text=text, speaker=speaker)


def C(start_word, end_word, score):
    return SimpleNamespace(start_word=start_word, end_word=end_word, score=score)


def brute_force(words, candidates, budget_ms, required_speakers):
    """返回 (总评分, 总时长, 原时间顺序的候选序号元组) 或 None（无可行组合）。"""
    m = len(candidates)
    required = set(required_speakers)
    durs = [
        words[c.end_word].end_ms - words[c.start_word].start_ms for c in candidates
    ]
    best = None  # ((总评分, -总时长), 序号元组)
    for subset in range(1 << m):
        sel = [i for i in range(m) if subset >> i & 1]
        # 原时间顺序 = 词区间顺序（互不重叠时唯一）
        sel.sort(key=lambda i: (candidates[i].start_word, candidates[i].end_word, i))
        total_score = 0
        total_dur = 0
        speakers = set()
        ok = True
        for pos, i in enumerate(sel):
            c = candidates[i]
            if pos and c.start_word <= candidates[sel[pos - 1]].end_word:
                ok = False  # 词区间重叠
                break
            total_score += c.score
            total_dur += durs[i]
            speakers.add(words[c.start_word].speaker)
        if not ok or total_dur > budget_ms or not required <= speakers:
            continue
        key = (total_score, -total_dur)
        seq = tuple(sel)
        if best is None or key > best[0] or (key == best[0] and seq < best[1]):
            best = (key, seq)
    if best is None:
        return None
    (total_score, neg_dur), seq = best
    return total_score, -neg_dur, seq


class CandidateValidationTests(unittest.TestCase):
    WORDS = [
        W(0, 500, speaker="A"),
        W(500, 1000, speaker="A"),
        W(1000, 1500, speaker="B"),
        W(1500, 2000, speaker="B"),
    ]

    def test_index_out_of_range(self):
        for bad in (C(0, 4, 1), C(2, 5, 1), C(3, 2, 1), C(-1, 1, 1)):
            with self.assertRaises(DomainError) as ctx:
                plan_trailer(self.WORDS, [bad], 10_000, ["A"])
            self.assertEqual(ctx.exception.code, "candidate_index_out_of_range")
            self.assertEqual(ctx.exception.extra["candidate_index"], 0)

    def test_multi_speaker_candidate_rejected(self):
        with self.assertRaises(DomainError) as ctx:
            plan_trailer(self.WORDS, [C(1, 2, 1)], 10_000, ["A"])
        self.assertEqual(ctx.exception.code, "candidate_not_single_speaker")
        self.assertEqual(ctx.exception.extra["candidate_index"], 0)

    def test_single_speaker_run_boundary(self):
        # [1, 1] 属 A、[1, 2] 跨界：恰在说话人边界处判定
        clips, _, _ = plan_trailer(self.WORDS, [C(1, 1, 1)], 10_000, ["A"])
        self.assertEqual([c["candidate_index"] for c in clips], [0])
        with self.assertRaises(DomainError):
            plan_trailer(self.WORDS, [C(0, 2, 1)], 10_000, ["A"])

    def test_word_validation_reused(self):
        words = [W(1000, 2000), W(900, 1500)]  # 时间倒序
        with self.assertRaises(DomainError) as ctx:
            plan_trailer(words, [C(0, 0, 1)], 10_000, ["A"])
        self.assertEqual(ctx.exception.code, "time_out_of_order")

    def test_empty_words_reused(self):
        with self.assertRaises(DomainError) as ctx:
            plan_trailer([], [C(0, 0, 1)], 10_000, ["A"])
        self.assertEqual(ctx.exception.code, "empty_words")


class SelectionTests(unittest.TestCase):
    # 词序列：A 说词 0～1，B 说词 2～3；每词 500ms，端点相接
    WORDS = [
        W(0, 500, speaker="A"),
        W(500, 1000, speaker="A"),
        W(1000, 1500, speaker="B"),
        W(1500, 2000, speaker="B"),
    ]

    def test_basic_plan_and_trailer_times(self):
        candidates = [C(0, 1, 3), C(2, 3, 5)]
        clips, total_score, total_dur = plan_trailer(
            self.WORDS, candidates, 10_000, ["A", "B"]
        )
        self.assertEqual(total_score, 8)
        self.assertEqual(total_dur, 2000)
        self.assertEqual(
            clips,
            [
                {
                    "index": 0,
                    "candidate_index": 0,
                    "speaker": "A",
                    "score": 3,
                    "start_word": 0,
                    "end_word": 1,
                    "source_start_ms": 0,
                    "source_end_ms": 1000,
                    "duration_ms": 1000,
                    "trailer_start_ms": 0,
                    "trailer_end_ms": 1000,
                },
                {
                    "index": 1,
                    "candidate_index": 1,
                    "speaker": "B",
                    "score": 5,
                    "start_word": 2,
                    "end_word": 3,
                    "source_start_ms": 1000,
                    "source_end_ms": 2000,
                    "duration_ms": 1000,
                    "trailer_start_ms": 1000,
                    "trailer_end_ms": 2000,
                },
            ],
        )

    def test_overlap_excludes_combination(self):
        # 两个候选词区间重叠，只能二选一；评分高者胜
        candidates = [C(0, 1, 3), C(1, 1, 9)]
        clips, total_score, _ = plan_trailer(self.WORDS, candidates, 10_000, ["A"])
        self.assertEqual(total_score, 9)
        self.assertEqual([c["candidate_index"] for c in clips], [1])

    def test_budget_excludes_combination(self):
        # 两候选共 2000ms，预算 1500 放不下两段，而 A、B 都必选 -> 无可行组合
        candidates = [C(0, 1, 3), C(2, 3, 5)]
        with self.assertRaises(DomainError) as ctx:
            plan_trailer(self.WORDS, candidates, 1500, ["A", "B"])
        self.assertEqual(ctx.exception.code, "infeasible_trailer_plan")

    def test_budget_forces_choice(self):
        candidates = [C(0, 1, 3), C(2, 3, 5)]
        clips, total_score, total_dur = plan_trailer(
            self.WORDS, candidates, 1500, ["B"]
        )
        self.assertEqual((total_score, total_dur), (5, 1000))
        self.assertEqual([c["candidate_index"] for c in clips], [1])

    def test_required_speaker_forces_low_score_candidate(self):
        # 只要求 A 时可单拿高分段；要求 B 也在场时必须带上 0 分的 B 段
        candidates = [C(0, 1, 9), C(2, 3, 0)]
        clips, total_score, _ = plan_trailer(self.WORDS, candidates, 10_000, ["A"])
        self.assertEqual([c["candidate_index"] for c in clips], [0])
        self.assertEqual(total_score, 9)
        clips, total_score, _ = plan_trailer(
            self.WORDS, candidates, 10_000, ["A", "B"]
        )
        self.assertEqual([c["candidate_index"] for c in clips], [0, 1])
        self.assertEqual(total_score, 9)

    def test_unrequired_speaker_candidates_usable(self):
        # C 不在必选之列，但其高分片段仍可入选
        words = self.WORDS + [W(2000, 2500, speaker="C")]
        candidates = [C(0, 0, 1), C(4, 4, 9)]
        clips, total_score, _ = plan_trailer(words, candidates, 10_000, ["A"])
        self.assertEqual(total_score, 10)
        self.assertEqual([c["candidate_index"] for c in clips], [0, 1])

    def test_infeasible_when_speaker_missing(self):
        with self.assertRaises(DomainError) as ctx:
            plan_trailer(self.WORDS, [C(0, 1, 3)], 10_000, ["B"])
        err = ctx.exception
        self.assertEqual(err.code, "infeasible_trailer_plan")
        self.assertEqual(err.extra["candidate_count"], 1)
        self.assertEqual(err.extra["budget_ms"], 10_000)
        self.assertEqual(err.extra["required_speakers"], ["B"])

    def test_infeasible_when_budget_too_small(self):
        with self.assertRaises(DomainError) as ctx:
            plan_trailer(self.WORDS, [C(0, 1, 3)], 999, ["A"])
        self.assertEqual(ctx.exception.code, "infeasible_trailer_plan")

    def test_duplicate_required_speakers_deduped(self):
        clips, total_score, _ = plan_trailer(
            self.WORDS, [C(0, 1, 3)], 10_000, ["A", "A"]
        )
        self.assertEqual(total_score, 3)
        self.assertEqual(len(clips), 1)

    def test_zero_duration_candidate_at_zero_budget(self):
        words = [W(500, 500, speaker="A")]  # 零时长词
        clips, total_score, total_dur = plan_trailer(words, [C(0, 0, 7)], 0, ["A"])
        self.assertEqual((total_score, total_dur), (7, 0))
        self.assertEqual(clips[0]["trailer_start_ms"], 0)
        self.assertEqual(clips[0]["trailer_end_ms"], 0)


class TieBreakTests(unittest.TestCase):
    WORDS = [
        W(0, 500, speaker="A"),
        W(500, 1000, speaker="A"),
        W(1000, 1500, speaker="A"),
    ]

    def test_same_score_minimizes_duration(self):
        # 两个候选重叠（共享词 0）且同分：选时长更短者
        candidates = [C(0, 1, 5), C(0, 0, 5)]  # 1000ms vs 500ms
        clips, total_score, total_dur = plan_trailer(
            self.WORDS, candidates, 10_000, ["A"]
        )
        self.assertEqual((total_score, total_dur), (5, 500))
        self.assertEqual([c["candidate_index"] for c in clips], [1])

    def test_lexicographic_by_candidate_index_not_time(self):
        # 候选 0（词 1～2）在时间上晚于候选 1（词 0～1），两者重叠且同分同时长：
        # 按候选序号序列裁决，(0,) < (1,)，与时间先后无关
        candidates = [C(1, 2, 5), C(0, 1, 5)]
        clips, _, _ = plan_trailer(self.WORDS, candidates, 10_000, ["A"])
        self.assertEqual([c["candidate_index"] for c in clips], [0])

    def test_lexicographic_sequence_in_time_order(self):
        # 序列按原时间顺序排列后再做字典序比较：
        # 组合 {1,3}（词 1、词 0）时间序为 (3,1)，组合 {2}（词 0～1）为 (2,)，
        # 同分同时长时 (2,) < (3,1)；若误按候选序号顺序排列则会错选 (1,3)
        candidates = [C(2, 2, 0), C(1, 1, 3), C(0, 1, 6), C(0, 0, 3)]
        clips, total_score, total_dur = plan_trailer(
            self.WORDS, candidates, 1000, ["A"]
        )
        self.assertEqual((total_score, total_dur), (6, 1000))
        self.assertEqual([c["candidate_index"] for c in clips], [2])

    def test_emission_follows_time_order(self):
        # 选中片段按原时间顺序输出，预告片时刻也随之累加（与候选输入顺序无关）
        candidates = [C(2, 2, 1), C(0, 0, 1)]
        clips, _, _ = plan_trailer(self.WORDS, candidates, 10_000, ["A"])
        self.assertEqual([c["candidate_index"] for c in clips], [1, 0])
        self.assertEqual(
            [(c["trailer_start_ms"], c["trailer_end_ms"]) for c in clips],
            [(0, 500), (500, 1000)],
        )

    def test_shorter_prefix_wins(self):
        # 零分零时长候选 0 与 5 分候选 1：同分同时长下 (0,1) 与 (1,) 比较，
        # 首元素 0 < 1，故带上零分候选
        words = [W(300, 300, speaker="A"), W(500, 1000, speaker="A")]
        candidates = [C(0, 0, 0), C(1, 1, 5)]
        clips, total_score, total_dur = plan_trailer(words, candidates, 10_000, ["A"])
        self.assertEqual((total_score, total_dur), (5, 500))
        self.assertEqual([c["candidate_index"] for c in clips], [0, 1])


class DifferentialTests(unittest.TestCase):
    def test_matches_brute_force(self):
        random.seed(20260925)
        checked = 0
        for trial in range(400):
            n = random.randint(1, 12)
            t = 0
            words = []
            for _ in range(n):
                gap = random.choice([0, 0, 0, 100])
                t += gap
                dur = random.choice([0, 300, 500, 800])
                words.append(W(t, t + dur, speaker=random.choice("AAB")))
                t += dur

            # 在同一说话人连续区间内生成合法候选
            runs = []
            i = 0
            for j in range(1, n + 1):
                if j == n or words[j].speaker != words[j - 1].speaker:
                    runs.append((i, j - 1))
                    i = j
            m = random.randint(1, 9)
            candidates = []
            for _ in range(m):
                a, b = random.choice(runs)
                s = random.randint(a, b)
                e = random.randint(s, b)
                candidates.append(C(s, e, random.randint(0, 9)))
            budget = random.choice([0, 400, 1000, 2500, 10_000])
            # 必选说话人大多取自实际出现的说话人，偶发不存在者以覆盖不可行情形
            pool = sorted({w.speaker for w in words})
            if random.random() < 0.15:
                pool.append("ZZZ")
            required = random.sample(pool, random.randint(1, min(3, len(pool))))
            if random.random() < 0.1:
                required.append(required[0])  # 偶发重复必选说话人

            expected = brute_force(words, candidates, budget, required)
            if expected is None:
                with self.assertRaises(DomainError) as ctx:
                    plan_trailer(words, candidates, budget, required)
                self.assertEqual(ctx.exception.code, "infeasible_trailer_plan")
            else:
                exp_score, exp_dur, exp_seq = expected
                clips, total_score, total_dur = plan_trailer(
                    words, candidates, budget, required
                )
                seq = tuple(c["candidate_index"] for c in clips)
                self.assertEqual(total_score, exp_score, f"trial {trial} score")
                self.assertEqual(total_dur, exp_dur, f"trial {trial} duration")
                self.assertEqual(seq, exp_seq, f"trial {trial} tie-break")
                # 额外校验：预告片时刻从 0 累加、时长与源时刻自洽、不超预算
                t_acc = 0
                for c in clips:
                    self.assertEqual(c["trailer_start_ms"], t_acc)
                    self.assertEqual(
                        c["duration_ms"], c["source_end_ms"] - c["source_start_ms"]
                    )
                    self.assertEqual(
                        c["source_start_ms"], words[c["start_word"]].start_ms
                    )
                    self.assertEqual(c["source_end_ms"], words[c["end_word"]].end_ms)
                    t_acc += c["duration_ms"]
                    self.assertEqual(c["trailer_end_ms"], t_acc)
                self.assertEqual(t_acc, total_dur)
                self.assertLessEqual(total_dur, budget)
                # 词区间互不重叠、必选说话人全覆盖
                for a, b in zip(clips, clips[1:]):
                    self.assertLess(a["end_word"], b["start_word"])
                covered = {c["speaker"] for c in clips}
                self.assertTrue(set(required) <= covered)
                checked += 1
        self.assertGreater(checked, 100)  # 确保有足够多的可行对拍样本


if __name__ == "__main__":
    unittest.main()
