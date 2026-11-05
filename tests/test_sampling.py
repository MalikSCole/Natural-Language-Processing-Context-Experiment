import unittest

from experiments.prepare_study import eligible_positions


class SamplingTests(unittest.TestCase):
    def test_content_filter_before_freezing(self):
        source = ["标题", "甲", "乙", "丙", "......", "…", "!?", "A1", "中文"]
        reference = ["R"] * len(source)
        self.assertEqual(eligible_positions(source, reference, True), [7, 8])

    def test_no_headings_or_incomplete_context(self):
        self.assertEqual(eligible_positions(["标题", "甲", "乙", "丙", "丁", "戊"],
                                            ["Title", "A", "B", "C", "D", ""]), [4])

    def test_rejects_misaligned_or_blank_source_chapter(self):
        self.assertEqual(eligible_positions(["甲"] * 5, ["A"] * 4), [])
        self.assertEqual(eligible_positions(["甲", "", "乙", "丙", "丁"], ["A"] * 5), [])
