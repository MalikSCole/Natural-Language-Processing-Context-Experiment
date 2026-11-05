import unittest

from sacrebleu.metrics import BLEU, CHRF

from src.significance import paired_cluster_test, holm_adjust


class SignificanceTests(unittest.TestCase):
    def test_holm_monotonic_adjustment(self):
        self.assertEqual(holm_adjust([0.04, 0.01]), [0.04, 0.02])
        self.assertEqual(holm_adjust([0.04, 0.03]), [0.06, 0.06])

    def test_identical_outputs_have_zero_difference(self):
        text = ["The wind stopped at dawn.", "She opened the window slowly."]
        for metric in [BLEU(), CHRF()]:
            result = paired_cluster_test(metric, text, text, text, ["a", "b"],
                                         permutations=40, bootstrap=40)
            self.assertEqual(result["difference"], 0)
            self.assertEqual(result["p_two_sided"], 1)
            self.assertEqual(result["ci95_difference"], [0, 0])

    def test_one_cluster_is_one_exchangeable_unit(self):
        refs = ["The wind stopped at dawn."] * 10
        result = paired_cluster_test(BLEU(), ["Wrong text entirely."] * 10,
                                     refs, refs, ["same novel"] * 10,
                                     permutations=40, bootstrap=40)
        self.assertEqual(result["clusters"], 1)
        self.assertEqual(result["p_two_sided"], 1)
        self.assertGreater(result["difference"], 0)

    def test_rejects_unpaired_input(self):
        with self.assertRaises(ValueError):
            paired_cluster_test(BLEU(), ["a"], [], ["a"], ["book"])
