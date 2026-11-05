import unittest
import tempfile
from pathlib import Path

from src.context import build_context
from src.dataset import Sentence
from src.translator import messages_for


class PromptTests(unittest.TestCase):
    def test_reference_never_enters_prompt(self):
        rows = [Sentence("a", 0, "甲", "SECRET REFERENCE"),
                Sentence("a", 1, "乙", "OTHER SECRET")]
        prompt = str(messages_for(build_context(rows, 1, 1)))
        self.assertIn("甲", prompt)
        self.assertIn("乙", prompt)
        self.assertNotIn("SECRET", prompt)
        baseline = str(messages_for(build_context(rows, 1, 0)))
        self.assertNotIn("甲", baseline)


class MetricTests(unittest.TestCase):
    def test_identical_predictions_and_coverage_validation(self):
        from src.evaluate import summarize
        rows = [dict(chapter_id="a", sentence_id=0, source="甲", reference="The wind stopped at dawn.",
                     prediction="The wind stopped at dawn.", context_size=size, truncated=False)
                for size in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            scores = summarize(rows, Path(folder) / "summary.csv")
            for score in scores:
                self.assertAlmostEqual(score["bleu"], 100)
                self.assertAlmostEqual(score["chrf"], 100)
            with self.assertRaises(ValueError):
                summarize(rows[:-1], Path(folder) / "bad.csv")
            rows[0]["truncated"] = True
            with self.assertRaises(ValueError):
                summarize(rows, Path(folder) / "bad.csv")
