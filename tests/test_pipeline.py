import csv
import tempfile
import unittest
from pathlib import Path

from experiments.run_all import prepare_inputs
from src.context import build_context
from src.dataset import Sentence, load_dataset


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [Sentence("a", i, f"中文{i}", f"English {i}") for i in range(4)]
        self.rows.append(Sentence("b", 0, "新章", "New chapter"))

    def test_context_order_and_boundary(self):
        self.assertEqual(build_context(self.rows, 3, 3).previous_sources,
                         ("中文0", "中文1", "中文2"))
        self.assertEqual(build_context(self.rows, 4, 3).previous_sources, ())
        self.assertEqual(build_context(self.rows, 3, 0).previous_sources, ())
        self.assertEqual(build_context(self.rows, 1, 3).previous_sources, ("中文0",))
        self.assertNotIn("English", build_context(self.rows, 3, 3).input_text)

    def test_invalid_context(self):
        with self.assertRaises(ValueError):
            build_context(self.rows, 0, 4)
        with self.assertRaises(IndexError):
            build_context(self.rows, -1, 0)

    def test_loader_rejects_bad_order(self):
        for body in ["a,2,中,English\na,1,文,Text\n",
                     "a,0,中,English\nb,0,文,Text\na,1,字,Word\n",
                     "a,0,,English\n"]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "data.csv"
                path.write_text("chapter_id,sentence_id,source,reference\n" + body,
                                encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dataset(path)

    def test_sample_export(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inputs.csv"
            self.assertEqual(prepare_inputs(Path("data/samples/synthetic.csv"), output, 20), 80)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 80)
            self.assertEqual([int(row["context_size"]) for row in rows[:4]], [0, 1, 2, 3])
            self.assertEqual(len({row["source"] for row in rows[:4]}), 1)


if __name__ == "__main__":
    unittest.main()
