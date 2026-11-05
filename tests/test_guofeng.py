import tempfile
import unittest
from pathlib import Path

from src.guofeng import load_guofeng
from src.context import build_context


class GuofengTests(unittest.TestCase):
    def load_pair(self, source, reference):
        with tempfile.TemporaryDirectory() as folder:
            zh, en = Path(folder) / "sample.zh", Path(folder) / "sample.en"
            zh.write_text(source, encoding="utf-8")
            en.write_text(reference, encoding="utf-8")
            return load_guofeng(zh, en)

    def test_preserves_order_and_book_boundaries(self):
        template = ('<BOOK id="a">\n<CHAPTER id="1">\n{one}\n{two}\n'
                    '</CHAPTER>\n</BOOK>\n<BOOK id="b">\n'
                    '<CHAPTER id="1">\n{three}\n</CHAPTER>\n</BOOK>\n')
        rows = self.load_pair(template.format(one="甲", two="乙", three="丙"),
                              template.format(one="A", two="B", three="C"))
        self.assertEqual([r.sentence_id for r in rows], [0, 1, 0])
        self.assertEqual(rows[1].reference, "B")
        self.assertNotEqual(rows[0].chapter_id, rows[2].chapter_id)
        self.assertEqual(build_context(rows, 2, 3).previous_sources, ())

    def test_rejects_misalignment_and_invalid_structure(self):
        valid = '<BOOK id="a">\n<CHAPTER id="1">\n甲\n</CHAPTER>\n</BOOK>\n'
        for invalid in [valid.replace("甲", "A\nB"),
                        valid.replace('id="1"', 'id="2"'),
                        valid.replace("</BOOK>", ""),
                        valid.replace("<BOOK id=\"a\">", "")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.load_pair(valid, invalid)

    def test_missing_reference_preserves_source_context(self):
        template = '<BOOK id="a">\n<CHAPTER id="1">\n{}\n</CHAPTER>\n</BOOK>\n'
        rows = self.load_pair(template.format("甲\n乙\n丙"), template.format("A\n\nC"))
        self.assertEqual([r.reference for r in rows], ["A", "", "C"])
        self.assertEqual(build_context(rows, 2, 2).previous_sources, ("甲", "乙"))
        with self.assertRaises(ValueError):
            self.load_pair(template.format("甲\n\n丙"), template.format("A\nB\nC"))
