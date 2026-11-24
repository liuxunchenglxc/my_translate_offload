import unittest
from unittest.mock import patch

from worker.errors import StaleAttempt
from worker.worker import TaskWorker
from worker.document import _text_blocks, _split_long_paragraph
from worker.model import LocalChatModel
from worker.worker import validate_extractable_content


class DocumentTests(unittest.TestCase):
    def test_text_blocks_reconstruct_source(self):
        source = "Title\n\nParagraph one.\n\nParagraph two.\n"
        blocks = _text_blocks(source, markdown=False)
        self.assertEqual("".join(block["text"] for block in blocks), source)

    def test_crlf_paragraphs_reconstruct_source(self):
        source = "Title\r\n\r\nParagraph one.\r\n\r\nParagraph two.\r\n"
        blocks = _text_blocks(source, markdown=False)
        self.assertEqual("".join(block["text"] for block in blocks), source)
        separators = [block for block in blocks if not block["translate"]]
        self.assertGreaterEqual(len(separators), 2)

    def test_markdown_fence_is_fixed(self):
        source = "# Heading\n\n```python\nprint('hello')\n```\n\nText."
        blocks = _text_blocks(source, markdown=True)
        self.assertEqual("".join(block["text"] for block in blocks), source)
        code = [block for block in blocks if block["text"].startswith("```")]
        self.assertEqual(len(code), 1)
        self.assertFalse(code[0]["translate"])

    def test_long_split_reconstructs_source(self):
        source = ("First sentence. Second sentence. " * 200) + "End."
        parts = _split_long_paragraph(source, 300)
        self.assertTrue(all(len(part) <= 300 for part in parts))
        self.assertEqual("".join(parts), source)

    def test_model_translation_preserves_outer_whitespace(self):
        model = LocalChatModel()
        model.chat = lambda *args, **kwargs: "translated text"
        translated = model.translate("\n  source text  \n", "简体中文", "English", [])
        self.assertEqual(translated, "\n  translated text  \n")

    def test_auto_source_language_is_described_to_the_model(self):
        model = LocalChatModel()
        captured = {}
        def fake_chat(system, user, **kwargs):
            captured["system"] = system
            return "translated text"
        model.chat = fake_chat
        model.translate("source text", "简体中文", "自动检测", [])
        self.assertIn("automatically detected source language", captured["system"])


    def test_worker_writes_attempt_sidecar_not_server_record(self):
        class FakeStore:
            def __init__(self):
                self.files = {
                    "task-key.task.json": {
                        "attempt": 2,
                        "phase": "translation",
                        "status": "queued",
                    }
                }
            def load_json(self, name, missing_ok=False):
                value = self.files.get(name)
                if value is None and not missing_ok:
                    raise FileNotFoundError(name)
                return value
            def save_json(self, name, value):
                self.files[name] = value
            def close(self):
                pass

        store = FakeStore()
        with patch("worker.worker.PersonalStore", return_value=store):
            worker = TaskWorker("task-key", 2, "translation")
            worker.update(status="running", message="test heartbeat")
            self.assertEqual(store.files["task-key.task.json"]["status"], "queued")
            self.assertEqual(
                store.files["task-key.attempt.2.state.json"]["status"],
                "running",
            )
            store.files["task-key.task.json"]["attempt"] = 3
            with self.assertRaises(StaleAttempt):
                worker.update(status="running")
            worker.store.close()

    def test_image_only_pdf_is_rejected(self):
        document = {"format": "pdf_text", "blocks": [{"text": "---", "translate": False}]}
        with self.assertRaisesRegex(RuntimeError, "不执行 OCR"):
            validate_extractable_content(document)

    def test_markdown_code_only_document_is_allowed(self):
        document = {"format": "markdown", "blocks": [{"text": "```\ncode\n```", "translate": False}]}
        validate_extractable_content(document)


if __name__ == "__main__":
    unittest.main()
