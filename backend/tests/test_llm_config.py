"""Tests for the shared Groq model configuration.

Run from backend/:  python -m unittest discover -s tests -v
"""

import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


class GroqModelConfigTest(unittest.TestCase):
    def test_defaults_to_gpt_oss_20b_when_env_unset(self):
        from llm_config import groq_model

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_MODEL", None)
            self.assertEqual(groq_model(), "openai/gpt-oss-20b")

    def test_env_var_overrides_default(self):
        from llm_config import groq_model

        with mock.patch.dict(os.environ, {"GROQ_MODEL": "qwen/qwen3.8-27b"}):
            self.assertEqual(groq_model(), "qwen/qwen3.8-27b")

    def test_no_agent_hardcodes_a_model_name(self):
        hardcoded = re.compile(r'model\s*=\s*["\']')
        offenders = [
            f"{path.name}:{lineno}"
            for path in sorted((BACKEND_DIR / "agents").glob("*.py"))
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if hardcoded.search(line)
        ]
        self.assertEqual(offenders, [], f"Hardcoded model names found: {offenders}")


if __name__ == "__main__":
    unittest.main()
