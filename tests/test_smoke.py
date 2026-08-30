"""Optional live-API smoke test. Run with JOBAPPS_SMOKE=1."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.config import load_env
from jobapps.llm import estimate_cost_usd, generate_structured, usage_records
from jobapps.models import TextPayload

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.getenv("JOBAPPS_SMOKE", "").strip() in {"1", "true", "yes"}, "set JOBAPPS_SMOKE=1")
class LiveApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_env()
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise unittest.SkipTest("OPENAI_API_KEY is not set")
        if not os.getenv("OPENAI_REPAIR_MODEL", "").strip() and not os.getenv("OPENAI_WRITER_MODEL", "").strip():
            raise unittest.SkipTest("Set OPENAI_REPAIR_MODEL or OPENAI_WRITER_MODEL")

    def test_structured_output_tokens_and_cost(self) -> None:
        from jobapps.config import repair_model, writer_model
        from jobapps.llm import begin_usage_collection

        begin_usage_collection()
        model = os.getenv("OPENAI_REPAIR_MODEL", "").strip() or writer_model()
        payload = generate_structured(
            system="Return a short grounded sentence.",
            user="Write the word ping.",
            model=model,
            schema=TextPayload,
            purpose="smoke",
        )
        self.assertTrue(payload.text.strip())
        records = usage_records()
        self.assertTrue(records)
        last = records[-1]
        self.assertEqual(last.provider, "openai")
        self.assertEqual(last.model, model)
        self.assertGreaterEqual(last.input_tokens, 1)
        self.assertGreaterEqual(last.output_tokens, 1)
        estimated = estimate_cost_usd(
            last.model, last.input_tokens, last.cached_input_tokens, last.output_tokens
        )
        self.assertGreaterEqual(estimated, 0.0)
        self.assertAlmostEqual(last.estimated_cost_usd, estimated)


if __name__ == "__main__":
    unittest.main()
