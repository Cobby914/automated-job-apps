from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.career import CanonicalBullet, Metric, _validate_canonical_sources, load_career_bank
from jobapps.models import numeric_claim_tokens

ROOT = Path(__file__).resolve().parents[1]


class CareerBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = load_career_bank(ROOT / "career")

    def test_loads_profile_experiences_projects_skills(self) -> None:
        self.assertEqual(self.bank.profile.contact.name, "Colin Kwon")
        self.assertTrue(self.bank.profile.education)
        ids = {item.id for item in self.bank.experiences}
        self.assertEqual(
            ids,
            {"mk-lending", "tena", "fipet", "uci-scalesense", "commit-the-change"},
        )
        project_ids = {item.id for item in self.bank.projects}
        self.assertIn("genome-sequencing", project_ids)
        self.assertIn("unity-rl", project_ids)
        self.assertIn("unix-shell", project_ids)

    def test_source_ids_are_unique(self) -> None:
        seen: set[str] = set()
        for record in [*self.bank.experiences, *self.bank.projects]:
            for source_id in record.source_ids():
                self.assertNotIn(source_id, seen, source_id)
                seen.add(source_id)
        self.assertGreater(len(seen), 40)

    def test_track_membership(self) -> None:
        exp = self.bank.experience_by_id()
        self.assertIn("ai", exp["uci-scalesense"].tracks)
        self.assertNotIn("swe", exp["uci-scalesense"].tracks)
        self.assertIn("swe", exp["fipet"].tracks)
        self.assertNotIn("ai", exp["fipet"].tracks)
        projects = self.bank.project_by_id()
        self.assertIn("swe", projects["reel-in"].tracks)
        self.assertNotIn("ai", projects["reel-in"].tracks)
        self.assertIn("ai", projects["smart-step"].tracks)

    def test_yaml_skills_whitelist(self) -> None:
        allowed = self.bank.skills.allowed_names()
        for name in ("python", "c/c++", "linux/unix", "pytorch", "carla", "posix", "canvas api", "sumo"):
            self.assertIn(name, allowed, name)
        self.assertNotIn("kubernetes", allowed)

    def test_metrics_declare_kind(self) -> None:
        for record in [*self.bank.experiences, *self.bank.projects]:
            for metric in record.metrics:
                self.assertIn(metric.kind, {"absolute", "relative", "count"}, metric.id)

    def test_canonical_numeric_claims_have_metric_support(self) -> None:
        for record in [*self.bank.experiences, *self.bank.projects]:
            for bullet in record.bullets:
                if bullet.text.strip().lower().startswith("stack:"):
                    continue
                claims = numeric_claim_tokens(bullet.text)
                if not claims:
                    continue
                cited = {item.id for item in record.metrics if item.id in set(bullet.sources)}
                self.assertTrue(cited, bullet.id)

    def test_load_rejects_unknown_canonical_source(self) -> None:
        broken = self.bank.model_copy(
            update={
                "experiences": [
                    self.bank.experiences[0].model_copy(
                        update={
                            "bullets": [
                                CanonicalBullet(
                                    id="bad.bullet.1",
                                    text="Invented claim with no grounding.",
                                    sources=["missing.metric.9"],
                                )
                            ]
                        }
                    )
                ]
            }
        )
        with self.assertRaises(ValueError) as ctx:
            _validate_canonical_sources(broken)
        self.assertIn("unknown source", str(ctx.exception))

    def test_relative_metrics_are_labeled(self) -> None:
        tena = self.bank.experience_by_id()["tena"]
        self.assertEqual(tena.metrics[0].kind, "relative")
        self.assertIn("relative", tena.metrics[0].text.casefold())
        genome = self.bank.project_by_id()["genome-sequencing"]
        self.assertEqual(genome.metrics[0].kind, "absolute")
        self.assertIn("0.40", genome.metrics[0].text)
        self.assertIn("0.70", genome.metrics[0].text)

    def test_metric_requires_kind(self) -> None:
        with self.assertRaises(ValueError):
            Metric.model_validate({"id": "x.metric.1", "text": "Grew 30%."})
        with self.assertRaises(ValueError):
            Metric.model_validate(
                {"id": "x.metric.1", "text": "Grew 30%.", "kind": "delta"}
            )

    def test_canonical_kind_mismatch_is_rejected(self) -> None:
        tena = self.bank.experience_by_id()["tena"]
        broken = self.bank.model_copy(
            update={
                "experiences": [
                    tena.model_copy(
                        update={
                            "bullets": [
                                CanonicalBullet(
                                    id="bad.kind.1",
                                    text="Drove 30 extra visits after launch.",
                                    sources=[tena.metrics[0].id],
                                )
                            ]
                        }
                    )
                ]
            }
        )
        with self.assertRaises(ValueError) as ctx:
            _validate_canonical_sources(broken)
        self.assertIn("relative metric", str(ctx.exception))
        self.assertIn("percent", str(ctx.exception).casefold())


if __name__ == "__main__":
    unittest.main()
