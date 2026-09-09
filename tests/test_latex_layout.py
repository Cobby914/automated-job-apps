"""Rendered layout regression checks (requires a local TeX installation)."""

from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobapps.latex import compile_tex_with_log, escape_latex, render_documents
from jobapps.models import Contact, Job, TailoredResume


@unittest.skipUnless(
    shutil.which("tectonic") or (shutil.which("xelatex") and shutil.which("latexmk")),
    "A TeX compiler is needed to measure rendered bullet spacing",
)
class BulletLayoutTests(unittest.TestCase):
    def test_nearly_full_bullets_do_not_add_blank_lines(self) -> None:
        # These single-line bullets produced an empty line in actual packets.
        bullets = [
            "Short control bullet.",
            "Engineered a Python/CARLA pipeline for synchronized radar-camera data, labels, calibration, and actor metadata.",
            "Built a Python sync engine across Canvas and Notion REST APIs, automating task tracking with GitHub Actions.",
        ]
        with TemporaryDirectory() as tmp:
            tex, _ = render_documents(
                Job(company="Example", title="Intern", description="Layout test"),
                Contact(name="Example"),
                TailoredResume(experience=[], projects=[], education=[], skills=[]),
                None, Path(tmp), compile_pdf=False,
            )
            preamble = tex.read_text(encoding="utf-8").split(r"\begin{document}")[0]
            measurements = []
            for index, bullet in enumerate(bullets):
                measurements.append(
                    r"\setbox0=\vbox{\hsize=\textwidth"
                    r"\resumeSubHeadingListStart\item"
                    r"\resumeItemListStart\resumeItem{" + escape_latex(bullet) + "}"
                    r"\resumeItem{Following bullet.}\resumeItemListEnd"
                    r"\resumeSubHeadingListEnd}"
                    + rf"\typeout{{BULLET{index}:\the\ht0}}"
                )
            tex.write_text(
                preamble + "\n\\begin{document}\n" + "\n".join(measurements)
                + "\nLayout check.\n\\end{document}\n", encoding="utf-8",
            )
            if compiler := shutil.which("tectonic"):
                result = subprocess.run(
                    [compiler, "--untrusted", "--only-cached", "--keep-logs", str(tex)],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                log = tex.with_suffix(".log").read_text(encoding="utf-8")
            else:
                _, log = compile_tex_with_log(tex)
            heights = re.findall(r"BULLET\d:([\d.]+)pt", log)
            self.assertEqual(len(heights), len(bullets), log)
            for height in heights[1:]:
                # Glyph ascenders vary slightly; an extra line adds ~12pt.
                self.assertAlmostEqual(float(height), float(heights[0]), delta=1.0)
