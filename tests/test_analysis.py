"""Analysis regressions using explicitly labeled development-only fixtures.

Every temporary artifact lives under work/real_smoke/analysis_tests. No test
creates or modifies assignment Results. Sine-wave fixtures test measurements,
not synthesis quality; existing reconstruction WAVs are only read.
"""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import yaml

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT.parent.parent / "work" / "real_smoke"
SPEC = importlib.util.spec_from_file_location("analyze_results", PROJECT / "scripts" / "analyze_results.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        fixture_parent = WORK / "analysis_tests"
        fixture_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="development-", dir=fixture_parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "DEVELOPMENT_ONLY.txt").write_text("Synthetic analysis regression fixture; never assignment inference.\n")
        self.result = self.root / "development_results"
        (self.result / "predictions").mkdir(parents=True)
        (self.result / "test_generated_audios").mkdir()
        self.checkpoint = self.root / "development-checkpoint.bin"
        self.checkpoint.write_bytes(b"development test hash fixture, not a trained model")
        self.samples = self.root / "development_samples.json"
        self.samples.write_text(json.dumps([{"id": "DEV1", "text": "Development fixture only."}]))
        self.config = self.root / "development_config.yaml"
        self.config.write_text(yaml.safe_dump({"preprocess": {"preprocessing": {
            "audio": {"sampling_rate": 22050}, "stft": {"hop_length": 256}}}}))
        self.arrays = {"phones": np.array(["SIL", "a", "b", "SIL", "SIL", "c", "SIL"]),
                       "duration_frames": np.array([1, 2, 0, 3, 4, 2, 1]),
                       "pitch_hz": np.array([-50., 120., 200., -60., -70., 240., -80.]),
                       "energy": np.array([-1., .1, .2, -2., -3., .3, -4.])}
        total = int(self.arrays["duration_frames"].sum())
        self.arrays["duration_seconds"] = self.arrays["duration_frames"] * 256 / 22050
        self.arrays["mel"] = np.zeros((total, 80), dtype=np.float32)
        self.npz = self.result / "predictions" / "DEV1.npz"
        self.save_prediction()
        self.wav = self.result / "test_generated_audios" / "DEV1.wav"
        samples = .2 * np.sin(2 * np.pi * 220 * np.arange(total * 256) / 22050)
        samples[:2] = [1., -1.]
        sf.write(self.wav, samples, 22050, subtype="PCM_16")
        self.records = [{"id": "DEV1", "text": "Development fixture only.",
                         "phonemes": self.arrays["phones"].tolist(), "checkpoint_step": 3,
                         "checkpoint_sha256": analysis.sha256(self.checkpoint),
                         "audio": {"sha256": analysis.sha256(self.wav), "frames": len(samples),
                                   "seconds": len(samples) / 22050}}]
        self.save_manifest()

    def save_prediction(self):
        np.savez_compressed(self.npz, **self.arrays)

    def save_manifest(self):
        (self.result / "inference_manifest.json").write_text(json.dumps(self.records))

    def analyze(self, **kwargs):
        return analysis.analyze_language(self.root, "English", result_dir=self.result,
                                         checkpoint=self.checkpoint, samples_path=self.samples,
                                         config_path=self.config, development=True,
                                         skip_acoustic_f0=True, **kwargs)

    def test_measures_saved_wav_and_excludes_sil_from_predictions(self):
        output = self.analyze()
        item = output["samples"][0]
        waveform, rate = sf.read(self.wav)
        self.assertEqual(output["scope"], "development fixture evidence")
        self.assertEqual(output["source"]["test_samples_sha256"], analysis.sha256(self.samples))
        self.assertAlmostEqual(item["wav"]["rms"], np.sqrt(np.mean(waveform ** 2)))
        self.assertEqual(item["wav"]["peak"], 1.)
        self.assertEqual(item["wav"]["full_scale_sample_count"], 2)
        self.assertAlmostEqual(item["wav"]["clipping_fraction_at_pcm_rails"], 2 / len(waveform))
        variance = item["prediction"]["predicted_variance"]
        self.assertAlmostEqual(variance["pitch_hz_non_sil"]["mean"], (120 + 200 + 240) / 3)
        self.assertAlmostEqual(variance["pitch_hz_non_sil"]["duration_weighted_mean"], 180.)
        self.assertAlmostEqual(variance["energy_non_sil"]["mean"], .2)
        self.assertEqual(item["prediction"]["zero_duration_non_sil_phone_count"], 1)
        self.assertEqual(item["prediction"]["max_phone"]["phone"], "a")
        self.assertEqual(item["prediction"]["max_pause"]["duration_frames"], 7)
        self.assertEqual(item["prediction"]["max_pause"]["start_phone_index"], 3)
        self.assertEqual(item["prediction"]["max_pause"]["end_phone_index_inclusive"], 4)
        self.assertEqual(item["acoustic_f0"]["status"], "skipped")
        self.assertNotIn("voiced_fraction", item["acoustic_f0"])

    def test_writes_json_and_csv_with_distinct_predicted_and_acoustic_columns(self):
        output = self.analyze()
        analysis.write_analysis(self.result, output)
        saved = json.loads((self.result / "analysis.json").read_text())
        self.assertEqual(saved["samples"][0]["wav"]["sha256"], analysis.sha256(self.wav))
        with (self.result / "analysis.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["acoustic_f0_status"], "skipped")
        self.assertEqual(row["acoustic_voiced_f0_hz_mean"], "")
        self.assertGreater(float(row["predicted_pitch_hz_non_sil_mean"]), 0)

    def test_refuses_changed_exact_test_text(self):
        self.records[0]["text"] = "Development fixture only!"
        self.save_manifest()
        with self.assertRaisesRegex(analysis.AnalysisError, "exact original"):
            self.analyze()

    def test_refuses_stale_checkpoint_hash(self):
        self.checkpoint.write_bytes(b"changed development checkpoint")
        with self.assertRaisesRegex(analysis.AnalysisError, "checkpoint hash mismatch"):
            self.analyze()

    def test_refuses_changed_wav_hash(self):
        waveform, rate = sf.read(self.wav)
        sf.write(self.wav, waveform * .5, rate)
        with self.assertRaisesRegex(analysis.AnalysisError, "WAV hash mismatch"):
            self.analyze()

    def test_missing_prediction_fails_without_output(self):
        self.npz.unlink()
        with self.assertRaisesRegex(analysis.AnalysisError, "prediction file IDs"):
            self.analyze()
        self.assertFalse((self.result / "analysis.json").exists())

    def test_zero_frame_prediction_fails(self):
        self.arrays["duration_frames"][:] = 0
        self.arrays["duration_seconds"][:] = 0
        self.arrays["mel"] = np.zeros((0, 80))
        self.save_prediction()
        with self.assertRaisesRegex(analysis.AnalysisError, "all duration frames are zero"):
            self.analyze()

    def test_invalid_shapes_nonfinite_and_duration_conversion_fail(self):
        original = {key: value.copy() for key, value in self.arrays.items()}
        for name, value, message in [
                ("pitch_hz", np.zeros((1, 7)), "numeric vector"),
                ("energy", np.array([np.nan] * 7), "non-finite"),
                ("mel", np.zeros((13, 79)), "mel must be finite"),
                ("duration_seconds", np.zeros(7), "does not match"),
                ("duration_frames", np.array([1, 2, -.5, 3, 4, 2, 1]), "nonnegative integers")]:
            with self.subTest(name=name):
                self.arrays = {key: array.copy() for key, array in original.items()}
                self.arrays[name] = value
                self.save_prediction()
                with self.assertRaisesRegex(analysis.AnalysisError, message):
                    self.analyze()

    def test_duplicate_or_missing_manifest_ids_fail(self):
        for records in ([self.records[0], self.records[0]], []):
            with self.subTest(records=len(records)):
                self.records = records
                self.save_manifest()
                with self.assertRaisesRegex(analysis.AnalysisError, "sample IDs"):
                    self.analyze()

    def test_wav_length_must_match_predicted_mel(self):
        waveform, rate = sf.read(self.wav)
        sf.write(self.wav, waveform[:-1], rate)
        self.records[0]["audio"] = {"sha256": analysis.sha256(self.wav), "frames": len(waveform) - 1,
                                    "seconds": (len(waveform) - 1) / rate}
        self.save_manifest()
        with self.assertRaisesRegex(analysis.AnalysisError, "WAV length differs"):
            self.analyze()

    def test_development_cannot_write_results_and_final_disallows_overrides(self):
        with self.assertRaisesRegex(analysis.AnalysisError, "under Results"):
            analysis.analyze_language(self.root, "English", development=True,
                                      result_dir=self.root / "Results" / "English")
        with self.assertRaisesRegex(analysis.AnalysisError, "overrides require"):
            analysis.analyze_language(self.root, "English", result_dir=self.result)

    def test_cli_failure_leaves_no_analysis(self):
        self.npz.unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            code = analysis.main(["--language", "English", "--development", "--result-dir", str(self.result),
                                  "--checkpoint", str(self.checkpoint), "--samples", str(self.samples),
                                  "--config", str(self.config), "--skip-acoustic-f0"])
        self.assertEqual(code, 1)
        self.assertFalse((self.result / "analysis.json").exists())
        self.assertFalse((self.result / "analysis.csv").exists())

    def test_world_measures_development_sine_separately_from_predictor(self):
        if importlib.util.find_spec("pyworld") is None:
            self.skipTest("Optional pyworld unavailable")
        waveform = .2 * np.sin(2 * np.pi * 220 * np.arange(22050) / 22050)
        result = analysis.acoustic_f0(waveform, 22050, 256)
        self.assertEqual(result["status"], "measured")
        self.assertGreater(result["voiced_fraction"], .5)
        self.assertAlmostEqual(result["voiced_f0_hz"]["median"], 220, delta=5)
        self.assertEqual(result["source"], "decoded generated WAV")

    def test_reads_existing_development_reconstruction_when_available(self):
        paths = sorted(WORK.glob("*/**/*.target_reconstruction.wav"))
        if not paths:
            self.skipTest("Existing real-smoke reconstruction WAV unavailable")
        waveform, measured = analysis.wav_measurements(paths[0], 22050)
        self.assertGreater(measured["sample_count"], 0)
        self.assertTrue(np.isfinite(measured["rms"]))
        self.assertEqual(measured["sample_count"], len(waveform))
        self.assertEqual(measured["sha256"], analysis.sha256(paths[0]))


if __name__ == "__main__":
    unittest.main()
