import unittest
import numpy as np
from src.analyze import summarize_pcm


class AudioFeatureTests(unittest.TestCase):
    def test_silence(self):
        pcm = np.zeros(16000, dtype="<i2").tobytes()
        result = summarize_pcm(pcm)
        self.assertAlmostEqual(result["duration_seconds"], 1.0)
        self.assertEqual(result["median_dbfs"], -100.0)
        self.assertEqual(result["high_energy_fraction"], 0.0)

    def test_tone_is_louder_than_silence(self):
        t = np.arange(16000) / 16000.0
        tone = (0.1 * 32767 * np.sin(2 * np.pi * 200 * t)).astype("<i2")
        result = summarize_pcm(tone.tobytes())
        self.assertGreater(result["median_dbfs"], -40.0)
        self.assertLess(result["median_dbfs"], -15.0)


if __name__ == "__main__":
    unittest.main()
