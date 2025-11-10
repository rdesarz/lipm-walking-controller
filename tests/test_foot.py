import unittest

from biped_walking_controller.foot import WalkingPhase, Params, compute_walking_phase


class TestWalkingPhase(unittest.TestCase):
    def setUp(self):
        self.params = Params()

    def test_begin_double_support(self):
        phase = WalkingPhase.DOUBLE_SUPPORT

        # When we reach the time threshold, we switch to a
        result = compute_walking_phase(phase, 0.0, self.params)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
