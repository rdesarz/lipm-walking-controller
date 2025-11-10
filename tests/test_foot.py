import unittest

from biped_walking_controller.foot import WalkingPhase, Params, compute_walking_phase


class TestWalkingPhase(unittest.TestCase):
    def setUp(self):
        self.params = Params()

    def test_begin_double_support(self):
        phase = WalkingPhase.DOUBLE_SUPPORT

        result = compute_walking_phase(phase, 0.0, self.params)

        self.assertIsNone(result)

    def test_switch_to_single_support(self):
        phase = WalkingPhase.DOUBLE_SUPPORT

        result = compute_walking_phase(phase, self.params.t_ds + 0.1, self.params)

        self.assertEqual(result, phase.SINGLE_SUPPORT)

    def test_do_not_switch_to_ds_if_beginning_of_phase(self):
        phase = WalkingPhase.SINGLE_SUPPORT

        result = compute_walking_phase(
            phase, 0.0, self.params, contact_force=self.params.force_threshold + 10
        )

        self.assertIsNone(result)

    def test_do_not_switch_to_ds_if_force_too_low(self):
        phase = WalkingPhase.SINGLE_SUPPORT

        result = compute_walking_phase(
            phase,
            self.params.t_ss * 0.75,
            self.params,
            contact_force=self.params.force_threshold - 10,
        )

        self.assertIsNone(result)

    def test_switch_to_ds_if_force_too_low_and_phase_close_to_end(self):
        phase = WalkingPhase.SINGLE_SUPPORT

        result = compute_walking_phase(
            phase,
            self.params.t_ss * 0.75,
            self.params,
            contact_force=self.params.force_threshold + 10,
        )

        self.assertEqual(result, phase.DOUBLE_SUPPORT)


if __name__ == "__main__":
    unittest.main()
