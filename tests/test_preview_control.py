# tests/test_preview_control_unittest.py
import unittest
import numpy as np
from lipm_walking_controller.preview_control import (
    PreviewControllerParams,
    compute_preview_control_matrices,
)

from numpy.linalg import eigvals


class TestPreviewControl(unittest.TestCase):
    def setUp(self):
        self.dt = 0.01
        self.params = PreviewControllerParams(
            zc=0.8,
            g=9.81,
            Qe=1.0,
            Qx=np.diag([1e-6, 1e-6, 1e-6]),
            R=np.array([[1e-4]]),
            n_preview_steps=150,
        )

    def test_shapes_and_finiteness(self):
        mats = compute_preview_control_matrices(self.params, self.dt)

        self.assertEqual(mats.A.shape, (3, 3))
        self.assertEqual(mats.B.shape, (3, 1))
        self.assertEqual(mats.C.shape, (1, 3))

        # Gi scalar
        self.assertTrue(np.isscalar(mats.Gi) or np.array(mats.Gi).shape == ())
        self.assertEqual(mats.Gx.shape, (1, 3))
        self.assertEqual(mats.Gd.shape, (self.params.n_preview_steps - 1,))

        for arr in [mats.A, mats.B, mats.C, mats.Gx, mats.Gd]:
            self.assertTrue(np.all(np.isfinite(arr)))

        self.assertTrue(np.isfinite(float(mats.Gi)))

    def test_first_preview_gain_equals_minus_Gi(self):
        mats = compute_preview_control_matrices(self.params, self.dt)

        self.assertAlmostEqual(mats.Gd[0], -float(mats.Gi), places=12)

    def test_min_preview_steps(self):
        p = PreviewControllerParams(
            zc=0.8, g=9.81, Qe=1.0, Qx=np.eye(3) * 1e-6, R=np.array([[1e-4]]), n_preview_steps=2
        )

        mats = compute_preview_control_matrices(p, self.dt)

        self.assertEqual(mats.Gd.shape, (1,))
        self.assertTrue(np.isfinite(mats.Gd[0]))

    def test_closed_loop_stability(self):
        # Rebuild internal terms and check spectral radius(Ac) < 1
        dt = self.dt
        zc, g = self.params.zc, self.params.g

        A = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=float)
        B = np.array([[dt**3 / 6.0], [dt**2 / 2.0], [dt]], dtype=float)
        C = np.array([[1.0, 0.0, -zc / g]], dtype=float)

        A1 = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
        B1 = np.vstack((C @ B, B))
        Q = np.block([[self.params.Qe, np.zeros((1, 3))], [np.zeros((3, 1)), self.params.Qx]])
        R = self.params.R

        from scipy.linalg import solve_discrete_are

        K = solve_discrete_are(A1, B1, Q, R)
        inv_term = np.linalg.inv(R + B1.T @ K @ B1)
        Ac = A1 - B1 @ inv_term @ (B1.T @ K @ A1)

        rho = float(np.max(np.abs(eigvals(Ac))))

        self.assertLess(rho, 1.0)

    def test_gains_change_with_Qe(self):
        base = compute_preview_control_matrices(self.params, self.dt)

        hiQe = PreviewControllerParams(
            zc=self.params.zc,
            g=self.params.g,
            Qe=1e3,
            Qx=self.params.Qx,
            R=self.params.R,
            n_preview_steps=self.params.n_preview_steps,
        )
        alt = compute_preview_control_matrices(hiQe, self.dt)

        self.assertFalse(np.allclose(base.Gx, alt.Gx))
        self.assertFalse(np.isclose(float(base.Gi), float(alt.Gi)))
        self.assertFalse(np.allclose(base.Gd, alt.Gd))


if __name__ == "__main__":
    unittest.main()
