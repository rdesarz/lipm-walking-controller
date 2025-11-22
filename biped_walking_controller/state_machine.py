from dataclasses import dataclass
from enum import Enum


class WalkingState(Enum):
    """Enumeration of the high-level walking phases."""

    INIT = 0  # Initialization (pre-walk)
    DS = 1  # Double support phase
    SS_LEFT = 2  # Single support, left foot stance
    SS_RIGHT = 3  # Single support, right foot stance
    END = 4  # End of walking sequence


@dataclass
class WalkingStateMachineParams:
    """Timing and contact parameters for the walking state machine.

    Attributes
    ----------
    t_init : float
        Duration of the initial phase before starting walking [s].
    t_end : float
        Duration of the final phase after the last step [s].
    t_ss : float
        Nominal duration of a single support phase [s].
    t_ds : float
        Nominal duration of a double support phase [s].
    force_threshold : float
        Contact force threshold used to detect foot contact [N].
    """

    t_init: float = 2.0  # [s]
    t_end: float = 2.0  # [s]
    t_ss: float = 0.8  # [s]
    t_ds: float = 0.3  # [s]
    force_threshold: float = 50  # [N]


class WalkingStateMachine:
    """Finite state machine controlling walking phases.

    The state machine advances through INIT, DS, SS_LEFT, SS_RIGHT, and END
    based on elapsed time and measured foot contact forces.
    """

    def __init__(
        self,
        params: WalkingStateMachineParams,
        initial_state: WalkingState = WalkingState.INIT,
    ):
        """Initialize the walking state machine.

        Parameters
        ----------
        params : WalkingStateMachineParams
            Timing and contact parameters for the state machine.
        initial_state : WalkingState, optional
            Initial state of the machine, by default WalkingState.INIT.
        """
        self.params = params
        self.state = initial_state
        self.t_start = 0.0
        self.steps = None
        self.steps_foot = None
        self.step_idx = None

    def update_steps(self, steps_pose, steps_foot):
        """Set the planned sequence of steps.

        Parameters
        ----------
        steps_pose :
            Container describing the planned footsteps (e.g. list/array of poses).
            Only its length is used here to know when the last step is reached.
            :param steps_foot:
        """
        self.steps = steps_pose
        self.steps_foot = steps_foot
        self.step_idx = 0

    def update(
        self,
        t: float,
        rf_contact_force: float,
        lf_contact_force: float,
    ):
        """Update the state machine given time and contact forces.

        Parameters
        ----------
        t : float
            Current time [s].
        rf_contact_force : float
            Measured contact force under the right foot [N].
        lf_contact_force : float
            Measured contact force under the left foot [N].

        Notes
        -----
        This method updates the internal state of the machine in-place.
        If no steps have been provided with `update_steps`, it returns immediately.
        """
        if self.steps is None:
            return

        delta_t = self.get_elapsed_time_in_state(t)

        if self.state == WalkingState.INIT:
            self._try_transition_to_single_support(t, self.params.t_init)

        elif self.state == WalkingState.DS:
            self._try_transition_to_single_support(t, self.params.t_ds)

        elif self.state == WalkingState.SS_RIGHT:
            self._try_transition_to_ds_or_end(t, lf_contact_force)

        elif self.state == WalkingState.SS_LEFT:
            self._try_transition_to_ds_or_end(t, rf_contact_force)

        elif self.state == WalkingState.END:
            if delta_t > self.params.t_end:
                self.steps = None

    def get_current_state(self) -> WalkingState:
        """Return the current walking state.

        Returns
        -------
        WalkingState
            Current state of the finite state machine.
        """
        return self.state

    def get_step_idx(self) -> int:
        return self.step_idx

    def get_elapsed_time_in_state(self, t) -> float:
        return t - self.t_start

    def _switch_single_support_leg(self, t: float):
        """Switch to the next single-support state and reset the phase timer.

        Parameters
        ----------
        t : float
            Current time [s], used as the new phase start time.
        """
        self.t_start = t
        self.state = (
            WalkingState.SS_RIGHT
            if self.steps_foot[self.step_idx] is Foot.RIGHT
            else WalkingState.SS_LEFT
        )

    def _try_transition_to_single_support(self, t: float, duration: float):
        """Transition from INIT/DS to the next single-support state if duration elapsed.

        Parameters
        ----------
        t : float
            Current time [s].
        duration : float
            Required time spent in the current phase before switching [s].
        """
        if t - self.t_start > duration:
            self._switch_single_support_leg(t)

    def _is_last_step(self) -> bool:
        """Check whether the current step is the last one.

        Returns
        -------
        bool
            True if the current step index corresponds to the last planned step,
            False otherwise.
        """
        return self.step_idx == len(self.steps) - 2

    def _try_transition_to_ds_or_end(self, t: float, contact_force: float):
        """Transition from single support to double support or END.

        Parameters
        ----------
        t : float
            Current time [s].
        contact_force : float
            Measured contact force of the swing foot [N].

        Notes
        -----
        The transition is triggered when:
        - at least half of `t_ss` has elapsed since `t_start`, and
        - the swing foot contact force exceeds `force_threshold`.

        If the current step is the last one, the next state is END.
        Otherwise the state transitions to DS and the step index is incremented.
        """
        if (
            t - self.t_start > 0.5 * self.params.t_ss
            and contact_force > self.params.force_threshold
        ):
            self.t_start = t
            self.state = WalkingState.END if self._is_last_step() else WalkingState.DS
            self.step_idx += 1


class Foot(Enum):
    RIGHT = 1
    LEFT = 2
