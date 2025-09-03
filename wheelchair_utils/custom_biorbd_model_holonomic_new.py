"""
This file is the custom model got from Anais Farr's work that implement explicit holonomic constraints
We need to adapt it to include rolling constraints and our close-loop constraint.

NOTE: Bioptim implentation might have changed a bit since the creation of this file, so we might need an extra effort to
adapt it to the current version of bioptim
"""

from typing import Callable, override
from bioptim import HolonomicBiorbdModel, HolonomicTorqueDynamics, HolonomicConstraintsList, ParameterList

import biorbd as biorbd_eigen
from biorbd import segment_index, marker_index
from biorbd_casadi import GeneralizedCoordinates
import numpy as np
from casadi import vertcat, acos, atan2, sin, cos, MX, sqrt, Function

from external.bioptim.models.utils import cache_function


class CustomHolonomicBiorbdModel(HolonomicBiorbdModel):
    def __init__(
        self,
        bio_model: str,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        mwc_phase: str = None,
    ):
        super().__init__(bio_model, friction_coefficients, parameters)
        self.mwc_phase = mwc_phase

    @staticmethod
    def inverse_kinematics_2d(l1, l2, xp, yp):
        """
        Inverse kinematics with elbow down solution.
        Parameters
        ----------
        l1:
            The length of the arm
        l2:
            The length of the forearm
        xp:
            Coordinate on x of the marker of the contact point on handrim in the arm's frame
        yp:
            Coordinate on y of the marker of the contact point on handrim in the arm's frame

        Returns
        -------
        theta:
            The dependent joint
        """

        theta2 = acos((xp**2 + yp**2 - (l2**2 + l1**2)) / (2 * l1 * l2))
        theta1 = atan2(
            (-xp * l2 * sin(theta2) + yp * (l1 + l2 * cos(theta2))),
            (xp * (l1 + l2 * cos(theta2)) + yp * l2 * sin(theta2)),
        )
        return vertcat(theta1, theta2)

    def compute_v_from_u_explicit_symbolic(self, u: MX):
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! symbolic version of the function

        Parameters
        ----------
        u: MX
            The generalized coordinates of independent joint

        Returns
        -------
        theta:
            The angle of the dependent joint

        """
        model_eigen = biorbd_eigen.Model(self.model.path().absolutePath().to_string())
        index_segment_mwc = segment_index(self.model, "Wheelchair")
        r_wheel = model_eigen.localJCS()[index_segment_mwc].to_array()[1, -1]

        if self.mwc_phase == "push_phase":
            index_segment_ref = segment_index(self.model, "ContactFrame")
            index_forearm = segment_index(self.model, "Forearm")
            index_arm = segment_index(self.model, "Arm")
            index_marker_handrim = marker_index(self.model, "handrim_contact_point")
            index_marker_hand = marker_index(self.model, "hand")
            #
            # # Find length arm and forearm
            forearm_JCS_trans = self.model.segments()[index_forearm].localJCS().trans().to_mx()
            hand_JCS_trans = self.model.marker(index_marker_hand).to_mx()
            l1 = sqrt(forearm_JCS_trans[0] ** 2 + forearm_JCS_trans[1] ** 2)
            l2 = sqrt(hand_JCS_trans[0] ** 2 + hand_JCS_trans[1] ** 2)

            v = self.q_v[1:]  # NOTE : all the dependent joints except the translation of the wheelchair
            q = vertcat(0, u, v)

            # markers = self.markers()(q, self.parameters)
            # marker_handrim_in_mwc_x = markers[index_marker_handrim, 0]
            # wheelchair_to_shoulder_translation = model_eigen.localJCS()[index_arm].to_array()[1, -1]
            # marker_handrim_in_mwc_y = markers[index_marker_handrim, 1] - r_wheel - wheelchair_to_shoulder_translation

            # marker_handrim = self.marker(index=index_marker_handrim)(q, self.parameters)

            q_biorbd = GeneralizedCoordinates(q)
            marker_handrim = self.model.marker(q_biorbd, index_marker_handrim).to_mx()
            marker_handrim_in_mwc_x = marker_handrim[0]
            wheelchair_to_shoulder_translation = model_eigen.localJCS()[index_arm].to_array()[1, -1]
            marker_handrim_in_mwc_y = marker_handrim[1] - r_wheel - wheelchair_to_shoulder_translation

            # # Find position dependent joint
            theta = self.inverse_kinematics_2d(
                l1=l1,
                l2=l2,
                xp=marker_handrim_in_mwc_x,
                yp=marker_handrim_in_mwc_y,
            )
            return vertcat(-u * r_wheel, theta)

        else:
            return vertcat(-u[0] * r_wheel)

    @override
    @cache_function
    def compute_q(self) -> MX:
        q_v = self.compute_v_from_u_explicit_symbolic(self.q_u)
        biorbd_return = self.state_from_partition(self.q_u, q_v)
        return Function("compute_q", [self.q_u, self.q_v_init], [biorbd_return], ["q_u", "q_v_init"], ["q"])


class HolonomicTorqueWheelchairModel(CustomHolonomicBiorbdModel, HolonomicTorqueDynamics):
    def __init__(
        self,
        bio_model: str,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        holonomic_constraints: HolonomicConstraintsList | None = None,
        dependent_joint_index: list[int] | tuple[int, ...] = None,
        independent_joint_index: list[int] | tuple[int, ...] = None,
        mwc_phase: str = None,
    ):
        CustomHolonomicBiorbdModel.__init__(self, bio_model, friction_coefficients, parameters, mwc_phase)
        if holonomic_constraints is not None:
            self.set_holonomic_configuration(holonomic_constraints, dependent_joint_index, independent_joint_index)
        HolonomicTorqueDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return HolonomicTorqueWheelchairModel, dict(
            bio_model=self.path, friction_coefficients=self.friction_coefficients
        )
