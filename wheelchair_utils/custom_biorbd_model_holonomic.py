"""
This file is the custom model got from Anais Farr's work that implement explicit holonomic constraints
We need to adapt it to include rolling constraints and our close-loop constraint.

NOTE: Bioptim implentation might have changed a bit since the creation of this file, so we might need an extra effort to
adapt it to the current version of bioptim
"""

import biorbd as biorbd_eigen
import biorbd_casadi as biorbd
import casadi as cas
from biorbd import marker_index, segment_index
from casadi import MX, vertcat, inv

from bioptim import HolonomicBiorbdModel, ConfigureProblem, DynamicsFunctions


class BiorbdModelCustomHolonomic(HolonomicBiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints.
    """

    def __init__(self, bio_model: str | biorbd.Model):
        super().__init__(bio_model)

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

        theta2 = cas.acos((xp**2 + yp**2 - (l2**2 + l1**2)) / (2 * l1 * l2))
        theta1 = cas.atan2(
            (-xp * l2 * cas.sin(theta2) + yp * (l1 + l2 * cas.cos(theta2))),
            (xp * (l1 + l2 * cas.cos(theta2)) + yp * l2 * cas.sin(theta2)),
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
        index_segment_ref = segment_index(self.model, "ContactFrame")
        index_forearm = segment_index(self.model, "Forearm")
        index_marker_handrim = marker_index(self.model, "handrim_contact_point")
        index_marker_hand = marker_index(self.model, "hand")
        #
        # # Find length arm and forearm
        forearm_JCS_trans = self.model.segments()[index_forearm].localJCS().trans().to_mx()
        hand_JCS_trans = self.model.marker(index_marker_hand).to_mx()
        l1 = cas.sqrt(forearm_JCS_trans[0] ** 2 + forearm_JCS_trans[1] ** 2)  # TODO: Maybe problem with square ?
        l2 = cas.sqrt(hand_JCS_trans[0] ** 2 + hand_JCS_trans[1] ** 2)

        v = MX.sym("v", self.nb_dependent_joints - 1)  ## TODO : automatiser
        q = vertcat(0, u, v)
        #
        # # # Matrix RT "Wheel location" (ref)
        R_wheel_global = self.model.globalJCS(q, index_segment_ref).transpose().to_mx()
        # #
        # # # Perform the forward kinematics
        model_eigen = biorbd_eigen.Model(self.model.path().absolutePath().to_string())
        r_wheel = model_eigen.localJCS()[0].to_array()[1, -1]

        markers = self.markers(q)
        marker_handrim_in_mwc_x = markers[index_marker_handrim][0]
        marker_handrim_in_mwc_y = markers[index_marker_handrim][1] - r_wheel - 0.771

        # # Find position dependent joint
        theta = self.inverse_kinematics_2d(
            l1=l1,
            l2=l2,
            xp=marker_handrim_in_mwc_x,
            yp=marker_handrim_in_mwc_y,
        )
        return vertcat(-u * r_wheel, theta)

    @staticmethod
    def holonomic_torque_driven(ocp, nlp, mapping):
        """
        Tell the program which variables are states and controls.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        name = "q_u"
        names_u = [nlp.model.name_dof[i] for i in mapping["q"].to_first.map_idx]
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, names_u, ocp, nlp, True, False, False, axes_idx=axes_idx)

        name = "qdot_u"
        names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
        names_udot = [names_qdot[i] for i in mapping["qdot"].to_first.map_idx]
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, names_udot, ocp, nlp, True, False, False, axes_idx=axes_idx)

        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.holonomic_torque_driven, expand=False)

    def partitioned_forward_dynamics(
        self, q_u, qdot_u, tau, external_forces=None, f_contacts=None, q_v_init=None
    ) -> MX:
        """
        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """
        if external_forces is not None:
            raise NotImplementedError("External forces are not implemented yet.")
        if f_contacts is not None:
            raise NotImplementedError("Contact forces are not implemented yet.")

        # compute q and qdot
        q = self.compute_q(q_u, q_v_init=q_v_init)
        qdot = self.compute_qdot(q, qdot_u)

        partitioned_mass_matrix = self.partitioned_mass_matrix(q)
        m_uu = partitioned_mass_matrix[: self.nb_independent_joints, : self.nb_independent_joints]
        m_uv = partitioned_mass_matrix[: self.nb_independent_joints, self.nb_independent_joints :]
        m_vu = partitioned_mass_matrix[self.nb_independent_joints :, : self.nb_independent_joints]
        m_vv = partitioned_mass_matrix[self.nb_independent_joints :, self.nb_independent_joints :]

        coupling_matrix_vu = self.coupling_matrix(q)
        modified_mass_matrix = (
            m_uu
            + m_uv @ coupling_matrix_vu
            + coupling_matrix_vu.T @ m_vu
            + coupling_matrix_vu.T @ m_vv @ coupling_matrix_vu
        )
        second_term = m_uv + coupling_matrix_vu.T @ m_vv

        # compute the non-linear effect
        non_linear_effect = self.partitioned_non_linear_effect(q, qdot, external_forces, f_contacts)
        non_linear_effect_u = non_linear_effect[: self.nb_independent_joints]
        non_linear_effect_v = non_linear_effect[self.nb_independent_joints :]

        modified_non_linear_effect = non_linear_effect_u + coupling_matrix_vu.T @ non_linear_effect_v

        # compute the tau
        partitioned_tau = self.partitioned_tau(tau)
        tau_u = partitioned_tau[: self.nb_independent_joints]
        tau_v = partitioned_tau[self.nb_independent_joints :]

        modified_generalized_forces = tau_u + coupling_matrix_vu.T @ tau_v

        qddot_u = inv(modified_mass_matrix) @ (
            modified_generalized_forces - second_term @ self.biais_vector(q, qdot) - modified_non_linear_effect
        )

        return qddot_u

    def compute_q(self, q_u: MX, q_v_init: MX = None) -> MX:
        """
        EXPLICIT

        Compute the dependent joint from the independent joint
        and integrates them into the variable q

        The major change here is that we explicitly compute q_v from q_u

        Parameters
        ----------
        q_u: the q state of the independent joint
        q_v_init

        Returns
        -------
        q: states of the dependent and independent joint

        """
        q_v = self.compute_v_from_u_explicit_symbolic(q_u)  # THIS FUNCTION NEED TO BE CHANGED TO OUR APPLICATION !
        return self.state_from_partition(q_u, q_v)
