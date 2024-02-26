import numpy as np
from casadi import MX, SX, vertcat, Function

from bioptim import (
    OptimalControlProgram,
    ConfigureProblem,
    DynamicsFunctions,
    NonLinearProgram,
    DynamicsEvaluation,
    HolonomicBiorbdModel,
    SolutionMerge,
)
from .custom_biorbd_model_holonomic import BiorbdModelCustomHolonomic


def custom_dynamic(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    stochastic_variables: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    u = DynamicsFunctions.get(nlp.states["q_u"], states)
    udot = DynamicsFunctions.get(nlp.states["qdot_u"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    uddot = nlp.model.partitioned_forward_dynamics(u, udot, tau)

    return DynamicsEvaluation(dxdt=vertcat(udot, uddot), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    name_u = [nlp.model.name_dof[i] for i in range(nlp.model.nb_independent_joints)]
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, "q_u")
    ConfigureProblem.configure_new_variable("q_u", name_u, ocp, nlp, True, False, False, axes_idx=axes_idx)

    name = "qdot_u"
    name_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot_u")
    name_udot = [name_qdot[i] for i in range(nlp.model.nb_independent_joints)]
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
    ConfigureProblem.configure_new_variable(name, name_udot, ocp, nlp, True, False, False, axes_idx=axes_idx)

    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic)


def compute_all_states(sol, bio_model: BiorbdModelCustomHolonomic, tau_bimapping):
    """
    Compute all the states from the solution of the optimal control program

    Parameters
    ----------
    bio_model: HolonomicBiorbdModel
        The biorbd model
    sol:
        The solution of the optimal control program

    Returns
    -------

    """

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    n = states["q_u"].shape[1]

    q = np.zeros((bio_model.nb_q, n))
    qdot = np.zeros((bio_model.nb_q, n))
    qddot = np.zeros((bio_model.nb_q, n))
    lambdas = np.zeros((bio_model.nb_dependent_joints, n))
    tau = np.zeros((bio_model.nb_tau, n))

    ## TODO : en auto et propre
    # for i, independent_joint_index in enumerate(bio_model.independent_joint_index):
    #     tau[independent_joint_index, :-1] = controls["tau"][i, :]
    # ddl_actuated = tau_bimapping["tau"].to_first.map_idx  # parmi ceux là, qui sont les indep.
    # for i, dependent_joint_index in enumerate(bio_model.dependent_joint_index):
    #     tau[dependent_joint_index, :-1] = controls["tau"][i, :]

    tau[2:4, :-1] = controls["tau"][0:2, :]  # on met coude-epaule dans les num associés

    # Partitioned forward dynamics
    q_u_sym = MX.sym("q_u_sym", bio_model.nb_independent_joints, 1)
    qdot_u_sym = MX.sym("qdot_u_sym", bio_model.nb_independent_joints, 1)
    tau_sym = MX.sym("tau_sym", bio_model.nb_tau, 1)
    partitioned_forward_dynamics_func = Function(
        "partitioned_forward_dynamics",
        [q_u_sym, qdot_u_sym, tau_sym],
        [bio_model.partitioned_forward_dynamics(q_u_sym, qdot_u_sym, tau_sym)],
    )
    # Lagrangian multipliers
    q_sym = MX.sym("q_sym", bio_model.nb_q, 1)
    qdot_sym = MX.sym("qdot_sym", bio_model.nb_q, 1)
    qddot_sym = MX.sym("qddot_sym", bio_model.nb_q, 1)
    compute_lambdas_func = Function(
        "compute_the_lagrangian_multipliers",
        [q_sym, qdot_sym, qddot_sym, tau_sym],
        [bio_model.compute_the_lagrangian_multipliers(q_sym, qdot_sym, qddot_sym, tau_sym)],
    )

    for i in range(n):
        q_v_i = bio_model.compute_v_from_u_explicit_symbolic(states["q_u"][:, i])
        q_v_i_function = Function("q_v_i_eval", [], [q_v_i])
        q_v_i = q_v_i_function()["o0"]
        q[:, i] = (
            bio_model.state_from_partition(states["q_u"][:, i][:, np.newaxis], q_v_i).toarray().squeeze()
        )  # TODO : add error si mauvaises dimensions
        qdot[:, i] = bio_model.compute_qdot(q[:, i], states["qdot_u"][:, i]).toarray().squeeze()
        qddot_u_i = partitioned_forward_dynamics_func(states["q_u"][:, i], states["qdot_u"][:, i], tau[:, i]).toarray()
        qddot[:, i] = bio_model.compute_qddot(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
        lambdas[:, i] = compute_lambdas_func(q[:, i], qdot[:, i], qddot[:, i], tau[:, i]).toarray().squeeze()

    return q, qdot, qddot, lambdas
