"""
# todo: Implement a rolling constraint only and do OCP with it.
"""

import numpy as np
from casadi import MX, SX, vertcat, Function, jacobian

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    DynamicsFunctions,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    SolutionMerge,
    DynamicsEvaluation,
    BiMappingList,
    PhaseDynamics,
    HolonomicBiorbdModel,
    HolonomicConstraintsList,
    InterpolationType,
)
from custom_biorbd_model_holonomic import BiorbdModelCustomHolonomic


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


def generate_rolling_joint_constraint(
    biorbd_model: BiorbdModelCustomHolonomic,
    translation_joint_index: int,
    rotation_joint_index: int,
    radius: float = 1,
) -> tuple[Function, Function, Function]:
    """Generate a rolling joint constraint between two joints"""

    # symbolic variables to create the functions
    q_sym = MX.sym("q", biorbd_model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", biorbd_model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", biorbd_model.nb_qdot, 1)

    constraint = q_sym[translation_joint_index] + radius * q_sym[rotation_joint_index]

    constraint_jacobian = jacobian(constraint, q_sym)

    constraint_func = Function(
        "rolling_joint_constraint",
        [q_sym],
        [constraint],
        ["q"],
        ["rolling_joint_constraint"],
    ).expand()

    constraint_jacobian_func = Function(
        "rolling_joint_constraint_jacobian",
        [q_sym],
        [constraint_jacobian],
        ["q"],
        ["rolling_joint_constraint_jacobian"],
    ).expand()

    constraint_double_derivative = (
        constraint_jacobian_func(q_sym) @ q_ddot_sym
        + jacobian(constraint_jacobian_func(q_sym) @ q_dot_sym, q_sym) @ q_dot_sym
    )

    constraint_double_derivative_func = Function(
        "rolling_joint_constraint_double_derivative",
        [q_sym, q_dot_sym, q_ddot_sym],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["rolling_joint_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func


def compute_all_states(sol, bio_model: BiorbdModelCustomHolonomic):
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

    for i, independent_joint_index in enumerate(bio_model.independent_joint_index):
        tau[independent_joint_index, :-1] = controls["tau"][i, :]
    for i, dependent_joint_index in enumerate(bio_model.dependent_joint_index):
        tau[dependent_joint_index, :-1] = controls["tau"][i, :]

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
        q_v_i = bio_model.compute_q_explicit(states["q_u"][:, i]).toarray()
        q[:, i] = bio_model.state_from_partition(states["q_u"][:, i][:, np.newaxis], q_v_i).toarray().squeeze()
        qdot[:, i] = bio_model.compute_qdot(q[:, i], states["qdot_u"][:, i]).toarray().squeeze()
        qddot_u_i = partitioned_forward_dynamics_func(states["q_u"][:, i], states["qdot_u"][:, i], tau[:, i]).toarray()
        qddot[:, i] = bio_model.compute_qddot(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
        lambdas[:, i] = compute_lambdas_func(q[:, i], qdot[:, i], qddot[:, i], tau[:, i]).toarray().squeeze()

    return q, qdot, qddot, lambdas


def prepare_ocp(
    biorbd_model_path: str,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    n_shooting=50,
) -> OptimalControlProgram:
    """

    Prepare the program
    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolverBase
        The type of ode solver used
    n_shooting: int
        The number of shooting points

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModelCustomHolonomic(biorbd_model_path)
    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        key="rolling_joint_constraint",
        constraints_fcn=generate_rolling_joint_constraint,
        biorbd_model=bio_model,
        translation_joint_index=0,
        rotation_joint_index=1,
        radius=0.35,
    )

    holonomic_constraints.add(
        key="holonomic_constraint",
        constraints_fcn=generate_close_loop_constraint,
        biorbd_model=bio_model,
        marker_1="handrim_contact_point",
        marker_2="hand",
        index=slice(0, 2),
        local_frame_index=0,
    )

    bio_model.set_holonomic_configuration(
        constraints_list=holonomic_constraints, independent_joint_index=[1], dependent_joint_index=[0, 2, 3]
    )

    final_time = 1.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=1.5, max_bound=3)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        custom_configure,
        dynamic_function=custom_dynamic,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        # skip_continuity=True,
    )

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    variable_bimapping = BiMappingList()
    variable_bimapping.add("q", to_second=[None, 0, None, None], to_first=[1])
    variable_bimapping.add("qdot", to_second=[None, 0, None, None], to_first=[1])
    x_bounds = BoundsList()
    x_bounds["q_u"] = bio_model.bounds_from_ranges("q", mapping=variable_bimapping)
    x_bounds["qdot_u"] = bio_model.bounds_from_ranges("qdot", mapping=variable_bimapping)

    # Gérer les positions initiales/finales des DDL
    FRM_end_position = 1.2
    r_wheel = 0.35  # TODO : aller récupérer automatiquement dans le .bioMod
    shoulder_flex_start = -np.pi / 4
    shoulder_flex_end = np.pi / 4
    elbow_flex_start = np.pi / 2
    elbow_flex_end = np.pi / 6

    x_bounds["q_u"].min[0, 0] = 0.4
    x_bounds["q_u"].max[0, 0] = 0.4
    x_bounds["q_u"].min[0, -1] = -0.7
    x_bounds["q_u"].max[0, -1] = -0.7
    # x_bounds["q_u"][1, 0] = -np.pi / 2 + shoulder_flex_start
    # x_bounds["q_u"][1, -1] = -np.pi / 2 + shoulder_flex_end
    # x_bounds["q_u"][2, 0] = elbow_flex_start
    # x_bounds["q_u"][2, -1] = elbow_flex_end
    # Vitesses angulaires = 0
    # x_bounds["qdot_u"].min[:, 0] = 0
    # x_bounds["qdot_u"].max[:, 0] = 0
    # x_bounds["qdot_u"].min[:, -1] = 0
    # x_bounds["qdot_u"].max[:, -1] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(key="q_u", initial_guess=np.zeros((1,)), interpolation=InterpolationType.CONSTANT)
    x_init.add(key="qdot_u", initial_guess=np.zeros((1,)), interpolation=InterpolationType.CONSTANT)

    # Define control path constraint
    tau_min, tau_max, tau_init = -50, 50, 1

    variable_bimapping.add("tau", to_second=[None, None, 0, 1], to_first=[2, 3])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2)
    u_bounds["tau"].min[0, 0] = 0
    u_bounds["tau"].max[0, 0] = 0

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=[tau_init] * 2)
    # ------------- #

    return (
        OptimalControlProgram(
            bio_model,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=ode_solver,
            variable_mappings=variable_bimapping,
            use_sx=False,
            n_threads=8,
        ),
        bio_model,
        variable_bimapping,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "wheelchair_model.bioMod"
    n_shooting = 50
    ocp, bio_model, variable_bimapping = prepare_ocp(biorbd_model_path=model_path, n_shooting=n_shooting)
    # ocp.add_plot_penalty(CostType.CONSTRAINTS)
    # --- Solve the program --- #

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=False), _max_iter=500))
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # # --- Show results --- #
    q, qdot, qddot, lambdas = compute_all_states(sol, bio_model, variable_bimapping)
    #
    import bioviz

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()

    import matplotlib.pyplot as plt

    time = sol.decision_time(to_merge=SolutionMerge.NODES)
    plt.title("Lagrange multipliers of the holonomic constraint")
    plt.plot(time, lambdas[0, :], label="rolling")
    plt.plot(time, lambdas[1, :], label="F_x")
    plt.plot(time, lambdas[2, :], label="F_y")
    plt.xlabel("Time (s)")
    plt.ylabel("Lagrange multipliers (N)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
