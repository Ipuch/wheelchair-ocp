"""
# todo: not implemented at all, just a copy of the example
It needs to be a example that simulates the pushing phase of a propulsion cycle of a wheelchair.
"""
from casadi import MX, SX, vertcat, Function, jacobian, sqrt, atan2
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    DynamicsFunctions,
    ParameterList,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    BiMappingList,
    PhaseDynamics,
    HolonomicConstraintsList,
)
from biorbd_casadi import marker_index, segment_index, NodeSegment, Vector3d
import numpy as np

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

    u = DynamicsFunctions.get(nlp.states["u"], states)
    udot = DynamicsFunctions.get(nlp.states["udot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    uddot = nlp.model.forward_dynamics_constrained_independent(u, udot, tau)

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
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, "u")
    ConfigureProblem.configure_new_variable(
        "u", name_u, ocp, nlp, True, False, False, axes_idx=axes_idx
    )

    name = "udot"
    name_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
    name_udot = [name_qdot[i] for i in range(nlp.model.nb_independent_joints)]
    axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
    ConfigureProblem.configure_new_variable(
        name, name_udot, ocp, nlp, True, False, False, axes_idx=axes_idx
    )

    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic)


def generate_close_loop_constraint(
    biorbd_model, marker_1: str, marker_2: str, index: slice = slice(0, 3), local_frame_index: int = None, parameters: MX = MX(),
) -> tuple[Function, Function, Function]:
    """Generate a close loop constraint between two markers"""

    # symbolic variables to create the functions
    q_sym = MX.sym("q", biorbd_model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", biorbd_model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", biorbd_model.nb_qdot, 1)

    # symbolic markers in global frame
    marker_1_sym = biorbd_model.marker(q_sym, index=marker_index(biorbd_model.model, marker_1))
    marker_2_sym = biorbd_model.marker(q_sym, index=marker_index(biorbd_model.model, marker_2))

    # if local frame is provided, the markers are expressed in the same local frame
    if local_frame_index is not None:
        jcs_t = biorbd_model.homogeneous_matrices_in_global(q_sym, local_frame_index, inverse=True)
        marker_1_sym = (jcs_t.to_mx() @ vertcat(marker_1_sym, 1))[:3]
        marker_2_sym = (jcs_t.to_mx() @ vertcat(marker_2_sym, 1))[:3]

    # the constraint is the distance between the two markers, set to zero
    constraint = (marker_1_sym - marker_2_sym)[index]
    # the jacobian of the constraint
    constraint_jacobian = jacobian(constraint, q_sym)

    constraint_func = Function(
        "holonomic_constraint",
        [q_sym, parameters],
        [constraint],
        ["q"],
        ["holonomic_constraint"],
    ).expand()

    constraint_jacobian_func = Function(
        "holonomic_constraint_jacobian",
        [q_sym, parameters],
        [constraint_jacobian],
        ["q"],
        ["holonomic_constraint_jacobian"],
    ).expand()

    # the double derivative of the constraint
    constraint_double_derivative = (
        constraint_jacobian_func(q_sym) @ q_ddot_sym + constraint_jacobian_func(q_dot_sym) @ q_dot_sym
    )

    constraint_double_derivative_func = Function(
        "holonomic_constraint_double_derivative",
        [q_sym, q_dot_sym, q_ddot_sym, parameters],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["holonomic_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func


def generate_close_loop_constraint_polar_coordinates(
        biorbd_model: BiorbdModelCustomHolonomic,
        marker_1: str,
        wheel_frame_index: int = None,
        handrim_radius: float = 0.35,
        contact_angle: float = None,
) -> tuple[Function, Function, Function]:
    """
    Generate a close loop constraint between two markers, in polar coordinates for the wheel.
    In order to get lagrange multipliers as radial and tangential forces applied on the wheel.
    assumed in x-y plane

    Parameters
    ----------
    biorbd_model: BiorbdModelCustomHolonomic
        The biorbd model
    marker_1: str
        The name of the marker to constraint
    wheel_frame_index: int
        The index of the wheel frame
    handrim_radius: float
        The radius of the handrim
    contact_angle: float
        polar angle of the contact point on the wheel, in radian need to be symbolic to be able to find the best location.
    **extra_params, optional
        Extra parameters to pass to the constraint function
    """

    # symbolic variables to create the functions
    q_sym = MX.sym("q", biorbd_model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", biorbd_model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", biorbd_model.nb_qdot, 1)

    # symbolic markers in global frame
    marker_1_sym = biorbd_model.marker(q_sym, index=marker_index(biorbd_model.model, marker_1))

    # if local frame is provided, the markers are expressed in the same local frame
    if wheel_frame_index is not None:
        jcs_t = biorbd_model.homogeneous_matrices_in_global(q_sym, wheel_frame_index, inverse=True)
        marker_1_sym = (jcs_t.to_mx() @ vertcat(marker_1_sym, 1))[:3]

    # express in polar coordinates
    radial_constraint = sqrt((marker_1_sym[0]) ** 2 + (marker_1_sym[1]) ** 2) - handrim_radius
    tangential_constraint = atan2(marker_1_sym[1], marker_1_sym[0]) - contact_angle

    # the constraint
    constraint = vertcat(radial_constraint, tangential_constraint)
    # the jacobian of the constraint
    constraint_jacobian = jacobian(constraint, q_sym)

    constraint_func = Function(
        "holonomic_constraint",
        [q_sym, parameters],
        [constraint],
        ["q"],
        ["holonomic_constraint"],
    ).expand()

    constraint_jacobian_func = Function(
        "holonomic_constraint_jacobian",
        [q_sym, parameters],
        [constraint_jacobian],
        ["q"],
        ["holonomic_constraint_jacobian"],
    ).expand()

    # the double derivative of the constraint
    constraint_double_derivative = (
        constraint_jacobian_func(q_sym) @ q_ddot_sym + jacobian(constraint_jacobian_func(q_sym) @ q_dot_sym, q_sym) @ q_dot_sym
    )

    constraint_double_derivative_func = Function(
        "holonomic_constraint_double_derivative",
        [q_sym, q_dot_sym, q_ddot_sym, parameters],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["holonomic_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func


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
        constraint_jacobian_func(q_sym) @ q_ddot_sym + jacobian(constraint_jacobian_func(q_sym) @ q_dot_sym, q_sym) @ q_dot_sym
    )

    constraint_double_derivative_func = Function(
        "rolling_joint_constraint_double_derivative",
        [q_sym, q_dot_sym, q_ddot_sym],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["rolling_joint_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func


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

    bio_model.set_holonomic_configuration(constraints_list=holonomic_constraints,
                                          independent_joint_index=[0, 2, 3],
                                          dependent_joint_index=[1]
                                          )

    # constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
    #     biorbd_model=bio_model,
    #     marker_1="marker_4",
    #     marker_2="handrim_contact",
    #     local_frame_index=0,
    #     parameters=contact_angle,
    # )

    parameters = ParameterList()

    # def set_contact_angle(biomodel: BiorbdModelCustomHolonomic, contact_angle: MX):
    #     """Set the contact angle of the wheel"""
    #     biomodel.add_extra_parameter("contact_angle", contact_angle)
    #
    # parameters.add(
    #     parameter_name="contact_angle",  # the polar angle of the contact point in the wheel frame
    #     function=set_contact_angle,
    #     initial_guess=InitialGuess(0),
    #     bounds=Bounds(-2*np.pi, 2*np.pi),
    #     size=1,
    # )

    # bio_model.set_dependencies(independent_joint_index=[0, 2, 3], dependent_joint_index=[1])

    final_time = 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    mapping = BiMappingList()
    mapping.add("q", to_second=[0, None, 1, 2], to_first=[0, 2, 3])
    mapping.add("qdot", to_second=[0, None, 1, 2], to_first=[0, 2, 3])
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q", mapping=mapping)
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot", mapping=mapping)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(key="q", initial_guess=[0, 0, 0, 0, 0, 0])

    # Define control path constraint
    tau_min, tau_max, tau_init = -50, 50, 0

    variable_bimapping = BiMappingList()

    variable_bimapping.add("tau", to_second=[None, None, 0, 1], to_first=[2, 3])
    u_bounds = BoundsList()
    u_bounds.add("tau", min=[tau_min]*2, max=[tau_max]*2)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init]*2)
    # ------------- #

    return OptimalControlProgram(
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
        use_sx=False,
        variable_mappings=variable_bimapping,
        parameters=parameters,
        n_threads=8,
    ) , bio_model


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "wheelchair_model.bioMod"
    n_shooting = 50
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=n_shooting)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=500))

    # --- Show results --- #
    # sol.animate()
    q = np.zeros((4, n_shooting + 1))
    for i, ui in enumerate(sol.states["u"].T):
        vi = bio_model.compute_v_from_u_numeric(ui, v_init=np.zeros(2)).toarray()
        qi = bio_model.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi

    import bioviz
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()


if __name__ == "__main__":
    main()
