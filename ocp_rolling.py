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
    HolonomicBiorbdModel,
    CostType,
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

    constraint = q_sym[translation_joint_index] - radius * q_sym[rotation_joint_index]

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
    bio_model = HolonomicBiorbdModel(biorbd_model_path)

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
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

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

    # make the wheelchair move in a straight line
    start = 0
    end = 1
    x_bounds["q"].min[0, 0] = start
    x_bounds["q"].max[0, 0] = start
    x_bounds["q"].min[0, -1] = end
    x_bounds["q"].max[0, -1] = end

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(key="q", initial_guess=[0, 0, 0, 0, 0, 0])

    # Define control path constraint
    tau_min, tau_max, tau_init = -50, 50, 0

    variable_bimapping = BiMappingList()

    # variable_bimapping.add("tau", to_second=[None, None, 1, 2], to_first=[2, 3])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min]*4, max_bound=[tau_max]*4)

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=[tau_init]*4)
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
    ocp.add_plot_penalty(CostType.ALL)
    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _max_iter=500))

    sol.graphs()
    # --- Show results --- #
    # sol.animate()
    q = np.zeros((4, n_shooting + 1))
    for i, ui in enumerate(sol.states["u"].T):
        qi = bio_model.compute_q(ui, q_v_init=np.zeros(1)).toarray()
        print(qi[0])
        q[:, i] = qi.squeeze()

    # q idx 1 stay between 0 and 2pi
    # q[0, :] = 0
    q[1, :] = np.mod(q[1, :], 2 * np.pi)

    import bioviz
    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()


if __name__ == "__main__":
    main()
