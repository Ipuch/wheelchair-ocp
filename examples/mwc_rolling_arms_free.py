"""
This example uses bioptim to generate and solve an optimal control problem with a simple wheelchair/user 2D model with the following assumptions:
- there is a torque around the wheel axle
- the upper limbs are actuated (shoulder, elbow) but no constraint is given (i.e. no handrim contact and nothing in cost function)
"""

import numpy as np

from bioptim import (
    OptimalControlProgram,
    DynamicsOptionsList,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    SolutionMerge,
    BiMappingList,
    PhaseDynamics,
    HolonomicConstraintsList,
    InterpolationType,
)
from wheelchair_utils import (
    configure_holonomic_torque_driven,
    generate_rolling_joint_constraint,
    holonomic_torque_driven_state_space_dynamics,
    compute_all_states_from_indep_qu,
)

from wheelchair_utils.custom_biorbd_model_holonomic_new import HolonomicTorqueWheelchairModel


def prepare_ocp(
    biorbd_model_path: str,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    n_shooting=50,
    final_time=1.5,
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

    holonomic_constraints = HolonomicConstraintsList()
    holonomic_constraints.add(
        key="rolling_joint_constraint",
        constraints_fcn=generate_rolling_joint_constraint,
        translation_joint_index=0,
        rotation_joint_index=1,
        radius=0.35,
    )

    bio_model = HolonomicTorqueWheelchairModel(
        biorbd_model_path,
        holonomic_constraints=holonomic_constraints,
        independent_joint_index=[1, 2, 3],
        dependent_joint_index=[0],
        mwc_phase="free",
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=1.5, max_bound=3)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(
        ode_solver=ode_solver,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
    )

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    variable_bimapping = BiMappingList()
    variable_bimapping.add("q", to_second=[None, 0, 1, 2], to_first=[1, 2, 3])
    variable_bimapping.add("qdot", to_second=[None, 0, 1, 2], to_first=[1, 2, 3])
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

    x_bounds["q_u"][0, 0] = 0
    x_bounds["q_u"][0, -1] = -FRM_end_position / r_wheel
    x_bounds["q_u"][1, 0] = -np.pi / 2 + shoulder_flex_start
    x_bounds["q_u"][1, -1] = -np.pi / 2 + shoulder_flex_end
    x_bounds["q_u"][2, 0] = elbow_flex_start
    x_bounds["q_u"][2, -1] = elbow_flex_end
    # Vitesses angulaires = 0
    x_bounds["qdot_u"].min[:, 0] = 0
    x_bounds["qdot_u"].max[:, 0] = 0
    x_bounds["qdot_u"].min[:, -1] = 0
    x_bounds["qdot_u"].max[:, -1] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(key="q_u", initial_guess=np.zeros((3,)), interpolation=InterpolationType.CONSTANT)
    x_init.add(key="qdot_u", initial_guess=np.zeros((3,)), interpolation=InterpolationType.CONSTANT)

    # Define control path constraint
    tau_min, tau_max, tau_init = -50, 50, 1

    # variable_bimapping.add("tau", to_second=[None, 0, 1, 2], to_first=[1, 2, 3])
    variable_bimapping.add("tau", to_second=[None, None, 0, 1], to_first=[2, 3], phase=0)

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 3, max_bound=[tau_max] * 3)
    u_bounds["tau"].min[0, 0] = 0
    u_bounds["tau"].max[0, 0] = 0

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=[tau_init] * 3)
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

    model_path = "models/wheelchair_model.bioMod"
    n_shooting = 50
    ocp, bio_model, variable_bimapping = prepare_ocp(
        biorbd_model_path=model_path, n_shooting=n_shooting, final_time=1.5
    )
    # ocp.add_plot_penalty(CostType.CONSTRAINTS)
    # --- Solve the program --- #

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=False), _max_iter=500))
    sol.graphs()
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # # --- Show results --- #
    q, qdot, qddot, lambdas = compute_all_states_from_indep_qu(sol, bio_model, variable_bimapping)
    #
    from pyorerun import PhaseRerun, BiorbdModel

    # building some time components
    nb_seconds = 1
    t_span = np.linspace(0, nb_seconds, n_shooting + 1)

    # loading biorbd model
    biorbd_model = BiorbdModel(model_path)

    rerun_biorbd = PhaseRerun(t_span)
    rerun_biorbd.add_animated_model(biorbd_model, q[0])
    rerun_biorbd.rerun("animation")


if __name__ == "__main__":
    main()
