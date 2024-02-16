"""
The simplest example of a wheelchair moving in a straight line with no holonomic constraints
i.e. no rolling, no closed-loop on the handrim.
"""

import numpy as np

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    BiMappingList,
    BiorbdModel,
    InterpolationType,
    PhaseDynamics,
    SolutionMerge,
)


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
    bio_model = BiorbdModel(biorbd_model_path)

    final_time = 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=0.6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)

    # Path Constraints
    constraints = ConstraintList()

    # Boundaries
    mapping = BiMappingList()
    # mapping.add("q", [0, None, 1, 2], [0, 2, 3])
    # mapping.add("qdot", [0, None, 1, 2], [0, 2, 3])
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")

    # make the wheelchair move in a straight line
    start = 0
    end = 0.5
    x_bounds["q"].min[0, 0] = start
    x_bounds["q"].max[0, 0] = start
    x_bounds["q"].min[0, -1] = end
    x_bounds["q"].max[0, -1] = end

    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(key="q", initial_guess=np.zeros((bio_model.nb_q,)), interpolation=InterpolationType.CONSTANT)
    x_init.add(key="qdot", initial_guess=np.zeros((bio_model.nb_qdot,)), interpolation=InterpolationType.CONSTANT)

    # Define control path constraint
    tau_min, tau_max, tau_init = -50, 50, 0

    variable_bimapping = BiMappingList()
    variable_bimapping.add("tau", to_second=[None, None, 0, 1], to_first=[2, 3])
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * 2)
    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=False,
        variable_mappings=variable_bimapping,
        n_threads=1,
    ), bio_model


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "wheelchair_model.bioMod"
    n_shooting = 50
    ocp, bio_model = prepare_ocp(biorbd_model_path=model_path, n_shooting=n_shooting)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=500))

    # # --- Show results --- #
    # # sol.animate()
    import bioviz
    viz = bioviz.Viz(model_path)
    viz.load_movement(sol.decision_states(to_merge=SolutionMerge.NODES)["q"])
    viz.exec()


if __name__ == "__main__":
    main()
