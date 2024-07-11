"""
This example uses bioptim to generate and solve an optimal control problem with a simple wheelchair/user 2D model with the following assumptions:
- only the upper limbs are actuated (shoulder, elbow)
- closed-loop constraint implemented to maintain hand on handrim (push phase)
- wheel angle = only independent DoF piloting others (mwc translation + arm angles through inverse kinematics)
"""

import numpy as np

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
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
    PhaseTransitionList,
)
from wheelchair_utils.custom_biorbd_model_holonomic import BiorbdModelCustomHolonomic
from wheelchair_utils.dynamics import compute_all_states_from_indep_qu
from wheelchair_utils.dynamics import holonomic_torque_driven_state_space_dynamics, configure_holonomic_torque_driven
from wheelchair_utils.holonomic_constraints import generate_close_loop_constraint, generate_rolling_joint_constraint
from wheelchair_utils.phase_transitions import custom_phase_transition_post


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
    bio_model = (
        BiorbdModelCustomHolonomic(biorbd_model_path, "push_phase"),
        BiorbdModelCustomHolonomic(biorbd_model_path, "recovery_phase"),  # remettre le marqueur au bon endroit
    )

    # Constraints
    holonomic_constraints_push = HolonomicConstraintsList()
    holonomic_constraints_push.add(
        key="rolling_joint_constraint",
        constraints_fcn=generate_rolling_joint_constraint,
        biorbd_model=bio_model[0],
        translation_joint_index=0,
        rotation_joint_index=1,
        radius=0.35,
    )

    holonomic_constraints_push.add(
        key="holonomic_constraint",
        constraints_fcn=generate_close_loop_constraint,
        biorbd_model=bio_model[0],
        marker_1="handrim_contact_point",
        marker_2="hand",
        index=slice(0, 2),
        local_frame_index=1,
    )

    holonomic_constraints_recovery = HolonomicConstraintsList()
    holonomic_constraints_recovery.add(
        key="rolling_joint_constraint",
        constraints_fcn=generate_rolling_joint_constraint,
        biorbd_model=bio_model[1],
        translation_joint_index=0,
        rotation_joint_index=1,
        radius=0.35,
    )

    bio_model[0].set_holonomic_configuration(
        constraints_list=holonomic_constraints_push, independent_joint_index=[1], dependent_joint_index=[0, 2, 3]
    )

    bio_model[1].set_holonomic_configuration(
        constraints_list=holonomic_constraints_recovery, independent_joint_index=[1, 2, 3], dependent_joint_index=[0]
    )

    # Objective functions
    final_time = (1.5, 2.2)
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot_u", weight=100, multi_thread=False, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, multi_thread=False, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=1.3, max_bound=1.8, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=2, max_bound=3, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_holonomic_torque_driven,
        dynamic_function=holonomic_torque_driven_state_space_dynamics,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        phase=0,
    )
    dynamics.add(
        configure_holonomic_torque_driven,
        dynamic_function=holonomic_torque_driven_state_space_dynamics,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        phase=1,
    )

    # Path Constraints
    constraints = ConstraintList()
    variable_bimapping_push = BiMappingList()
    variable_bimapping_push.add("q", to_second=[None, 0, None, None], to_first=[1])
    variable_bimapping_push.add("qdot", to_second=[None, 0, None, None], to_first=[1])

    variable_bimapping_recovery = BiMappingList()
    variable_bimapping_recovery.add("q", to_second=[None, 0, 1, 2], to_first=[1, 2, 3])
    variable_bimapping_recovery.add("qdot", to_second=[None, 0, 1, 2], to_first=[1, 2, 3])

    x_bounds = BoundsList()
    x_bounds.add("q_u", bounds=bio_model[0].bounds_from_ranges("q", mapping=variable_bimapping_push), phase=0)
    x_bounds.add("qdot_u", bounds=bio_model[0].bounds_from_ranges("qdot", mapping=variable_bimapping_push), phase=0)

    x_bounds.add("q_u", bounds=bio_model[1].bounds_from_ranges("q", mapping=variable_bimapping_recovery), phase=1)
    x_bounds.add("qdot_u", bounds=bio_model[1].bounds_from_ranges("qdot", mapping=variable_bimapping_recovery), phase=1)

    # Gérer les positions initiales/finales des DDL : wheel angle
    x_bounds[0]["q_u"].min[0, 0] = 0.3
    x_bounds[0]["q_u"].max[0, 0] = 0.5
    x_bounds[0]["q_u"].min[0, -1] = -0.7
    x_bounds[0]["q_u"].max[0, -1] = -0.4

    # x_bounds[1]["q_u"].min[0, 0] = -0.6
    # x_bounds[1]["q_u"].max[0, 0] = -0.6
    x_bounds[1]["q_u"].min[0, -1] = -5
    x_bounds[1]["q_u"].max[0, -1] = -4
    x_bounds[1]["q_u"].min[1, -1] = -2.2
    x_bounds[1]["q_u"].max[1, -1] = -2.1
    x_bounds[1]["q_u"].min[2, -1] = 0.69
    x_bounds[1]["q_u"].max[2, -1] = 0.71

    # Vitesses angulaires = 0
    x_bounds[0]["qdot_u"].min[:, 0] = 0
    x_bounds[0]["qdot_u"].max[:, 0] = 0
    x_bounds[0]["qdot_u"].max[:, -1] = -1
    x_bounds[0]["qdot_u"].min[:, -1] = -10

    # x_bounds[1]["qdot_u"].min[:, 0] = 0
    # x_bounds[1]["qdot_u"].max[:, 0] = 0
    # x_bounds[1]["qdot_u"].min[:, -1] = -0.1
    # x_bounds[1]["qdot_u"].max[:, -1] = 0.1

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(key="q_u", initial_guess=np.zeros((1,)), interpolation=InterpolationType.CONSTANT, phase=0)
    x_init.add(key="qdot_u", initial_guess=np.zeros((1,)), interpolation=InterpolationType.CONSTANT, phase=0)
    x_init.add(key="q_u", initial_guess=np.zeros((3,)), interpolation=InterpolationType.CONSTANT, phase=1)
    x_init.add(key="qdot_u", initial_guess=np.zeros((3,)), interpolation=InterpolationType.CONSTANT, phase=1)

    # Define control path constraint
    tau_min, tau_max, tau_init = -50, 50, 0

    variable_bimapping_ocp = BiMappingList()
    variable_bimapping_ocp.add("tau", to_second=[None, None, 0, 1], to_first=[2, 3], phase=0)
    variable_bimapping_ocp.add("tau", to_second=[None, None, 0, 1], to_first=[2, 3], phase=1)
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * 2, max_bound=[tau_max] * 2, phase=1)
    u_bounds[0]["tau"].min[0, 0] = 0
    u_bounds[0]["tau"].max[0, 0] = 0
    # u_bounds[1]["tau"].min[0, 0] = 0

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=[tau_init] * 2, phase=0)
    u_init.add("tau", initial_guess=[tau_init] * 2, phase=1)

    phase_transition = PhaseTransitionList()
    phase_transition.add(custom_phase_transition_post, phase_pre_idx=0)
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
            variable_mappings=variable_bimapping_ocp,
            phase_transitions=phase_transition,
            use_sx=False,
            n_threads=8,
        ),
        bio_model,
        variable_bimapping_ocp,
    )


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "models/wheelchair_model.bioMod"
    n_shooting = (20, 20)
    ocp, bio_model, variable_bimapping = prepare_ocp(biorbd_model_path=model_path, n_shooting=n_shooting)
    # ocp.add_plot_penalty(CostType.CONSTRAINTS)
    # --- Solve the program --- #

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=False), _max_iter=500))
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # # --- Show results --- #
    q, qdot, qddot, lambdas = compute_all_states_from_indep_qu(sol, bio_model, variable_bimapping)
    #
    q_cycle = np.hstack(q)

    import bioviz

    viz = bioviz.Viz(model_path)
    viz.load_movement(q_cycle)
    viz.exec()

    # import matplotlib.pyplot as plt
    #
    # time = sol.decision_time(to_merge=SolutionMerge.NODES)
    # fig, axs = plt.subplots(1, 3)
    #
    # axs[0].plot(controls["tau"][0, :], label=r"$\tau_{shoulder}$")
    # axs[0].plot(controls["tau"][1, :], label=r"$\tau_{elbow}$")
    # axs[0].set_title("Controls of the OCP - actuated DoF")
    # axs[0].set_ylabel("Torque (N.m)")
    # axs[0].legend()
    #
    # axs[1].plot(time, qdot[2, :], "o", label=r"$\dot{theta}_{shoulder}$")
    # axs[1].plot(time, qdot[-1, :], "o", label=r"$\dot{theta}_{elbow}$")
    # axs[1].set_title("Controls of the OCP - actuated DoF")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Torque (N.m)")
    # axs[1].legend()
    #
    # axs[2].plot(time, lambdas[0, :], label="rolling")
    # axs[2].plot(time, lambdas[1, :], label=r"$F_r$")
    # axs[2].plot(time, lambdas[2, :], label=r"$F_{\theta}$")
    # axs[2].set_title("Lagrange multipliers of the holonomic constraint")
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Lagrange multipliers (N)")
    # axs[2].legend()

    # plt.show()


if __name__ == "__main__":
    main()
