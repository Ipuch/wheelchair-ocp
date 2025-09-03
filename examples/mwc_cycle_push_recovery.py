"""
This example uses bioptim to generate and solve an optimal control problem with a simple wheelchair/user 2D model with the following assumptions:
- only the upper limbs are actuated (shoulder, elbow)
- closed-loop constraint implemented to maintain hand on handrim (push phase)
- wheel angle = only independent DoF piloting others (mwc translation + arm angles through inverse kinematics)
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
    PhaseTransitionList,
    CostType,
)

from external.bioptim import OnlineOptim
from wheelchair_utils import compute_all_states_from_indep_qu
from wheelchair_utils.custom_biorbd_model_holonomic_new import HolonomicTorqueWheelchairModel

from wheelchair_utils.holonomic_constraints import generate_close_loop_constraint, generate_rolling_joint_constraint
from wheelchair_utils.phase_transitions import custom_phase_transition_post


def prepare_ocp(
    biorbd_model_path: str,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    n_shooting=50,
    final_time=(1.5, 0.8),
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
    final_time
        The time at the end of each phase

    Returns
    -------
    The ocp ready to be solved
    """
    holonomic_constraints_push = HolonomicConstraintsList()
    holonomic_constraints_push.add(
        key="rolling_joint_constraint",
        constraints_fcn=generate_rolling_joint_constraint,
        translation_joint_index=0,
        rotation_joint_index=1,
        radius=0.35,
    )
    holonomic_constraints_push.add(
        key="holonomic_constraint",
        constraints_fcn=generate_close_loop_constraint,
        marker_1="handrim_contact_point",
        marker_2="hand",
        index=slice(0, 2),
        local_frame_index=1,
    )

    holonomic_constraints_recovery = HolonomicConstraintsList()
    holonomic_constraints_recovery.add(
        key="rolling_joint_constraint",
        constraints_fcn=generate_rolling_joint_constraint,
        translation_joint_index=0,
        rotation_joint_index=1,
        radius=0.35,
    )

    bio_model = (
        HolonomicTorqueWheelchairModel(
            biorbd_model_path,
            mwc_phase="push_phase",
            holonomic_constraints=holonomic_constraints_push,
            independent_joint_index=[1],  # wheel angle
            dependent_joint_index=[0, 2, 3],  # x translation + arm
        ),
        HolonomicTorqueWheelchairModel(
            biorbd_model_path,
            mwc_phase="recovery_phase",
            holonomic_constraints=holonomic_constraints_recovery,
            independent_joint_index=[1, 2, 3],  # wheel angle + arm
            dependent_joint_index=[0],  # x translation (rolling constraint
        ),  # remettre le marqueur au bon endroit
    )

    # Objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, multi_thread=False, phase=1)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot_u", weight=100, multi_thread=False, phase=1)

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=1, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, min_bound=0.5, max_bound=2, phase=1)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver=ode_solver,
        phase=0,
    )
    dynamics.add(
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        ode_solver=ode_solver,
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
    # x_bounds[0]["q_u"].min[0, 0] = 0
    # x_bounds[0]["q_u"].max[0, 0] = 0
    x_bounds[0]["q_u"].min[0, 0] = 0.3
    x_bounds[0]["q_u"].max[0, 0] = 0.5

    # x_bounds[1]["q_u"].min[0, 0] = -0.6
    # x_bounds[1]["q_u"].max[0, 0] = -0.6
    x_bounds[1]["q_u"].min[0, -1] = -7
    x_bounds[1]["q_u"].max[0, -1] = -4
    x_bounds[1]["q_u"].min[1, -1] = -2.7
    x_bounds[1]["q_u"].max[1, -1] = -2.2
    x_bounds[1]["q_u"].min[2, -1] = 0.6
    x_bounds[1]["q_u"].max[2, -1] = 0.8

    # Vitesses angulaires = 0
    x_bounds[0]["qdot_u"].min[:, 0] = 0
    x_bounds[0]["qdot_u"].max[:, 0] = 0
    x_bounds[0]["qdot_u"].max[:, -1] = 2
    x_bounds[0]["qdot_u"].min[:, -1] = -2

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
            variable_mappings=variable_bimapping_ocp,
            phase_transitions=phase_transition,
            use_sx=False,
            n_threads=15,
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
    final_time = (1.5, 0.8)
    ocp, bio_model, variable_bimapping = prepare_ocp(
        biorbd_model_path=model_path,
        n_shooting=n_shooting,
        final_time=final_time,
    )
    # ocp.add_plot_penalty(CostType.CONSTRAINTS)

    # --- Solve the program --- #
    sol = ocp.solve(
        Solver.IPOPT(
            # show_online_optim=OnlineOptim.SERVER,
            # show_online_optim=OnlineOptim.MULTIPROCESS_SERVER,
            # show_online_optim=OnlineOptim.DEFAULT,
            show_options=dict(show_bounds=True),
            _max_iter=200,
        ),
    )
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # # --- Show results --- #
    q, qdot, qddot, lambdas = compute_all_states_from_indep_qu(sol, bio_model, variable_bimapping)
    q_cycle = np.hstack(q)

    from pyorerun import MultiPhaseRerun, BiorbdModel

    # building some time components
    t_span_0 = np.linspace(0, final_time[0], n_shooting[0] + 1)
    t_span_1 = np.linspace(final_time[0], final_time[0] + final_time[1], n_shooting[1] + 1)

    # loading biorbd model
    biorbd_model = BiorbdModel(model_path)

    multi_phase_rerun = MultiPhaseRerun()
    multi_phase_rerun.add_phase(sol.decision_time(to_merge=SolutionMerge.NODES)[0], phase=0)
    multi_phase_rerun.add_phase(sol.decision_time(to_merge=SolutionMerge.NODES)[1], phase=1)
    multi_phase_rerun.add_animated_model(biorbd_model, q[0], phase=0)
    multi_phase_rerun.add_animated_model(biorbd_model, q[1], phase=1)

    multi_phase_rerun.rerun("push_recovery")

    sol.graphs(show_bounds=True)

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
