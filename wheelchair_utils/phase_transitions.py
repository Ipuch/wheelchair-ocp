from casadi import MX, vertcat

from bioptim import PenaltyController


def custom_phase_transition_pre(controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # Take the values of q of the BioMod without holonomic constraints
    states_pre = controllers[0].states.cx  # pq pas le forearm_rotZ dans cet état ??

    nb_independent = controllers[1].model.nb_independent_joints
    u_post = controllers[1].states.cx[:nb_independent]
    udot_post = controllers[1].states.cx[nb_independent:]

    # Take the q of the independent joint and calculate the q of dependent joint
    v_post = controllers[1].model.compute_v_from_u_explicit_symbolic(u_post)
    q_post = controllers[1].model.state_from_partition(u_post, v_post)
    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.state_from_partition(udot_post, vdot_post)

    independent_index_pre = controllers[0].model.independent_joint_index  # continuity on independent indexes only
    states_post = vertcat(
        q_post[independent_index_pre], qdot_post[independent_index_pre]
    )  # slicing car pas besoin de tous les états

    return states_pre - states_post


def custom_phase_transition_post(controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # # Take the values of q of the BioMod without holonomics constraints
    # nb_independent = controllers[0].model.nb_independent_joints
    # u_pre = controllers[0].states.cx[:nb_independent]
    # udot_pre = controllers[0].states.cx[nb_independent:]
    #
    # # Take the q of the independent joint and calculate the q of dependent joint
    # v_pre = controllers[0].model.compute_v_from_u_explicit_symbolic(u_pre)
    # q_pre = controllers[0].model.state_from_partition(u_pre, v_pre)
    # Bvu = controllers[0].model.coupling_matrix(q_pre)
    # vdot_pre = Bvu @ udot_pre
    # qdot_pre = controllers[0].model.state_from_partition(udot_pre, vdot_pre)
    # q_pre = q_pre[slice(1, 4)]  # TODO : à clean pour aller chercher les indépendants de la phase suivante !!
    # qdot_pre = qdot_pre[slice(1, 4)]
    # states_pre = vertcat(q_pre, qdot_pre)
    #
    # states_post = controllers[1].states.cx
    #
    # return states_pre - states_post

    # Take the values of q of the BioMod without holonomics constraints
    nb_independent_pre = controllers[0].model.nb_independent_joints
    u_pre = controllers[0].states.cx[:nb_independent_pre]
    udot_pre = controllers[0].states.cx[nb_independent_pre : nb_independent_pre * 2]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_pre = controllers[0].model.compute_v_from_u_explicit_symbolic(u_pre)
    q_pre = controllers[0].model.state_from_partition(u_pre, v_pre)
    Bvu = controllers[0].model.coupling_matrix(q_pre)
    vdot_pre = Bvu @ udot_pre
    qdot_pre = controllers[0].model.state_from_partition(udot_pre, vdot_pre)

    states_pre = vertcat(q_pre, qdot_pre)

    nb_independent_post = controllers[1].model.nb_independent_joints
    u_post = controllers[1].states.cx[:nb_independent_post]
    udot_post = controllers[1].states.cx[nb_independent_post : nb_independent_post * 2]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_post = controllers[1].model.compute_v_from_u_explicit_symbolic(u_post)
    q_post = controllers[1].model.state_from_partition(u_post, v_post)
    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.state_from_partition(udot_post, vdot_post)

    states_post = vertcat(q_post, qdot_post)

    # tau_pre = controllers[0].states["tau"].cx
    # tau_post = controllers[1].states["tau"].cx
    #
    # states_pre = vertcat(states_pre, tau_pre)
    # states_post = vertcat(states_post, tau_post)

    return states_pre - states_post
