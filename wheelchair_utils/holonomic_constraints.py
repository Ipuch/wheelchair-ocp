from typing import Any

from casadi import MX, vertcat, Function, jacobian

from wheelchair_utils.custom_biorbd_model_holonomic import BiorbdModelCustomHolonomic


def generate_rolling_joint_constraint(
    translation_joint_index: int,
    rotation_joint_index: int,
    radius: float,
    model: Any,
) -> tuple[Function, Function, Function]:
    """Generate a rolling joint constraint between two joints"""

    # symbolic variables to create the functions
    q_sym = MX.sym("q", model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)

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


def generate_close_loop_constraint(
    marker_1: str,
    marker_2: str,
    index: slice = slice(0, 3),
    local_frame_index: int = None,
    model: Any = None,
) -> tuple[Function, Function, Function]:
    """Generate a close loop constraint between two markers"""

    # symbolic variables to create the functions
    q_sym = MX.sym("q", model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)

    # symbolic markers in global frame
    marker_1_sym = model.marker(index=model.marker_index(marker_1))(
        q_sym,
        model.parameters,
    )
    marker_2_sym = model.marker(index=model.marker_index(marker_2))(
        q_sym,
        model.parameters,
    )

    # if local frame is provided, the markers are expressed in the same local frame
    if local_frame_index is not None:
        jcs_t = model.homogeneous_matrices_in_global(local_frame_index, inverse=True)(q_sym, model.parameters)
        marker_1_sym = (jcs_t @ vertcat(marker_1_sym, 1))[:3]
        marker_2_sym = (jcs_t @ vertcat(marker_2_sym, 1))[:3]

    # the constraint is the distance between the two markers, set to zero
    constraint = (marker_1_sym - marker_2_sym)[index]
    # the jacobian of the constraint
    constraint_jacobian = jacobian(constraint, q_sym)

    constraint_func = Function(
        "holonomic_constraint",
        [q_sym],
        [constraint],
        ["q"],
        ["holonomic_constraint"],
    ).expand()

    constraint_jacobian_func = Function(
        "holonomic_constraint_jacobian",
        [q_sym],
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
        [q_sym, q_dot_sym, q_ddot_sym],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["holonomic_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func
