import numpy as np
from numba import njit


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _fill_rotation_matrix(rotation_vec, rotation_matrix):
    wx = rotation_vec[0]
    wy = rotation_vec[1]
    wz = rotation_vec[2]
    theta2 = wx * wx + wy * wy + wz * wz
    rotation_matrix[0, 0] = 1.0
    rotation_matrix[0, 1] = 0.0
    rotation_matrix[0, 2] = 0.0
    rotation_matrix[1, 0] = 0.0
    rotation_matrix[1, 1] = 1.0
    rotation_matrix[1, 2] = 0.0
    rotation_matrix[2, 0] = 0.0
    rotation_matrix[2, 1] = 0.0
    rotation_matrix[2, 2] = 1.0
    if theta2 < 1e-24:
        rotation_matrix[0, 1] = -wz
        rotation_matrix[0, 2] = wy
        rotation_matrix[1, 0] = wz
        rotation_matrix[1, 2] = -wx
        rotation_matrix[2, 0] = -wy
        rotation_matrix[2, 1] = wx
        return

    theta = np.sqrt(theta2)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / theta2

    k2_00 = -wy * wy - wz * wz
    k2_01 = wx * wy
    k2_02 = wx * wz
    k2_10 = wx * wy
    k2_11 = -wx * wx - wz * wz
    k2_12 = wy * wz
    k2_20 = wx * wz
    k2_21 = wy * wz
    k2_22 = -wx * wx - wy * wy

    rotation_matrix[0, 0] = 1.0 + b * k2_00
    rotation_matrix[0, 1] = -a * wz + b * k2_01
    rotation_matrix[0, 2] = a * wy + b * k2_02
    rotation_matrix[1, 0] = a * wz + b * k2_10
    rotation_matrix[1, 1] = 1.0 + b * k2_11
    rotation_matrix[1, 2] = -a * wx + b * k2_12
    rotation_matrix[2, 0] = -a * wy + b * k2_20
    rotation_matrix[2, 1] = a * wx + b * k2_21
    rotation_matrix[2, 2] = 1.0 + b * k2_22


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _rotate_point_about_axis(px, py, pz, ax, ay, az, ux, uy, uz, angle):
    vx = px - ax
    vy = py - ay
    vz = pz - az
    c = np.cos(angle)
    s = np.sin(angle)
    dot = vx * ux + vy * uy + vz * uz
    cross_x = vy * uz - vz * uy
    cross_y = vz * ux - vx * uz
    cross_z = vx * uy - vy * ux
    one_c = 1.0 - c
    return (
        ax + c * vx + s * cross_x + one_c * dot * ux,
        ay + c * vy + s * cross_y + one_c * dot * uy,
        az + c * vz + s * cross_z + one_c * dot * uz,
    )


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _apply_transformations_inplace(x, ori_coords, anchor, rot_atom_pairs,
                                   rot_affected_indices, rot_affected_offsets,
                                   coords, rotation_matrix):
    for i in range(ori_coords.shape[0]):
        coords[i, 0] = ori_coords[i, 0]
        coords[i, 1] = ori_coords[i, 1]
        coords[i, 2] = ori_coords[i, 2]

    for i in range(rot_atom_pairs.shape[0]):
        a1 = rot_atom_pairs[i, 0]
        a2 = rot_atom_pairs[i, 1]
        ax = coords[a1, 0]
        ay = coords[a1, 1]
        az = coords[a1, 2]
        ux = coords[a2, 0] - ax
        uy = coords[a2, 1] - ay
        uz = coords[a2, 2] - az
        norm = np.sqrt(ux * ux + uy * uy + uz * uz)
        if norm < 1e-12:
            continue
        inv_norm = 1.0 / norm
        ux *= inv_norm
        uy *= inv_norm
        uz *= inv_norm
        angle = x[6 + i]
        start = rot_affected_offsets[i]
        end = rot_affected_offsets[i + 1]
        for offset in range(start, end):
            atom_idx = rot_affected_indices[offset]
            rx, ry, rz = _rotate_point_about_axis(
                coords[atom_idx, 0], coords[atom_idx, 1], coords[atom_idx, 2],
                ax, ay, az, ux, uy, uz, angle
            )
            coords[atom_idx, 0] = rx
            coords[atom_idx, 1] = ry
            coords[atom_idx, 2] = rz

    _fill_rotation_matrix(x[3:6], rotation_matrix)
    tx = x[0]
    ty = x[1]
    tz = x[2]
    ax = anchor[0]
    ay = anchor[1]
    az = anchor[2]
    for i in range(coords.shape[0]):
        vx = coords[i, 0] - ax
        vy = coords[i, 1] - ay
        vz = coords[i, 2] - az
        coords[i, 0] = (rotation_matrix[0, 0] * vx + rotation_matrix[0, 1] * vy +
                        rotation_matrix[0, 2] * vz + ax + tx)
        coords[i, 1] = (rotation_matrix[1, 0] * vx + rotation_matrix[1, 1] * vy +
                        rotation_matrix[1, 2] * vz + ay + ty)
        coords[i, 2] = (rotation_matrix[2, 0] * vx + rotation_matrix[2, 1] * vy +
                        rotation_matrix[2, 2] * vz + az + tz)


@njit(error_model='numpy', fastmath=True, boundscheck=False, cache=True)
def _project_pose_gradient_inplace(x, ori_coords, anchor, rot_atom_pairs,
                                   rot_affected_indices, rot_affected_offsets,
                                   atom_grad, param_grad, internal_coords,
                                   internal_grad, rotation_matrix):
    _apply_transformations_inplace(
        x, ori_coords, anchor, rot_atom_pairs, rot_affected_indices,
        rot_affected_offsets, internal_coords, rotation_matrix
    )

    tx = x[0]
    ty = x[1]
    tz = x[2]
    ax = anchor[0]
    ay = anchor[1]
    az = anchor[2]
    for i in range(internal_coords.shape[0]):
        px = internal_coords[i, 0] - tx - ax
        py = internal_coords[i, 1] - ty - ay
        pz = internal_coords[i, 2] - tz - az
        internal_coords[i, 0] = (rotation_matrix[0, 0] * px + rotation_matrix[1, 0] * py +
                                 rotation_matrix[2, 0] * pz + ax)
        internal_coords[i, 1] = (rotation_matrix[0, 1] * px + rotation_matrix[1, 1] * py +
                                 rotation_matrix[2, 1] * pz + ay)
        internal_coords[i, 2] = (rotation_matrix[0, 2] * px + rotation_matrix[1, 2] * py +
                                 rotation_matrix[2, 2] * pz + az)

    for i in range(param_grad.shape[0]):
        param_grad[i] = 0.0

    tau_x = 0.0
    tau_y = 0.0
    tau_z = 0.0
    for i in range(atom_grad.shape[0]):
        gx = atom_grad[i, 0]
        gy = atom_grad[i, 1]
        gz = atom_grad[i, 2]
        param_grad[0] += gx
        param_grad[1] += gy
        param_grad[2] += gz

        bgx = rotation_matrix[0, 0] * gx + rotation_matrix[1, 0] * gy + rotation_matrix[2, 0] * gz
        bgy = rotation_matrix[0, 1] * gx + rotation_matrix[1, 1] * gy + rotation_matrix[2, 1] * gz
        bgz = rotation_matrix[0, 2] * gx + rotation_matrix[1, 2] * gy + rotation_matrix[2, 2] * gz
        internal_grad[i, 0] = bgx
        internal_grad[i, 1] = bgy
        internal_grad[i, 2] = bgz

        rx = internal_coords[i, 0] - ax
        ry = internal_coords[i, 1] - ay
        rz = internal_coords[i, 2] - az
        tau_x += ry * bgz - rz * bgy
        tau_y += rz * bgx - rx * bgz
        tau_z += rx * bgy - ry * bgx

    wx = x[3]
    wy = x[4]
    wz = x[5]
    theta2 = wx * wx + wy * wy + wz * wz
    if theta2 < 1e-16:
        param_grad[3] = tau_x + 0.5 * (-wz * tau_y + wy * tau_z)
        param_grad[4] = tau_y + 0.5 * (wz * tau_x - wx * tau_z)
        param_grad[5] = tau_z + 0.5 * (-wy * tau_x + wx * tau_y)
    else:
        theta = np.sqrt(theta2)
        a = (1.0 - np.cos(theta)) / theta2
        b = (theta - np.sin(theta)) / (theta2 * theta)
        k_tau_x = -wz * tau_y + wy * tau_z
        k_tau_y = wz * tau_x - wx * tau_z
        k_tau_z = -wy * tau_x + wx * tau_y
        dot = wx * tau_x + wy * tau_y + wz * tau_z
        k2_tau_x = wx * dot - theta2 * tau_x
        k2_tau_y = wy * dot - theta2 * tau_y
        k2_tau_z = wz * dot - theta2 * tau_z
        param_grad[3] = tau_x + a * k_tau_x + b * k2_tau_x
        param_grad[4] = tau_y + a * k_tau_y + b * k2_tau_y
        param_grad[5] = tau_z + a * k_tau_z + b * k2_tau_z

    for torsion_idx in range(rot_atom_pairs.shape[0] - 1, -1, -1):
        a1 = rot_atom_pairs[torsion_idx, 0]
        a2 = rot_atom_pairs[torsion_idx, 1]
        ax = internal_coords[a1, 0]
        ay = internal_coords[a1, 1]
        az = internal_coords[a1, 2]
        ux = internal_coords[a2, 0] - ax
        uy = internal_coords[a2, 1] - ay
        uz = internal_coords[a2, 2] - az
        norm = np.sqrt(ux * ux + uy * uy + uz * uz)
        if norm < 1e-12:
            continue
        inv_norm = 1.0 / norm
        ux *= inv_norm
        uy *= inv_norm
        uz *= inv_norm
        grad_theta = 0.0
        start = rot_affected_offsets[torsion_idx]
        end = rot_affected_offsets[torsion_idx + 1]
        for offset in range(start, end):
            atom_idx = rot_affected_indices[offset]
            rx = internal_coords[atom_idx, 0] - ax
            ry = internal_coords[atom_idx, 1] - ay
            rz = internal_coords[atom_idx, 2] - az
            deriv_x = ry * uz - rz * uy
            deriv_y = rz * ux - rx * uz
            deriv_z = rx * uy - ry * ux
            grad_theta += (internal_grad[atom_idx, 0] * deriv_x +
                           internal_grad[atom_idx, 1] * deriv_y +
                           internal_grad[atom_idx, 2] * deriv_z)
        param_grad[6 + torsion_idx] = grad_theta

        angle = -x[6 + torsion_idx]
        for offset in range(start, end):
            atom_idx = rot_affected_indices[offset]
            rx, ry, rz = _rotate_point_about_axis(
                internal_coords[atom_idx, 0], internal_coords[atom_idx, 1],
                internal_coords[atom_idx, 2], ax, ay, az, ux, uy, uz, angle
            )
            internal_coords[atom_idx, 0] = rx
            internal_coords[atom_idx, 1] = ry
            internal_coords[atom_idx, 2] = rz

            gx, gy, gz = _rotate_point_about_axis(
                ax + internal_grad[atom_idx, 0],
                ay + internal_grad[atom_idx, 1],
                az + internal_grad[atom_idx, 2],
                ax, ay, az, ux, uy, uz, angle
            )
            internal_grad[atom_idx, 0] = gx - ax
            internal_grad[atom_idx, 1] = gy - ay
            internal_grad[atom_idx, 2] = gz - az


def apply_transformations(x, ori_coords, anchor,
                          rot_atom_pairs, rot_affected_indices,
                          rot_affected_offsets):
    coords = np.empty_like(ori_coords)
    rotation_matrix = np.empty((3, 3), dtype=np.float64)
    _apply_transformations_inplace(
        x, ori_coords, anchor, rot_atom_pairs,
        rot_affected_indices, rot_affected_offsets, coords, rotation_matrix
    )
    return coords


def apply_transformations_inplace(x, ori_coords, anchor,
                                  rot_atom_pairs, rot_affected_indices,
                                  rot_affected_offsets, coords_buffer,
                                  rotation_matrix_buffer):
    _apply_transformations_inplace(
        x, ori_coords, anchor, rot_atom_pairs,
        rot_affected_indices, rot_affected_offsets,
        coords_buffer, rotation_matrix_buffer
    )


def project_transform_gradients_inplace(x, ori_coords, anchor,
                                        rot_atom_pairs, rot_affected_indices,
                                        rot_affected_offsets, atom_grad,
                                        grad_buffer, internal_coords_buffer,
                                        internal_grad_buffer,
                                        rotation_matrix_buffer):
    _project_pose_gradient_inplace(
        x, ori_coords, anchor, rot_atom_pairs,
        rot_affected_indices, rot_affected_offsets, atom_grad,
        grad_buffer, internal_coords_buffer, internal_grad_buffer,
        rotation_matrix_buffer
    )
