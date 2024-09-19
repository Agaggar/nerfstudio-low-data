import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def generate_hemisphere_points(radius, height, num_points, origin=[0, 0, 0]):
    # Generate a fine grid of points on a hemisphere
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi / 2, num_points // 2)
    # v = 1/.964 * np.tanh(2*v)

    # Meshgrid to generate grid of u, v values
    U, V = np.meshgrid(u, v)

    # Convert u, v to x, y, z
    x = radius * np.cos(U) * np.sin(V) + origin[0]
    y = radius * np.sin(U) * np.sin(V) + origin[1]
    z = height * np.cos(V) + origin[2]

    # Flatten x, y, z to get 1D arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Combine x, y, z coordinates into a 3D array
    points = np.column_stack((x_flat, y_flat, z_flat))

    return points

def uniform_hemi_angles(num_samples, longitude_range=[0., 2*np.pi], latitude_range=[0, np.pi/2], seed=0):
    np.random.seed(seed)
    theta = np.random.uniform(longitude_range[0], longitude_range[1], num_samples)
    u = np.random.uniform(0, 1, num_samples)
    phi_min_cos = np.cos(latitude_range[0])
    phi_max_cos = np.cos(latitude_range[1])
    phi = np.arccos(phi_min_cos - (phi_min_cos - phi_max_cos) * u)  # Transform to ensure uniform sampling over the hemisphere
    return np.vstack((theta, phi)).T

def vector_to_quaternion(vector):
    """
    Create a quaternion such that the x-axis is aligned with the input vector.

    Parameters:
    - vector: A numpy array representing the input vector [x, y, z].

    Returns:
    - quaternion: A numpy array representing the quaternion [x, y, z, w].
    """
    # Define the original x-axis
    x_axis = np.array([1, 0, 0])
    # Compute the axis of rotation (cross product)
    axis = np.cross(x_axis, vector)
    axis_norm = np.linalg.norm(axis)
    # If the input vector is parallel to the x-axis, handle the special cases
    if axis_norm == 0:
        if np.allclose(vector, x_axis):
            return np.array([0, 0, 0, 1])  # No rotation needed
        else:
            return R.from_euler('y', np.pi).as_quat()  # 180 degrees rotation around y-axis
    axis = axis / axis_norm
    # Compute the angle of rotation (dot product)
    angle = np.arccos(np.dot(x_axis, vector))
    rot_180_y = R.from_euler('y', np.pi)
    rot_180_x = R.from_euler('x', np.pi)
    rot = (R.from_rotvec(axis * angle) * rot_180_y * rot_180_x)
    return rot

def calculate_normal_vectors(points, variable_point):
    # Calculate normal vectors as unit vectors pointing from each point on the hemisphere to the variable point
    normals = (points - variable_point) / np.linalg.norm(points - variable_point, axis=1)[:, np.newaxis]
    return normals

def angle_from_normal(normal_vectors, points_per_circle=10):
    # Calculate the rotation matrix and angle for each normal vector
    rotation_matrices = np.zeros((len(normal_vectors), 3, 3))
    angles = np.zeros((len(normal_vectors), 3))
    circum_angle = np.linspace(np.pi/2, 5/2*np.pi, points_per_circle)
    count = 0
    for i, normal_vector in enumerate(normal_vectors):
        # Calculate the angle
        angle = np.arccos(np.dot(normal_vector, [0, 0, 1]) / np.linalg.norm(normal_vector))
        rpy = np.array([angle, 0, circum_angle[count]])
        angles[i] = rpy
        rotation_matrices[i] = R.from_euler('xyz', rpy).as_matrix()

        count += 1
        if count % points_per_circle == 0:
            count = 0

    return rotation_matrices, angles

def generate_hemi_angles(num_circles=10, points_per_circle=10):
    # Generate a fine grid of points on a hemisphere
    u = np.linspace(0, 2 * np.pi, points_per_circle)
    v = np.linspace(0, np.pi / 2, num_circles)

    # Meshgrid to generate grid of u, v values
    U, V = np.meshgrid(u, v)

    return np.column_stack((U.flatten(), V.flatten()))

def spherical_to_hemi(hemi_angles, radius=0.6, height=1.25, origin=[0, 0, -1.1]):
    if len(hemi_angles.shape) == 2:
        # hemi_angles is a (N, 2) pre-defined sphere
        angles = np.zeros_like(hemi_angles)
        x = np.zeros(len(hemi_angles))
        y = np.zeros(len(hemi_angles))
        z = np.zeros(len(hemi_angles))

        for count, (V, U) in enumerate(zip(hemi_angles[:, 0], hemi_angles[:, 1])):
            x[count] = radius * np.cos(U) * np.sin(V) + origin[0]
            y[count] = radius * np.sin(U) * np.sin(V) + origin[1]
            z[count] = height * np.cos(V) + origin[2]
        return np.column_stack((x, y, z))
    else:
        return np.array([radius * np.cos(hemi_angles[0]) * np.sin(hemi_angles[1]) + origin[0],
                         radius * np.sin(hemi_angles[0]) * np.sin(hemi_angles[1]) + origin[1],
                         height * np.cos(hemi_angles[1]) + origin[2]])

def generate_camera_rotations(num_circles=10, points_per_circle=10, radius=0.6, height=1.25, origin=[0, 0, -1.1], center_of_mass=[0, 0, -.5], euler_angles=False):
    hemi_angles = generate_hemi_angles(num_circles, points_per_circle)
    hemi_cart_points = spherical_to_hemi(hemi_angles, radius, height, origin)
    normals = calculate_normal_vectors(hemi_cart_points, center_of_mass)
    rotation, _ = angle_from_normal(normals, points_per_circle)
    return torch.from_numpy(np.concatenate((rotation, np.expand_dims(hemi_cart_points, axis=2)), axis=2))

def rotation_from_angle(angle, euler=False):
    if euler:
        return np.array([angle[1], 0.0, angle[0] + np.pi/2])
    else:
        return R.from_euler('xyz', np.array([angle[1], 0.0, angle[0] + np.pi/2])).as_matrix()
    # (np.column_stack(((hemi_angles[:15] + np.array([np.pi/2, 0]))[:, 1], np.zeros((15, 1)), (hemi_angles[:15] + np.array([np.pi/2, 0]))[:, 0])))

def rotation_from_hemi_angles(angles, points_per_circle=10, num_circles=10, radius=0.6, height=1.25, origin=[0, 0, -1.1], center_of_mass=[0, 0, -.5], euler_angles=False):
    if euler_angles:
        rotation_matrices = np.zeros((len(angles), 3))
    else:
        rotation_matrices = np.zeros((len(angles), 3, 3))
    circum_angle = np.linspace(np.pi/2, 5/2*np.pi, points_per_circle)

    if num_circles % 2 == 0:
        pitch_adj = np.hstack((np.linspace(0, np.pi/4, num_circles//2), np.linspace(np.pi/2, 2/3*np.pi, num_circles//2)))
    else:
        pitch_adj = np.hstack((np.linspace(0, np.pi/4, num_circles//2+1), np.linspace(np.pi/2, 2/3*np.pi, num_circles//2)))

    i = 0
    for count in range(len(angles)):
        # rotation_matrices[count] = R.from_euler('xyz', [azimuth_altitude(hp.squeeze())[1], 0, circum_angle[i]]).as_matrix()
        if euler_angles:
            rotation_matrices[count] = [pitch_adj[count//points_per_circle], 0, circum_angle[i]]
        else:
            rotation_matrices[count] = R.from_euler('xyz', [pitch_adj[count//points_per_circle], 0, circum_angle[i]]).as_matrix()
        i += 1
        if (count+1) % points_per_circle == 0:
            i = 0
    
    return rotation_matrices

def azimuth_altitude(point, r=.5, h=1.25, origin=[0, 0, -.85]):
    alt = np.arccos((point[2] - origin[2])/h)
    if ((point[2] - origin[2])/h) > 1:
        print("error in going from cartesian to polar", point)
    if np.abs(np.sin(alt)) < 1e-2:
        azi = 0
    else:
        if ((point[1] - origin[1]) / r / np.sin(alt)) > 1:
            print("error in going from cartesian to polar", point, np.sin(alt))
        azi = np.arcsin((point[1] - origin[1]) / r / np.sin(alt))
        # if point[0] > 0 and point[1] > 0 and (azi < 0 or azi > np.pi/2):
        #     print("should be q1", azi, point[:3])
        if (point[0] - origin[0]) < 0 and (point[1] - origin[1]) > 0 and azi > 0: # (azi < np.pi/2 or azi > np.pi):
            # print("should be q2", azi, point[:3])
            azi = np.pi - azi
        if (point[0] - origin[0]) < 0 and (point[1] - origin[1]) < 0 and azi < 0: # (azi < np.pi or azi > 3*np.pi/2):
            # print("should be q3", azi, point[:3])
            azi = -np.pi - azi
        # if point[0] > 0 and point[1] < 0 and (azi < 3*np.pi/2 or azi > 2*np.pi):
        #     print("should be q4", azi, point[:3])
        # if np.abs(azi - np.arccos((point[0] - origin[0]) / r / np.sin(alt))) > .1:
        #     print(azi, np.arccos((point[0] - origin[0]) / r / np.sin(alt)), point[:3])
    return [azi, alt]

def rotation_from_hp(points, num_points=10, num_circles=5, r=.5, h=1.25, origin=[0, 0, -1.1], center_of_mass=[0, 0, -.5], euler_angles=False):
    if euler_angles:
        rotation_matrices = np.zeros((len(points), 3))
    else:
        rotation_matrices = np.zeros((len(points), 3, 3))
    circum_angle = np.linspace(np.pi/2, 5/2*np.pi, num_points)

    if num_circles % 2 == 0:
        pitch_adj = np.hstack((np.linspace(0, np.pi/4, num_circles//2+1), np.linspace(np.pi/2, 2/3*np.pi, num_circles//2)))
    else:
        pitch_adj = np.hstack((np.linspace(0, np.pi/4, num_circles//2+1), np.linspace(np.pi/2, 2/3*np.pi, num_circles//2)))

    i = 0
    for count in range(len(points)):
        # rotation_matrices[count] = R.from_euler('xyz', [azimuth_altitude(hp.squeeze())[1], 0, circum_angle[i]]).as_matrix()
        if euler_angles:
            rotation_matrices[count] = [pitch_adj[count//num_points], 0, circum_angle[i]]
        else:
            rotation_matrices[count] = R.from_euler('xyz', [pitch_adj[count//num_points], 0, circum_angle[i]]).as_matrix()
        i += 1
        if (count+1) % num_points == 0:
            i = 0
    
    return rotation_matrices

def rpy_to_spherical(rpy):
    rpy[:, 1] = 0.
    theta = rpy[:, 2] - np.pi/2
    phi = rpy[:, 0]
    return np.vstack((theta, phi)).T