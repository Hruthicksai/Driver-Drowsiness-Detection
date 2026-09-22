import cv2
import numpy as np


def eye_aspect_ratio(eye):

    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    if C == 0:
        return 0

    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):

    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])

    if C == 0:
        return 0

    return (A + B) / (2.0 * C)


def get_head_pose_angle(shape, frame):

    image_points = np.array([
        shape[30],
        shape[8],
        shape[36],
        shape[45],
        shape[48],
        shape[54]
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    height, width = frame.shape[:2]

    focal_length = width

    center = (width / 2, height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs
    )

    if not success:
        return 0, 0, 0

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    pose_matrix = np.hstack(
        (rotation_matrix, translation_vector)
    )

    _, _, _, _, _, _, euler_angles = (
        cv2.decomposeProjectionMatrix(pose_matrix)
    )

    pitch = float(euler_angles[0])
    yaw = float(euler_angles[1])
    roll = float(euler_angles[2])

    return pitch, yaw, roll