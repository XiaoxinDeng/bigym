import numpy as np

# ----------------------------
# Helpers (w,x,y,z)
# ----------------------------
def normalize(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def quat_inv(q):
    return quat_conj(q)  # unit assumed

def rotate_vec_by_quat(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_inv(q))[1:]

def quat_from_two_unit_vectors(a, b, eps=1e-12):
    a = normalize(a); b = normalize(b)
    d = float(np.dot(a, b))
    if d > 1.0 - eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if d < -1.0 + eps:
        axis = normalize(np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float64)))
        if np.linalg.norm(axis) < eps:
            axis = normalize(np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float64)))
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    c = np.cross(a, b)
    q = np.array([1.0 + d, c[0], c[1], c[2]], dtype=np.float64)
    return normalize(q)