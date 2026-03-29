import numpy as np

# ============================================================
# FUNZIONI GEOMETRICHE esterne. Necessitano di essere pickle friendly per serializzare gli argomenti e le funzioni da inviare ai processi figli
# ============================================================

def _is_point_in_triangle_barycentric_2D(point, triangle, tolerance, mynormal):
    """ Baricenter 2D test - invariant to the translatione."""
    p  = np.array([point[0],       point[1]])
    v0 = np.array([triangle[0][0], triangle[0][1]])
    v1 = np.array([triangle[1][0], triangle[1][1]])
    v2 = np.array([triangle[2][0], triangle[2][1]])

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p  = p  - v0

    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-10:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1.0 + tolerance)


def _intersezione_piano_retta_NUM(triangle, point, normalGrid, tol=1e-12):
    """
    Plane-line intersection.
    """
    A  = np.array(triangle[0], dtype=np.float64)
    B  = np.array(triangle[1], dtype=np.float64)
    C  = np.array(triangle[2], dtype=np.float64)
    P0 = np.array(point,       dtype=np.float64)
    v  = np.array(normalGrid,  dtype=np.float64)

    AB = B - A
    AC = C - A
    n  = np.cross(AB, AC)

    norm_n = np.linalg.norm(n)
    if norm_n < tol:
        return None, None, False, None

    n = n / norm_n

    denom = np.dot(n, v)
    if abs(denom) < tol:
        return None, None, False, None

    t           = np.dot(n, A - P0) / denom
    P           = P0 + t * v
    dist_scalar = np.dot(n, P - A)
    P_proj      = P - dist_scalar * n
    dist        = abs(np.dot(n, P_proj - A))

    if dist < tol:
        return P_proj.tolist(), float(dist), True, float(t)
    else:
        return None, float(dist), False, None


def _filtro_piano_NUM(triangle, point, normalGrid, tol=1e-12):
    """Checks that the point lies on the triangle plane."""
    A = np.array(triangle[0], dtype=np.float64)
    B = np.array(triangle[1], dtype=np.float64)
    C = np.array(triangle[2], dtype=np.float64)
    P = np.array(point,       dtype=np.float64)

    AB = B - A
    AC = C - A
    n  = np.cross(AB, AC)

    norm_n = np.linalg.norm(n)
    if norm_n < tol:
        return None, 0.0, False

    n    = n / norm_n
    dist = abs(np.dot(n, P - A))

    if dist < tol:
        return P.tolist(), float(dist), True
    else:
        return None, float(dist), False

# ============================================================
# LOGICA DI PROIEZIONE COMUNE
# ============================================================

def _compute_candidates(t_point, nearest_indices,
                        mesh_vertices_local, chunk_mesh_faces,
                        tolerance, mynormal):
    """
    Calculate the projection candidates for a single point.
    Shared by both workers (open and closed).
    """
    candidates = []

    for local_index in nearest_indices:
        face     = chunk_mesh_faces[local_index]
        triangle = mesh_vertices_local[face]

        if not _is_point_in_triangle_barycentric_2D(
            t_point, triangle, tolerance, mynormal
        ):
            continue

        result = _intersezione_piano_retta_NUM(triangle, t_point, mynormal)

        if not result[2] or result[0] is None:
            continue

        try:
            projected_point = np.array(result[0])
            t_param         = result[3]

            filtered = _filtro_piano_NUM(triangle, projected_point, mynormal)

            if not filtered[2] or filtered[0] is None:
                continue

            proj   = np.array(filtered[0])
            soglia = float(filtered[1])
            dist   = np.linalg.norm(proj - t_point)

            candidates.append({
                'point_local': proj,
                'soglia':      soglia,
                'distance':    dist,
                't_param':     t_param
            })

        except (TypeError, IndexError, AttributeError):
            continue

    return candidates    