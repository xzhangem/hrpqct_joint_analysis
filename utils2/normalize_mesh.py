import numpy as np
import pytorch3d 
import trimesh 



def normalize_mesh_file(input_file):
    if isinstance(input_file, str):
        mesh = trimesh.load(input_file)
    else:
        mesh = input_file
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    center = np.mean(vertices, axis=0)

    max_distance = np.max(np.linalg.norm(vertices - center, axis=1))
    normalized_points = (vertices - center) / max_distance

    return normalized_points, faces, center, max_distance
