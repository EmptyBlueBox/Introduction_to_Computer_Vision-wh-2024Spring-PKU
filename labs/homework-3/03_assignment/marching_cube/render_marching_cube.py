# %%
import matplotlib.pyplot as plt
import numpy as np
from lookup_table import CaseNum2EdgeOffset, getCaseNum
import trimesh
import os
import time

# %%


def marching_cube(thres, cells):
    # vertices use dictionary to avoid duplicate axes
    vertex_array = {}
    vertex_num = 0
    rep_vertex_num = 0
    face_array = []
    t1 = time.time()
    # -------------------TODO------------------
    # compute vertices and faces
    # vertices: [N, 3]
    # faces: [M, 3], e.g. np.array([[0,1,2]]) means a triangle composed of vertices[0], vertices[1] and vertices[2]
    # for-loop is allowed to reduce difficulty
    for i in range(cells.shape[0]-1):
        for j in range(cells.shape[1]-1):
            for k in range(cells.shape[2]-1):
                # get the case number
                case_num = getCaseNum(i, j, k, thres, cells)
                vertice_tmp_stack = []

                for case in case_num:
                    if case == -1:
                        break
                    edge_corner = CaseNum2EdgeOffset[case]
                    value0 = cells[i+edge_corner[0], j+edge_corner[1], k+edge_corner[2]]
                    value1 = cells[i+edge_corner[3], j+edge_corner[4], k+edge_corner[5]]
                    alpha = (thres-value0)/(value1-value0)
                    xp = i + edge_corner[0] + alpha * (edge_corner[3] - edge_corner[0])
                    yp = j + edge_corner[1] + alpha * (edge_corner[4] - edge_corner[1])
                    zp = k + edge_corner[2] + alpha * (edge_corner[5] - edge_corner[2])
                    vertex_key = (round(xp, 5), round(yp, 5), round(zp, 5))  # Round to avoid floating point precision issues
                    # update vertex, add a new vertex
                    rep_vertex_num += 1
                    if vertex_key not in vertex_array:
                        vertex_array[vertex_key] = vertex_num  # add new vertex
                        vertex_num += 1

                    # update face, add a new face using vertex index
                    vertice_tmp_stack.append(vertex_array[vertex_key])
                    if len(vertice_tmp_stack) == 3:
                        # Every three vertices form one triangle
                        face_array.append(vertice_tmp_stack)
                        vertice_tmp_stack = []

    # -------------------TODO------------------
    t2 = time.time()
    print("\nTime taken by algorithm\n"+'-'*40+"\n{} s".format(t2-t1))
    vertex_list = list(vertex_array.keys())
    vertices = np.array(vertex_list)
    faces = np.array(face_array).reshape(-1, 3)
    print('rep_vertex_num:', rep_vertex_num, 'vertex_num:', vertex_num, 'face_num:', faces.shape[0])
    print('vertice:', vertices.shape, 'face:', np.array(face_array).shape)
    return vertices, faces


# %%
# reconstruct these two animals
shape_name_lst = ['spot', 'bob']
shape_name_lst = ['bob']
for shape_name in shape_name_lst:
    data = np.load(os.path.join('data', shape_name + '_cell.npy'))
    print(data.shape)
    verts, faces = marching_cube(thres=0, cells=data)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Save image using matplotlib
    mesh_txt = trimesh.exchange.obj.export_obj(mesh)
    mesh.show()
