import os
import scipy.io as sio
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
def loss(source, target, R):
  loss = np.sum(np.linalg.norm(target @ R.T - source, axis=0,keepdims=True))
  return loss

def Kabsch_Algorithm(source, target):
  Q_matrix = np.zeros((3,3))
  for i in range(source.shape[0]):
    Q_matrix = Q_matrix + source[i] @ target[i].T
  U, S, V_T = np.linalg.svd(Q_matrix)
  # avoid reflection
  F = np.array([[1, 0, 0], 
                [0, 1, 0], 
                [0, 0, np.linalg.det(U @ V_T)]])
  R_optimal = U @ F @ V_T
  return R_optimal

def icp(source, target, down_sample_rate, max_iterations=100, tolerance=1e-5):
  '''
  Iterative Closest Point (ICP) algorithm
  source: numpy array, (N, 3)
  target: numpy array, (N, 3)
  max_iterations: int, maximum number of iterations
  tolerance: float, difference in error to stop iterations
  return: SO(3) numpy array, (3, 3), Rotation matrix
  '''
  diff_center = np.mean(source,axis=0,keepdims=True) - np.mean(target,axis=0,keepdims=True)
  if down_sample_rate == 1:
    source_pc_downsampled = source
  else:
    source_pc_downsampled = source[::int(down_sample_rate/5)]
  target = target + diff_center
  target_pc_downsampled = target[::down_sample_rate]
  # Sample random z axis rotation for initial guess
  # random_rot = Rotation.random().as_euler('zyx')
  # Rot = Rotation.from_euler('z', random_rot[0]).as_matrix()
  Rot = Rotation.from_euler('z',-1).as_matrix()
  rot_target_pc_downsampled =  target_pc_downsampled @ Rot.T
  target_pc_downsampled_original = np.copy(target_pc_downsampled)
  for i in range(max_iterations):
    # Find the nearest neighbors
    associated_source, target_pc = data_association(source_pc_downsampled, rot_target_pc_downsampled)
    R = Kabsch_Algorithm(associated_source, target_pc)
    print("ICP iteration: ",i)
    LOSS = loss(associated_source, target_pc, R)
    print("Loss: ",LOSS)
    if LOSS < 2:
      break
    rot_target_pc_downsampled =  target_pc @ R.T

  return rot_target_pc_downsampled

def data_association(source_pc, target_pc):
  print("Numbers of Neighbors",target_pc.shape[0])
  neighbbors = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(source_pc)
  distances, indices = neighbbors.kneighbors(target_pc)
  assoc_source = source_pc[indices].reshape(-1,3)
  return assoc_source, target_pc

def read_canonical_model(model_name):
  '''
  Read canonical model from .mat file
  model_name: str, 'drill' or 'liq_container'
  return: numpy array, (N, 3)
  '''
  model_fname = os.path.join('./data', model_name, 'model.mat')
  model = sio.loadmat(model_fname)

  cano_pc = model['Mdata'].T / 1000.0 # convert to meter

  return cano_pc


def load_pc(model_name, id):
  '''
  Load point cloud from .npy file
  model_name: str, 'drill' or 'liq_container'
  id: int, point cloud id
  return: numpy array, (N, 3)
  '''
  pc_fname = os.path.join('./data', model_name, '%d.npy' % id)
  pc = np.load(pc_fname)

  return pc


def visualize_icp_result(source_pc, target_pc, pose):
  '''
  Visualize the result of ICP
  source_pc: numpy array, (N, 3)
  target_pc: numpy array, (N, 3)
  pose: SE(4) numpy array, (4, 4)
  '''
  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])

  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])

  source_pcd.transform(pose)

  o3d.visualization.draw_geometries([source_pcd, target_pcd])


