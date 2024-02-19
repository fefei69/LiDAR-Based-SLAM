
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result

if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 1 # number of point clouds

  source_pc = read_canonical_model(obj_name)
  source_pc_mean = source_pc - np.mean(source_pc)
  # print(source_pc_mean)
  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)
    diff_center = np.mean(source_pc,axis=0,keepdims=True) - np.mean(target_pc,axis=0,keepdims=True)
    target_pc = target_pc + diff_center

    # estimated_pose, you need to estimate the pose with ICP
    pose = np.eye(4)

    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc, pose)

