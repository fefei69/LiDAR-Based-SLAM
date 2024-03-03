import numpy as np
import matplotlib.pyplot as plt#; plt.ion() #this is for interactive plot (will cause the plot close immediately)
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial.transform import Rotation
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import pdb
from tqdm import tqdm
from load_data import *

def state_matrix_to_pose_matrix(state_matrix):
    pose_matrix = []
    for i in range(len(state_matrix)):
        r = Rotation.from_euler('z', state_matrix[i][2]).as_matrix()
        pose = homegenous_transformation(r,np.append(state_matrix[i][0:2],0))
        pose_matrix.append(pose)   
    return np.array(pose_matrix)

def visualize_icp_result(source_pc, target_pc, pose):
    '''
    Visualize the result of ICP
    source_pc: numpy array, (N, 3)
    target_pc: numpy array, (N, 3)
    pose: SE(3) numpy array, (4, 4)
    '''
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
    source_pcd.paint_uniform_color([0, 0, 1])

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
    target_pcd.paint_uniform_color([1, 0, 0])
    
    # target_pcd.transform(pose)
    source_pcd.transform(pose)
    o3d.visualization.draw_geometries([source_pcd, target_pcd])

def homegenous_transformation(R, t):
    T = np.eye(4)
    # Open3D might apply T directly (T @ pc)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def loss(source, target, R):
    loss = np.sum(np.linalg.norm(target @ R.T - source, axis=0,keepdims=True))
    return loss

def Kabsch_Algorithm(source, target):
    source = source - np.mean(source,axis=0)
    target = target - np.mean(target,axis=0)
    Q_matrix = np.zeros((3,3))
    for i in range(source.shape[0]):
        Q_matrix = Q_matrix + np.outer(source[i], target[i].T)
        # pdb.set_trace()
    U, S, V_T = np.linalg.svd(Q_matrix)
    # avoid reflection
    F = np.array([[1, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, np.linalg.det(U @ V_T)]])
    R_optimal = U @ F @ V_T
    return R_optimal

def icp_for_scan_matching(source, target, initial_rotation, initial_translation, down_sample_rate, rot_mat=None, max_iterations=15, tolerance=1e-5):
    '''
    Iterative Closest Point (ICP) algorithm
    source: numpy array, (N, 3)
    target: numpy array, (N, 3)
    max_iterations: int, maximum number of iterations
    tolerance: float, difference in error to stop iterations
    return: SO(3) numpy array, (3, 3), Rotation matrix
    '''
    p_0 = initial_translation
    if down_sample_rate == 1:
        source_pc_downsampled = source
    else:
        source_pc_downsampled = source[::int(down_sample_rate/5)]
    target_original = target.copy()
    source_original = source.copy()
    target = target + p_0
    target_pc_downsampled = target[::down_sample_rate]
    if rot_mat is False:
        Rot = Rotation.from_euler('z',initial_rotation).as_matrix()
    else:
        Rot = initial_rotation
    Old_Rot = Rot
    Old_Trans = p_0 
    rot_target_pc_downsampled =  target_pc_downsampled @ Rot.T
    for i in range(max_iterations):
        # Find the nearest neighbors
        associated_source, target_pc = data_association(source_pc_downsampled, rot_target_pc_downsampled)
        R = Kabsch_Algorithm(associated_source, target_pc)
        LOSS = loss(associated_source, target_pc, R)
        rot_target_pc_downsampled =  target_pc @ R.T
        translation = np.mean(associated_source,axis=0) - np.mean(rot_target_pc_downsampled,axis=0)
        New_Rot = R @ Old_Rot
        New_Trans = Old_Trans @ R.T + translation
        Old_Rot = New_Rot
        Old_Trans = New_Trans
        rot_target_pc_downsampled = rot_target_pc_downsampled + translation
    print("Loss: ",LOSS,"\n")
    Optimal_translation = np.mean(rot_target_pc_downsampled,axis=0,keepdims=True) - np.mean(target_original,axis=0,keepdims=True) @ New_Rot.T
    Optimal_translation_inverse = np.mean(target_original,axis=0,keepdims=True) - np.mean(rot_target_pc_downsampled,axis=0,keepdims=True) @ New_Rot
    T_target_to_source = homegenous_transformation(New_Rot, Optimal_translation)
    T_source_to_target = homegenous_transformation(New_Rot.T, Optimal_translation_inverse)
    return rot_target_pc_downsampled, T_source_to_target

def data_association(source_pc, target_pc):
    neighbbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(source_pc)
    distances, indices = neighbbors.kneighbors(target_pc)
    assoc_source = source_pc[indices].reshape(-1,3)
    return assoc_source, target_pc

def tic():
    return time.time()
def toc(tstart, name="Operation"):
    print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    '''
    INPUT 
    im              the map 
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
    xs,ys           physical x,y,positions you want to evaluate "correlation" 

    OUTPUT 
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                            np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr


def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y)).astype(np.int16)
    

def test_bresenham2D():
    import time
    sx = 0
    sy = 1
    print("Testing bresenham2D...")
    r1 = bresenham2D(sx, sy, 10, 5)
    r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
    r2 = bresenham2D(sx, sy, 9, 6)
    r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
    if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
        print("...Test passed.")
    else:
        print("...Test failed.")

    # Timing for 1000 random rays
    num_rep = 1000
    start_time = time.time()
    for i in range(0,num_rep):
        x,y = bresenham2D(sx, sy, 500, 200)
    print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Occupancy_Mapping(synced_lidar_ranges,POSE,sx,sy):
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -30  #meters
    MAP['ymin']  = -30
    MAP['xmax']  =  30
    MAP['ymax']  =  30 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float32) #DATA TYPE: char or int8
    S_x = []
    S_y = []
    x_traj = []
    y_traj = []
    for i in tqdm(range(synced_lidar_ranges.shape[1])):
        ranges = synced_lidar_ranges[:,i]
        angles = np.arange(-135,135.25,0.25)*np.pi/180.0
        # take valid indices
        indValid = np.logical_and((ranges < 10),(ranges> 0.1))
        ranges = ranges[indValid]
        angles = angles[indValid]
    
        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
    
        # convert position in the map frame here 
        # convert lidar frame world frame
        lidar_pc = np.stack((xs0-0.135,ys0,np.zeros(xs0.shape),np.ones(xs0.shape))).T
        lidar_in_world_frame = lidar_pc @ POSE[i].T
        init_pose = homegenous_transformation(np.eye(3),np.zeros(3))
        position =  init_pose @ POSE[i]
        x_traj.append(np.ceil((position[:3, 3][0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1)
        y_traj.append(np.ceil((position[:3, 3][1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1)
        # y_traj.append(np.ceil((MAP['ymax'] - position[:3, 3][1]) / MAP['res'] ).astype(np.int16)-1)
        in_map = np.logical_and(
                        np.logical_and(MAP['xmin'] <= lidar_in_world_frame[:, 0],
                           lidar_in_world_frame[:, 0] <= MAP['xmax']),
                        np.logical_and(MAP['ymin'] <= lidar_in_world_frame[:, 1],
                           lidar_in_world_frame[:, 1] <= MAP['ymax']))
        lidar_in_world_frame = lidar_in_world_frame[in_map,:]
        # For testing coordinate
        test_map = False
        Ex = []
        Ey = []
        if test_map == False:
            for j in range(lidar_in_world_frame.shape[0]):
                # convert from meters to cells
                ex = np.ceil((lidar_in_world_frame[:,0][j] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
                ey = np.ceil((lidar_in_world_frame[:,1][j] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
                Ex.append(ex)
                Ey.append(ey)
                rays = bresenham2D(sx[i],sy[i],ex,ey)
                MAP['map'][rays[1],rays[0]]-=np.log(2)
                MAP['map'][ey,ex]+=np.log(4)
                # For testing coordinate
                # MAP['map'][100:120,100:120]-=np.log(2)
    MAP['map'] = sigmoid(MAP['map'])

    #plot map
    plt.plot(sx, sy, color = 'orange', markersize=5)
    # For testing lidar transformation (lidar scans)
    # plt.plot(Ex,Ey,'.g',markersize=1)
    # np.save(f"map/MAP_values_imu_dataset{dataset}.npy",MAP['map'])
    # np.save("map/MAP_values_scan_matching_dataset21.npy",MAP['map'])
    # np.save(f"map/MAP_values_gtsam_icp_imu_icp_dataset{dataset}.npy",MAP['map'])
    plt.imshow(MAP['map'],cmap="cividis",interpolation='nearest')
    plt.colorbar()
    plt.title("Occupancy grid map")
    # plt.savefig(f"map/Occupancy_grid_map_imu_dataset{dataset}.png")
    plt.show()
  
  
def show_lidar():
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    ranges = np.load("test_ranges.npy")
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, ranges)
    ax.set_rmax(10)
    ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Lidar scan data", va='bottom')
    plt.show()
	

if __name__ == '__main__':
    print("This is a library file, not meant to be run standalone.")
    # show_lidar()
    # test_mapCorrelation()
    # test_bresenham2D()

