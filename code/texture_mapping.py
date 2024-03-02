import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
from load_data import *
from encoder_imu_odometry import transform_pose_matrix_to_cells
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from tqdm import tqdm
disp_path = f"../data/Disparity{dataset}/"
rgb_path = f"../data/RGB{dataset}/"

def transformation_for_image(image_cord,T):
    # hard coded shape for now 
    image_cord_reshape = image_cord.reshape(-1,4)
    transformed_image_cord = image_cord_reshape @ np.linalg.inv(T)
    image_cord_reshape_back = transformed_image_cord.reshape(480,640,4)
    return image_cord_reshape_back

def pose_transformation_for_image(image_cord,T):
    # hard coded shape for now 
    image_cord_reshape = image_cord.reshape(-1,4)
    transformed_image_cord = image_cord_reshape @ T.T
    image_cord_reshape_back = transformed_image_cord.reshape(480,640,4)
    return image_cord_reshape_back

def homegenous_transformation(R, t):
    T = np.eye(4)
    # Open3D might apply T directly (T @ pc)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def sync_data(disp_stamps,rgb_stamps):
    '''
    Use encoder time stamps to pair with lidar ranges,
    so the final time stamps will be encoder time stamps
    '''
    neighbbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(disp_stamps.reshape(-1,1))
    distances, indices = neighbbors.kneighbors(rgb_stamps.reshape(-1,1))
    return indices

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

if __name__ == '__main__':
    # init MAP
    print("numbers of disparity",disp_stamps.shape)
    print("numbers of rgb images",rgb_stamps.shape)
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -30  #meters
    MAP['ymin']  = -30
    MAP['xmax']  =  30
    MAP['ymax']  =  30 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'],3),dtype=np.uint8) #DATA TYPE: char or int8
    indices_dis = sync_data(disp_stamps,rgb_stamps)
    synced_lidar_indices = sync_data(lidar_stamsp,encoder_stamps)
    synced_lidar_stamps = lidar_stamsp[synced_lidar_indices]
    synced_lidar_indices = sync_data(synced_lidar_stamps,rgb_stamps)
    pose_store = np.load(f'results/Estimated_trajectory_dataset{dataset}_change_icp_input.npy')
    x_traj,y_traj = transform_pose_matrix_to_cells(pose_store)
    # minus 1 to match the index of poses
    pose_synced = pose_store[np.squeeze(synced_lidar_indices-1)] # (2289,4,4)
    # x_traj,y_traj = transform_pose_matrix_to_cells(pose_synced)
    # pdb.set_trace()
    # disparity images are more than rgb images
    for i, i_dis in enumerate(tqdm(np.squeeze(indices_dis))):
        # load RGBD image
        imd = cv2.imread(disp_path+f'disparity{dataset}_{i_dis}.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
        imc = cv2.imread(rgb_path+f'rgb{dataset}_{i+1}.png')[...,::-1] # (480 x 640 x 3)

        # print(imc.shape)
        # convert from disparity from uint16 to double
        disparity = imd.astype(np.float32)
        
        # get depth
        dd = (-0.00304 * disparity + 3.31)
        z = 1.03 / dd
        
        # calculate u and v coordinates 
        v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
        #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))
        
        
        # Rotation from optical frame to regular  
        R_r_to_o = np.array([[0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]])
        # test_otor_transform = homegenous_transformation(R_r_to_o,np.zeros(3))
        # get 3D coordinates
        fx = 585.05108211
        fy = 585.05108211
        cx = 315.83800193
        cy = 242.94140713
        x = (u-cx) / fx * z
        y = (v-cy) / fy * z

        # Test if the transformation to image is correct
        # image_cord = np.stack((x,y,z, np.ones(u.shape)),axis=2)
        # image_back = transformation_for_image(image_cord,test_otor_transform)

        # optical frame to kinect frame
        kinect_frame = np.stack((z,-x,-y,np.ones(z.shape)),axis=2)
        # yaw 0.021 pitch 0.36 Roll 0
        R_kninet_to_robot_center = Rotation.from_euler('zyx', [0.021, 0.36, 0]).as_matrix()
        # kinect frame to body frame
        T_to_rob_fr = homegenous_transformation(R_kninet_to_robot_center,np.array([0.18,0.005,0.36]))
        robot_body = pose_transformation_for_image(kinect_frame,T_to_rob_fr)
        # robot body to world frame
        image_world_frame = pose_transformation_for_image(robot_body,pose_synced[i])
        in_map = np.logical_and(
                        np.logical_and(MAP['xmin'] <= image_world_frame[..., 0],
                           image_world_frame[..., 0] <= MAP['xmax']),
                        np.logical_and(MAP['ymin'] <= image_world_frame[..., 1],
                           image_world_frame[..., 1] <= MAP['ymax']))
        x_w_fr_incell = np.ceil((image_world_frame[:,:,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        y_w_fr_incell = np.ceil((image_world_frame[:,:,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        # z_w_fr = np.ceil((image_world_frame[:,:,2] - MAP['zmin']) / MAP['res'] ).astype(np.int16)-1
        
        # calculate the location of each pixel in the RGB image
        rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
        rgbv = np.round((v * 526.37 + 16662.0)/fy)
        valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
        floor = image_world_frame[:,:,2] < 0.1
        floor = valid&floor&in_map
        MAP['map'][y_w_fr_incell[floor],x_w_fr_incell[floor],:] = imc[rgbv[floor].astype(int),rgbu[floor].astype(int)]
        # if i > 50:
        #     break

    color = 'blue'
    plt.imshow(MAP['map'])
    np.save("map/texture_mapping.npy",MAP['map'])
    plt.plot(x_traj,y_traj,color=plt.cm.cividis(0.15),label="Robot Trajectory",linewidth=3.0)
    plt.title(f"Texture mapping of dataset {dataset}")
    plt.legend()
    plt.savefig("map/Texture_mapping_cividis.png")
    plt.show()
    # # display valid RGB pixels
    # fig = plt.figure(figsize=(10, 13.3))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(z[valid],-x[valid],-y[valid],c=imc[rgbv[valid].astype(int),rgbu[valid].astype(int)]/255.0)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=0, azim=180)
    # plt.show()

    # # display disparity image
    # plt.imshow(normalize(imd), cmap='gray')
    # plt.show()