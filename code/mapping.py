from pr2_utils import *
from lidar_scan_matching import *

def generate_valid_scan():
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    ranges = synced_lidar_ranges[:,0]
    valid_indices = np.logical_and((ranges < 30),(ranges> 0.1))
    return ranges[valid_indices], angles[valid_indices]

# ranges = synced_lidar_ranges[:,0]
angles = np.arange(-135,135.25,0.25)*np.pi/180.0
# take valid indices
ranges, angles = generate_valid_scan()
# Transform lidar frame to world frame
x_lidar = []
y_lidar = []
for i in range(ranges.shape[0]):
    x = ranges[i] * np.cos(angles[i])
    y = ranges[i] * np.sin(angles[i])
    # lidar frame to robot body frame
    x_lidar.append(x-0.135)
    y_lidar.append(y)
x_lidar = np.asarray(x_lidar)
y_lidar = np.asarray(y_lidar)
lidar_pc = np.stack((x_lidar,y_lidar,np.zeros(x_lidar.shape),np.ones(x_lidar.shape))).T
lidar_in_world_frame = lidar_pc @ POSE[0] 
# plt.plot(lidar_in_world_frame[:,0],lidar_in_world_frame[:,1],"k.")
# plt.show()
sx = POSE[0][:3, 3][0]
sy = POSE[0][:3, 3][1]
occupied = []
for i in range(lidar_in_world_frame.shape[0]):
    ex = lidar_in_world_frame[:,0][i]
    ey = lidar_in_world_frame[:,1][i]
    # print(ex,ey)
    occupied.append(bresenham2D(sx,sy,ex,ey))
    # pdb.set_trace()
# occupied = np.asarray(occupied)
# print(occupied.shape)
test_mapCorrelation(synced_lidar_ranges,POSE,lidar_in_world_frame,ODOMETRY)
# print(occupied)
# print(lidar_in_world_frame.shape)
# plt.plot(lidar_in_world_frame[:,0],lidar_in_world_frame[:,1],"k.")
# plt.plot(lidar_pc[:,0],lidar_pc[:,1],"r.")
# plt.show()
# test_mapCorrelation(synced_lidar_ranges[:,0])
# plt.plot(ODOMETRY[:,0],ODOMETRY[:,1],"r",label="Odometry",linewidth=2.0)
# plt.show()
# bresenham2D(0,0,3,3)