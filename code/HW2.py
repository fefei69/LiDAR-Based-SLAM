import numpy as np
import transforms3d
a = np.array([[10,0,160.5],
              [0,10,120.5],
              [0,0,1]])

b = np.array([[-1,0,0],
              [0,-1,0],
              [0,0,1]])

c = np.array([[-0.2,0,0],
              [0,-0.2,0],
              [0,0,1]])

r = np.array([[0,-1,0],
              [0,0,-1],
              [1,0,0]])
p = np.array([1,1,0])

ext = np.array([[0.707,-0.707,0,0],
               [0,0,-1,0],
               [0.707,0.707,0,1.414],
               [0,0,0,1]])

I = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0]])

m = np.array([2,1,2,1])
R = np.array([[0.173 + 0.707 + 0.707, -0.147 - 0.707 - 0.653, 0.974 + 0 + 0.271],
             [0.974 + 0.707 + 0.707, 0.173 + 0.707 + 0.653, -0.147 - 0.271],
             [-0.147, 0.974 + 0.383, 0.173 + 1 + 0.924]])

print(2*R)
U,S,V = np.linalg.svd(2*R)
print(np.linalg.svd(2*R))
MID = np.array([[1,0,0],
                [0,1,0],
                [0,0,np.linalg.det(U @ V)]])
R_optimal = U @ MID @ V
# print(R_optimal)
# print( V)
# K = a @ b @ c
# print(np.sin(45*np.pi/180))
# rot_bd1wd = transforms3d.euler.euler2mat(45*np.pi/180,0,0,'rzyx')
# optical_cord =np.array([0.707, -2, 3.535, 1])
# # print(optical_cord)
# image_cord = K @ I @ optical_cord.T / 3.535
# print(image_cord)
