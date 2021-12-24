import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# root0, RHip1, RKnee2, RAnkle3, LHip4, Lknee5, KAnkle6, Neck7,Nose8,Head9,Lshoulder10,LElbow11,Lwist12,Rshoulder13,RElbow14,Rwist15
H36M_JOINTMAP = [
    [0,1],      
    [1,2],      #
    [2,3],      #
    [0,4],      
    [4,5],
    [5,6],
    [0,7],
    [7,8],
    [8,9],
    [7,10],
    [7,13],
    [10,11],
    [11,12],
    [13,14],
    [14,15]
    ]

MPII_JOINTMAP = [
    [0,1],      
    [1,2],      #
    [3,4],      #
    [4,5],      
    [6,0],
    [6,3],
    [6,7],
    [7,8],
    [8,9],
    [7,10],
    [7,13],
    [10,11],
    [11,12],
    [13,14],
    [14,15]
    ]


def show2Dpose(vis_2d_poses, ax,data_type='h36m', image_size=(1280,1000)):
    img = np.zeros((image_size[0], image_size[1], 3), np.uint8)

    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
    else:
        JOINTMAP = MPII_JOINTMAP

    json_keypoints = vis_2d_poses
    # json_keypoints = vis_2d_poses.T
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 

    for j in range(len(JOINTMAP)):
        child = tuple(np.array(json_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(json_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        #color = (0, 64 + 192 * (len(JOINTMAP) - j) / (len(JOINTMAP) - 1), 64 + 192 * j / (len(JOINTMAP) - 1))
        color = (255, 255, 255)
        cv2.circle(img, parent, 8, (255, 0, 0), -1)
        cv2.line(img, child, parent, color, 8) 

    plt.imshow(img)

def show2Dpose_img(vis_2d_poses, img, ax, data_type='h36m'):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
    else:
        JOINTMAP = MPII_JOINTMAP

    json_keypoints = vis_2d_poses
    # json_keypoints = vis_2d_poses.T
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 

    for j in range(len(JOINTMAP)):
        child = tuple(np.array(json_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(json_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        #color = (0, 64 + 192 * (len(JOINTMAP) - j) / (len(JOINTMAP) - 1), 64 + 192 * j / (len(JOINTMAP) - 1))
        color = (255, 255, 255)
        cv2.circle(img, parent, 8, (255, 0, 0), -1)
        cv2.line(img, child, parent, color, 8) 

    plt.imshow(img)


def show2Dpose2(vis_2d_poses, ax, data_type='h36m', cam_view='cam0', image_size=(1280,1000)):
    img1 = np.zeros((image_size[0], image_size[1], 3), np.uint8)
    img2 = np.zeros((image_size[0], image_size[1], 3), np.uint8)
    img3 = np.zeros((image_size[0], image_size[1], 3), np.uint8)
    img4 = np.zeros((image_size[0], image_size[1], 3), np.uint8)

    img_list = [img1,img2,img3,img4]
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
    else:
        JOINTMAP = MPII_JOINTMAP

    json_keypoints = vis_2d_poses[cam_view].reshape(2,16).T
    # json_keypoints = vis_2d_poses.T
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    cam_views = int(cam_view[-1])
    for j in range(len(JOINTMAP)):
        child = tuple(np.array(json_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(json_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        color = (0, 64 + 192 * (len(JOINTMAP) - j) / (len(JOINTMAP) - 1), 64 + 192 * j / (len(JOINTMAP) - 1))

        cv2.circle(img_list[cam_views], parent, 10, (255, 0, 0), -1)
        cv2.line(img_list[cam_views], child, parent, color, 2) 

    plt.imshow(img_list[cam_views])


def show3Dpose(channels, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    vals = channels
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
    else:
        JOINTMAP = MPII_JOINTMAP

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return [-RADIUS + xroot, RADIUS + xroot], [-RADIUS + yroot, RADIUS + yroot], [-RADIUS + zroot, RADIUS + zroot]

def show3Dpose_with_annot(annot, channels, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    vals = channels
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
    else:
        JOINTMAP = MPII_JOINTMAP

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([annot[i, c], annot[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c='blue')

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return [-RADIUS + xroot, RADIUS + xroot], [-RADIUS + yroot, RADIUS + yroot], [-RADIUS + zroot, RADIUS + zroot]