import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt 
from vis import show3Dpose, show2Dpose, show3Dpose_with_annot, show2Dpose_img
from three_D_transform import linear_eigen_triangulation, estimate_relative_pose_from_correspondence, polynomial_triangulation


json_file_path = "h36m_all_data.pickle"
with open(json_file_path, 'rb') as f:
    data = pickle.load(f)

json_file_path = "h36m_all_image_data.pickle"
with open(json_file_path, 'rb') as f:
    img_data = pickle.load(f)

cams = ["54138969","55011271","58860488","60457274"]
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

sample = dict()
anno3d = dict()
imagesames = dict()

intri = [data['intrinsics'][cams[0]],data['intrinsics'][cams[1]],data['intrinsics'][cams[2]],data['intrinsics'][cams[3]]]

# def regular_normalized3d(pose):
#     root_joints = pose.T[:, [0]]                                     
#     pose_norm = np.linalg.norm((pose.T - root_joints).reshape(-1, 48), ord=2, axis=1, keepdims=True)                     
#     pose /= pose_norm

#     return pose

def regular_normalized3d(poseset):
    pose_norm_list = []
    for i in range(len(poseset)):
        root_joints = poseset[i].T[:, [0]]                                     
        pose_norm = np.linalg.norm((poseset[i].T - root_joints).reshape(-1, 48), ord=2, axis=1, keepdims=True)                     
        poseset[i] /= pose_norm
        pose_norm_list.append(pose_norm)

    return poseset, np.array(pose_norm_list)

# def regular_normalized3d(pose):
#     root_joints = np.transpose(pose,(0,2,1))[:, :, [0]]     
#     new_pose = (np.transpose(pose,(0,2,1)) - root_joints)#.reshape(-1, 48)                  
#     pose_norm = np.linalg.norm(new_pose, ord=2, axis=1, keepdims=True)                     
#     new_pose /= pose_norm
#     new_pose = new_pose.reshape(-1,16,3)
#     return new_pose

# def regular_normalized3d(pose):
#     root_joints = pose.reshape(-1, 3, 16)[:, :, [0]]                                                # 추가 코드
#     pose = (pose.reshape(-1, 3, 16) - root_joints).reshape(-1, 48)
#     pose_norm = np.linalg.norm(pose, ord=2, axis=1, keepdims=True)                        # 추가 코드
#     pose /= pose_norm
#     pose = pose.reshape(-1,16,3)
#     return pose

def scaled_normalized3d(pose):    # 
    pose = pose.reshape(-1,48)
    scale_pose = np.sqrt(np.square(pose[:, 0:48]).sum(axis=1) / 48).reshape(-1,1)
    scaled_pose = pose[:, 0:48]/scale_pose
    return scaled_pose.reshape(-1,16,3)

def get_3d_triangulation(view_2d_pose,intri,view1=0,view2=1):
    _, R_est, T_est = estimate_relative_pose_from_correspondence(view_2d_pose[view2],view_2d_pose[view1],intri[view2],intri[view1])   # three_D_transform.py , R이랑 T를 추정 
    p_nom = np.dot(intri[view2],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    proj = np.dot(intri[view1],np.concatenate((R_est,T_est),axis=1))  
    triang_3d, _ = polynomial_triangulation(view_2d_pose[view2], p_nom, view_2d_pose[view1], proj) 
    triang_3d = triang_3d.T
    triang_3d = np.array([triang_3d[0],triang_3d[2],-triang_3d[1]]).T
    return triang_3d.reshape(-1,16,3)

Radius=1000
mpjpes = []
plt.ion()
fig = plt.figure(figsize=(12,12))
for idx in range(data['poses_2d_annot'][cams[0]].shape[0]):       # 18432개의 frame 
    if idx % 64 == 0:
        for c_idx, cam in enumerate(cams):
            p2d = data['poses_2d_annot'][cam][idx]
            p3d = data['poses_3d_annot'][cam][idx]

            sample['cam' + str(c_idx)] = p2d
            anno3d['cam' + str(c_idx)] = p3d

            img = img_data['image'][cam][idx]
            imagesames['cam' + str(c_idx)] = img

        poses_2d = {key:sample[key] for key in all_cams} 
        poses_3d = {key:anno3d[key] for key in all_cams}    # 카메라0,1,2,3 순으로 
        imageset = {key:imagesames[key] for key in all_cams} 

        view_2d_pose = np.array([poses_2d['cam0'],poses_2d['cam1'],poses_2d['cam2'],poses_2d['cam3']])
        view_3d_pose = np.array([poses_3d['cam0'],poses_3d['cam1'],poses_3d['cam2'],poses_3d['cam3']])
        # view_3d_pose = np.array([regular_normalized3d(poses_3d['cam0']),regular_normalized3d(poses_3d['cam1']),regular_normalized3d(poses_3d['cam2']),regular_normalized3d(poses_3d['cam3'])])

        triang_3d = get_3d_triangulation(view_2d_pose,intri,view1=0,view2=1)

        origin_3d_pose = view_3d_pose.copy()
        view_3d_pose, pose_norm = regular_normalized3d(view_3d_pose) 
        #view_3d_pose = scaled_normalized3d(view_3d_pose) * 4           # : scaling 필요없었음 

        triang_3d, _ = regular_normalized3d(triang_3d) 
        #triang_3d = scaled_normalized3d(triang_3d)[0] * 4

        view_3d_pose = view_3d_pose * pose_norm
        triang_3d = triang_3d[0] * pose_norm[1] # pose_norm[1]인 이유는 1번째 view의 triangulation이라서 

        mpjpe1 = np.mean(np.sqrt(np.sum((triang_3d - view_3d_pose[1])**2, axis=1)))      # 1번째 view 3d tria이랑 view_3d_pose[1] 1번째
        print(mpjpe1)
        # mpjpe2 = np.mean(np.sqrt(np.sum((origin_3d_pose[1] - view_3d_pose[1])**2, axis=1)))
        # print('origin3dannot & denorm3dannot', round(mpjpe2,2))                             # 0 나옴
        # mpjpe3 = np.mean(np.sqrt(np.sum((triang_3d - origin_3d_pose[1])**2, axis=1)))       # denorm과 비교랑 똑같음
        # print('trian3d & origin3dannot', round(mpjpe3,2))
        # print()
        mpjpes.append(mpjpe1)

        img_cam0 = cv2.cvtColor(cv2.imread('../' + imageset['cam0']), cv2.COLOR_BGR2RGB)
        img_cam1 = cv2.cvtColor(cv2.imread('../' + imageset['cam1']), cv2.COLOR_BGR2RGB)
        img_cam2 = cv2.cvtColor(cv2.imread('../' + imageset['cam2']), cv2.COLOR_BGR2RGB)
        img_cam3 = cv2.cvtColor(cv2.imread('../' + imageset['cam3']), cv2.COLOR_BGR2RGB)

        ax = fig.add_subplot('251', projection='3d', aspect='auto')
        show3Dpose(view_3d_pose[0], ax, data_type='h36m', radius=Radius, lcolor='blue',angles=(20,-60))

        ax = fig.add_subplot('252', projection='3d', aspect='auto')
        show3Dpose_with_annot(view_3d_pose[1],origin_3d_pose[1], ax, data_type='h36m', radius=Radius, lcolor='red',angles=(20,-60)) # normalize했다가 denorm한거, 원래annot

        ax = fig.add_subplot('253', projection='3d', aspect='auto')
        show3Dpose(view_3d_pose[2], ax, data_type='h36m', radius=Radius, lcolor='blue',angles=(20,-60))

        ax = fig.add_subplot('254', projection='3d', aspect='auto')
        show3Dpose(view_3d_pose[3], ax, data_type='h36m', radius=Radius, lcolor='blue',angles=(20,-60))
#v ---
        ax = fig.add_subplot('255', projection='3d', aspect='auto')
        # show3Dpose(triang_3d, ax, data_type='h36m', radius=1, lcolor='red',angles=(20,-60))
        show3Dpose_with_annot(view_3d_pose[1], triang_3d, ax, data_type='h36m', radius=Radius, lcolor='red',angles=(20,-60))
# -----------------------------------------------------------------------------------------
        ax = fig.add_subplot('256')
        show2Dpose(view_2d_pose[0], ax, data_type='h36m', image_size=(1000,1002))

        ax = fig.add_subplot('257')
        show2Dpose(view_2d_pose[1], ax, data_type='h36m', image_size=(1000,1002))

        ax = fig.add_subplot('258')
        show2Dpose(view_2d_pose[2], ax, data_type='h36m', image_size=(1000,1002))

        ax = fig.add_subplot('259')
        show2Dpose(view_2d_pose[3], ax, data_type='h36m', image_size=(1000,1002))

        ax = fig.add_subplot(2,5,10)
        show2Dpose_img(view_2d_pose[1], img_cam1, ax, data_type='h36m')


        plt.draw()
        plt.savefig('1.png')
        plt.pause(0.01)
        fig.clear()
    
print(np.mean(mpjpes))