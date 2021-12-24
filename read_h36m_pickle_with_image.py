import pickle
import cv2
import matplotlib.pyplot as plt 
from vis import show3Dpose, show2Dpose, show2Dpose_img

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

plt.ion()
fig = plt.figure(figsize=(12,12))
for idx in range(data['poses_2d_annot']['54138969'].shape[0]):       # 18432개의 frame 
    if idx % 4 == 0:
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

        vis_2d_cam0 = poses_2d['cam0']
        vis_2d_cam1 = poses_2d['cam1']
        vis_2d_cam2 = poses_2d['cam2']
        vis_2d_cam3 = poses_2d['cam3']

        vis_3d_cam0 = poses_3d['cam0']
        vis_3d_cam1 = poses_3d['cam1']
        vis_3d_cam2 = poses_3d['cam2']
        vis_3d_cam3 = poses_3d['cam3']
        
        img_cam0 = cv2.cvtColor(cv2.imread('../' + imageset['cam0']), cv2.COLOR_BGR2RGB)
        img_cam1 = cv2.cvtColor(cv2.imread('../' + imageset['cam1']), cv2.COLOR_BGR2RGB)
        img_cam2 = cv2.cvtColor(cv2.imread('../' + imageset['cam2']), cv2.COLOR_BGR2RGB)
        img_cam3 = cv2.cvtColor(cv2.imread('../' + imageset['cam3']), cv2.COLOR_BGR2RGB)


        ax = fig.add_subplot('241', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam0, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))

        ax = fig.add_subplot('242', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam1, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))

        ax = fig.add_subplot('243', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam2, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))

        ax = fig.add_subplot('244', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam3, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))
# -----------------------------------------------------------------------------------------
        ax = fig.add_subplot('245')
        show2Dpose_img(vis_2d_cam0, img_cam0, ax, data_type='h36m')

        ax = fig.add_subplot('246')
        show2Dpose_img(vis_2d_cam1, img_cam1, ax, data_type='h36m')

        ax = fig.add_subplot('247')
        show2Dpose_img(vis_2d_cam2, img_cam2, ax, data_type='h36m')

        ax = fig.add_subplot('248')
        show2Dpose_img(vis_2d_cam3, img_cam3, ax, data_type='h36m')

        plt.draw()
        plt.pause(0.01)
        fig.clear()
    
