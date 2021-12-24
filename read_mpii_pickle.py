import pickle
import matplotlib.pyplot as plt 
from vis import show3Dpose, show2Dpose

json_file_path = "mpii_all_data.pickle"
with open(json_file_path, 'rb') as f:
    data = pickle.load(f)

cams = ['cam0', 'cam2', 'cam7', 'cam8']
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

sample = dict()
anno3d = dict()


plt.ion()
fig = plt.figure(figsize=(12,12))
for idx in range(data['poses_2d_annot']['cam0'].shape[0]):       # 18432개의 frame 
    if idx % 4 == 0 and idx>130:
        for c_idx, cam in enumerate(cams):
            p2d = data['poses_2d_annot'][cam][idx]
            p3d = data['poses_3d_annot'][cam][idx]

            sample['cam' + str(c_idx)] = p2d
            anno3d['cam' + str(c_idx)] = p3d
            print(cam)
            print(c_idx)

        poses_2d = {key:sample[key] for key in all_cams} 
        poses_3d = {key:anno3d[key] for key in all_cams}    # 카메라0,1,2,3 순으로 

        vis_2d_cam0 = poses_2d['cam0'].reshape(2,16)
        vis_2d_cam1 = poses_2d['cam1'].reshape(2,16)
        vis_2d_cam2 = poses_2d['cam2'].reshape(2,16)
        vis_2d_cam3 = poses_2d['cam3'].reshape(2,16)

        vis_3d_cam0 = poses_3d['cam0'].reshape(3,16)
        vis_3d_cam1 = poses_3d['cam1'].reshape(3,16)
        vis_3d_cam2 = poses_3d['cam2'].reshape(3,16)
        vis_3d_cam3 = poses_3d['cam3'].reshape(3,16)

        ax = fig.add_subplot('241', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam0.T, ax, data_type='mpii', radius=1000, lcolor='blue',angles=(30,-120))

        ax = fig.add_subplot('242', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam1.T, ax, data_type='mpii', radius=1000, lcolor='blue',angles=(30,-120))

        ax = fig.add_subplot('243', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam2.T, ax, data_type='mpii', radius=1000, lcolor='blue',angles=(30,-120))

        ax = fig.add_subplot('244', projection='3d', aspect='auto')
        show3Dpose(vis_3d_cam3.T, ax, data_type='mpii', radius=1000, lcolor='blue',angles=(30,-120))
# -----------------------------------------------------------------------------------------
        ax = fig.add_subplot('245')
        show2Dpose(vis_2d_cam0.T, ax, data_type='mpii', image_size=(2024,2024))

        ax = fig.add_subplot('246')
        show2Dpose(vis_2d_cam1.T, ax, data_type='mpii', image_size=(2024,2024))

        ax = fig.add_subplot('247')
        show2Dpose(vis_2d_cam2.T, ax, data_type='mpii', image_size=(2024,2024))

        ax = fig.add_subplot('248')
        show2Dpose(vis_2d_cam3.T, ax, data_type='mpii', image_size=(2024,2024))

        plt.draw()
        plt.pause(0.01)
        fig.clear()
    
