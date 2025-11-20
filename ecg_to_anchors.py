import os

import h5py
import numpy as np
import neurokit2 as nk
import scipy.io as sio

# Parameters

c = 0
f_s = 200  # Sampling rate
f_d = 30  # Desired resampling rate

# data_folder = '/root/autodl-tmp/dataset/mmecg/finalPartialPublicData20221108'
data_folder = r'I:\dataset\MMECG202211\finalPartialPublicData20221108'
# save_folder = f'/root/autodl-tmp/dataset/mmecg/achor'
save_folder = r'I:\dataset\MMECG202211\anchor'
os.makedirs(save_folder, exist_ok=True)

def load_mat_auto(path):
    """
    自动检测 MATLAB .mat 文件版本并读取
    - v7 及以下：使用 scipy.io.loadmat
    - v7.3 (HDF5 格式)：使用 h5py
    """
    # 先读取文件头前 128 个字节，里面包含版本信息
    with open(path, 'rb') as f:
        header = f.read(128)

    # MATLAB v7.3 的文件头里包含 "MATLAB 7.3 MAT-file"
    if b'MATLAB 7.3' in header:
        # print("[INFO] Detected MATLAB v7.3 (HDF5) format, using h5py...")
        data = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                data[k] = f[k][()]
        return data
    else:
        # print("[INFO] Detected MATLAB v7 or lower format, using scipy.io.loadmat...")
        return sio.loadmat(path)

# Main loop
for ID in range(1, 92):
    print(f"---------obj {ID}---------")
    data_path = os.path.join(data_folder, f'{ID}.mat')
    data = load_mat_auto(data_path)
    data = data['data']
    ecgSignal = data['ECG'][0, 0][:, 0]
    print(f'get obj {ID} anchors')
    signals, info = nk.ecg_process(ecgSignal, sampling_rate=f_s)
    r_peaks = info["ECG_R_Peaks"]  # R峰索引位置（采样点）

    # Save SST
    save_path = os.path.join(save_folder, f'anchor_{ID}.npy')
    np.save(save_path, r_peaks)