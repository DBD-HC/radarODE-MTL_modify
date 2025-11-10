import os
import numpy as np
import neurokit2 as nk
# Parameters
from mmecg_to_sst_pyfn import load_mat_auto

c = 0
f_s = 200  # Sampling rate
f_d = 30  # Desired resampling rate

# data_folder = '/root/autodl-tmp/dataset/mmecg/finalPartialPublicData20221108'
data_folder = r'I:\dataset\MMECG202211\finalPartialPublicData20221108'
# save_folder = f'/root/autodl-tmp/dataset/mmecg/achor'
save_folder = r'I:\dataset\MMECG202211\anchor'
os.makedirs(save_folder, exist_ok=True)


# Main loop
for ID in range(1, 92):
    print(f"---------obj {ID}---------")
    data_path = os.path.join(data_folder, f'{ID}.mat')
    data = load_mat_auto(data_path)
    data = data['data']
    ecgSignal = data['ECG'][0, 0]
    print(f'get obj {ID} anchors')
    signals, info = nk.ecg_process(ecgSignal, sampling_rate=f_s)
    r_peaks = info["ECG_R_Peaks"]  # R峰索引位置（采样点）

    # Save SST
    save_path = os.path.join(save_folder, f'anchor_{ID}.npy')
    np.save(save_path, r_peaks)