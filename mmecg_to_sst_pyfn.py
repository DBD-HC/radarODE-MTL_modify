import os

import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import resample, savgol_filter, firwin, lfilter
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, Wavelet

# Parameters
c = 0
f_s = 200  # Sampling rate
f_d = 30  # Desired resampling rate
data_folder = '/root/autodl-tmp/dataset/mmecg/finalPartialPublicData20221108'
save_folder = f'/root/autodl-tmp/dataset/mmecg/sst/{f_d}Hz_half_01'
os.makedirs(save_folder, exist_ok=True)


def resample_sst(sst, f_org, f_desired):
    x_sampled = np.zeros((sst.shape[0], sst.shape[1], int(sst.shape[2] * f_desired / f_org)))
    for i in range(sst.shape[0]):
        for j in range(sst.shape[1]):
            x_sampled[i, j, :] = resample(sst[i, j, :], int(sst.shape[2] * f_desired / f_org))
    return x_sampled

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
    ecgSignal = data['ECG'][0, 0]
    RCG_data = data['RCG'][0, 0]
    status = data['physistatus'][0, 0][0]
    user_id = data['id'][0, 0][0, 0]
    SST = []
    print("SST for Point ", end="")
    for s in range(50):
        RCG = RCG_data[:, s]
        # WSST equivalent using ssqueezepy (CWT-based SST)

        T = len(RCG) / f_s
        f_min = 1 / T
        f_max = f_s / 2
        nv = 10

        # 计算尺度上下界 (与 MATLAB 对齐)
        fc = 0.849
        scale_max = fc * f_s / f_min
        scale_min = fc * f_s / f_max

        # 生成等比尺度序列（决定频率点数）
        n_octaves = np.log2(scale_max / scale_min)
        Nfreq = int(nv * n_octaves)  # = MATLAB 的频率点数

        scales = scale_min * 2 ** (np.arange(Nfreq) / nv)
        cwt, Wx, ssq_freqs, scales = ssq_cwt(RCG, wavelet='morlet', fs=f_s, scales=scales)
        # cwt, scales = ssq_cwt(RCG, fs=f_s, nv=10)  # nv: Voices per octave
        freq_len = cwt.shape[0]
        SST.append(np.abs(cwt[freq_len // 2:, :]))
        print(f"{s + 1}, ", end="")

    SST = np.array(SST)

    # Resample in time axis
    print(f"\nResample with {f_d}Hz")
    SST = resample_sst(SST, f_s, f_d)

    # Normalize [0,1]
    SST_max = np.max(SST, axis=(-2, -1), keepdims=True)
    SST_min = np.min(SST, axis=(-2, -1), keepdims=True)
    SST = (SST - SST_min) / (SST_max - SST_min + 1e-9)

    # Save SST
    save_path = os.path.join(save_folder, f'SST_{ID}.npy')
    np.save(save_path, SST)


# ECG smoothing
def smooth_ECG(ecgSignal, Fs):
    ecgSignal = savgol_filter(ecgSignal, 19, 9)  # Polynomial order=9, window=19

    # FIR LPF
    Fpass = 10
    Fstop = 40
    b = firwin(numtaps=101, cutoff=[Fpass, Fstop], fs=Fs, pass_zero=True)
    ecg_smooth = lfilter(b, 1, ecgSignal)
    return ecg_smooth


# Optional: plot 3D points
def plot3Dpoint(posXYZ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = posXYZ[:, 0], posXYZ[:, 1], posXYZ[:, 2]
    ax.scatter(x, y, z)
    plt.show()
