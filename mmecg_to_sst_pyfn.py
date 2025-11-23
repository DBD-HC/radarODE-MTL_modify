import os

import h5py
import numpy as np
import scipy.io as sio
import visdom
from scipy.signal import resample, savgol_filter, firwin, lfilter
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, Wavelet
from scipy.interpolate import interp1d

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


def exact_matlab_wsst_match(x, fs, voices_per_octave=10):
    """
    更精确地匹配MATLAB的wsst函数
    """
    N = len(x)

    # MATLAB wsst 的默认频率范围
    # 通常从 2*fs/N 到 fs/2
    min_freq = 1 * fs / N
    max_freq = fs / 2

    # 计算八度数和尺度数
    num_octaves = np.log2(max_freq / min_freq)
    nv = int(voices_per_octave * num_octaves)

    # 创建精确的频率数组（类似MATLAB）
    freq = 2 ** np.linspace(
        np.log2(min_freq),
        np.log2(max_freq),
        nv
    )

    # 对于Morlet小波，尺度与频率的关系为: scale = center_frequency / frequency
    # Morlet小波的中心频率约为0.8125
    center_freq = 0.8125
    scales = center_freq * fs / freq

    # 确保尺度数组是递减的（与MATLAB一致）
    scales = scales[::-1]
    freq = freq[::-1]

    # 执行SSQ-CWT
    Tx, freq_out, scales_out, w = ssq_cwt(
        x,
        wavelet='morlet',
        scales=scales,
        fs=fs,
        padtype='symmetric'
    )

    return Tx, freq_out, scales_out


viz = visdom.Visdom(env='cross domain', port=6006)
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
        # cwt, Wx, ssq_freqs, scales = ssq_cwt(RCG, wavelet='gmw', fs=f_s)
        cwt, _, _ = exact_matlab_wsst_match(RCG, f_s)
        # cwt, scales = ssq_cwt(RCG, fs=f_s, nv=10)  # nv: Voices per octave
        freq_len = cwt.shape[0]
        SST.append(np.abs(cwt[freq_len // 2:, :]))
        print(f"{s + 1}, ", end="")

    SST = np.array(SST)

    # Resample in time axis
    print(f"\nResample with {f_d}Hz")
    viz.heatmap(X=SST[0], win='1')
    SST = resample_sst(SST, f_s, f_d)

    # Normalize [0,1]
    SST_max = np.max(SST, axis=(-2, -1), keepdims=True)
    SST_min = np.min(SST, axis=(-2, -1), keepdims=True)
    SST = (SST - SST_min) / (SST_max - SST_min + 1e-9)

    # Save SST
    save_path = os.path.join(save_folder, f'SST_{ID}.npy')
    # np.save(save_path, SST)


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
