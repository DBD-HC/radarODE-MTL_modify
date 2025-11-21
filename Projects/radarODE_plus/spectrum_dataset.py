import os, re
import scipy.io as sio
import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
from tqdm import tqdm

RADAR_TRAIL_FORMAT = 'radar_u{user}_st{status}_id{id}.npy'
REF_TRAIL_FORMAT = 'ecg_u{user}_st{status}_id{id}.npy'
POS_TRAIL_FORMAT = 'pos_u{user}_st{status}_id{id}.npy'
RADAR_FILE_FORMAT = 'radar_u{user}_st{status}_s{sample}.npy'
REF_FILE_FORMAT = 'ecg_u{user}_st{status}_s{sample}.npy'
POS_FILE_FORMAT = 'pos_u{user}_st{status}_s{sample}.npy'
SST_FILE_FORMAT = 'sst_u{user}_st{status}_s{sample}.npy'
ANCHOR_FILE_FORMAT = 'anchor_u{user}_st{status}_s{sample}.npy'
USER = [i + 1 for i in range(11)]
STATUS = [i + 1 for i in range(4)]


def normal_ecg_torch_01(ECG):
    for itr in range(ECG.size(dim=0)):
        ECG[itr] = (ECG[itr] - torch.min(ECG[itr])) / \
                   (torch.max(ECG[itr]) - torch.min(ECG[itr]))
    return ECG


def normal_ecg(ECG):
    ECG = (ECG - np.min(ECG)) / (np.max(ECG) - np.min(ECG))
    return ECG


def normal_ecg_11(ECG):
    k = 2 / (np.max(ECG) - np.min(ECG))
    ECG = -1 + k * (ECG - np.min(ECG))
    return ECG


def get_all_files_in_directory(directory):
    file_paths = []
    # 使用os.walk遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for i in range(len(files) // 3):
            sst_ecg_pair = []
            sst_ecg_pair.append(os.path.join(
                root, "sst_seg_" + str(i) + '.npy'))
            sst_ecg_pair.append(os.path.join(root, "ecg_seg_" + str(i) + '.npy'))
            sst_ecg_pair.append(os.path.join(
                root, "anchor_seg_" + str(i) + '.npy'))
            file_paths.append(sst_ecg_pair)

    return file_paths


# add gaussian noise with certqin SNR to sst
def add_gaussian_sst(sst, snr_db):
    for i in range(sst.shape[0]):
        # 计算信号功率
        signal_power = np.mean(sst[i] ** 2)

        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # 生成高斯噪声
        noise = np.sqrt(noise_power) * np.random.randn(*sst[i].shape)
        sst[i] = sst[i] + noise
    return sst


# add abrupt noise with certqin length to sst
def add_abrupt_sst(sst, length=1):  # length 1 sec (length - 100)
    snr_db = -9  # extensive noise
    if length > 10:
        snr_db = 0  # mild noise
        length -= 10
    length = int(length * 30)
    length = sst.shape[-1] - 1 if length > sst.shape[-1] else length
    start = np.random.randint(0, sst.shape[-1] - length)
    # print(sst.shape)
    for i in range(sst.shape[0]):
        # 计算信号功率
        signal_power = np.mean(sst[i][:, start:start + length] ** 2)
        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        # 生成高斯噪声
        noise = np.sqrt(noise_power) * np.random.randn(*sst[i][:, start:start + length].shape)
        sst[i][:, start:start + length] = sst[i][:, start:start + length] + noise
    return sst


def down_sample(ecg, target_len=None, orig_fs=200, target_fs=30):
    if target_len is not None:
        ecg = np.interp(np.linspace(0, len(ecg), target_len),
                        np.arange(len(ecg)), ecg)
        return ecg
    else:
        # 原始时间轴
        t_orig = np.linspace(0, 1, len(ecg), endpoint=False)
        # 新时间轴
        t_new = np.linspace(0, 1, int(len(ecg) * target_fs / orig_fs), endpoint=False)
        # 线性插值
        return np.interp(t_new, t_orig, ecg)


def des_path_finder(index, path):
    for roots, dirs, files in os.walk(path):
        for dir_ in dirs:
            if re.search(f'_{index}_', dir_):
                return os.path.join(roots, dir_)


class SpectrumECGDataset(Dataset):
    def __init__(self, sst_ecg_root, filenames, sample2ecg_info, aug_snr=100, align_length=200):
        super().__init__()
        self.filenames = filenames
        self.sample2ecg_inf = sample2ecg_info
        self.sst_ecg_root = sst_ecg_root
        self.align_length = align_length
        self.aug_snr = aug_snr
        # self.all_sst_ecg_files = get_all_files_in_directory(self.sst_ecg_root)
        self.index_select = np.random.choice(
            np.arange(len(self.all_sst_ecg_files)), size=int(20 / 100 * len(self.all_sst_ecg_files)),
            replace=False)  # used for abrupt nosing (20%)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        index = index % len(self.all_sst_ecg_files)
        sst_ecg_path = self.all_sst_ecg_files[index]
        sst_data, ecg_data, anchor_data = np.load(sst_ecg_path[0]), np.load(
            sst_ecg_path[1]), np.load(sst_ecg_path[2])
        target_len = 260
        # pad ecg_data to target length with -100
        ppi_info = np.pad(ecg_data, (0, target_len -
                                     ecg_data.shape[-1]), 'constant', constant_values=-10)

        ecg_data = np.expand_dims(down_sample(ecg_data), 0)
        ppi_info = np.expand_dims(((ppi_info)), 0)
        anchor_data = np.expand_dims((anchor_data), 0)
        # sst is the normlized sst data, ppi_info is the original ecg signal with -10 padding to length of 260, ecg_data is the resampled ecg data with length of 200, anchor_data represent the position of the R peak in the ecg signal

        if self.aug_snr < 100:
            sst_data = add_gaussian_sst(sst_data, self.aug_snr)
        if self.aug_snr > 100 and index in self.index_select:
            sst_data = add_abrupt_sst(sst_data, self.aug_snr % 100)

        sst_data = torch.from_numpy(np.array(sst_data)).type(torch.FloatTensor)
        ecg_data = torch.from_numpy(np.array(ecg_data)).type(torch.FloatTensor)
        ppi_info = torch.from_numpy(np.array(ppi_info)).type(torch.FloatTensor)
        anchor_data = torch.from_numpy(np.array(anchor_data)).type(torch.FloatTensor)
        return sst_data, {'ECG_shape': ecg_data, 'PPI': ppi_info, 'Anchor': anchor_data}


class BaseDataset(Dataset):
    def __init__(self, sst_ecg_root, filenames, sample2ecg_info, aug_snr=100, align_length=200):
        super().__init__()
        self.filenames = filenames
        self.sample2ecg_inf = sample2ecg_info
        self.sst_ecg_root = sst_ecg_root
        self.align_length = align_length
        self.aug_snr = aug_snr
        self.index_select = np.random.choice(
            np.arange(len(self.filenames)), size=int(20 / 100 * len(self.filenames)),
            replace=False)  # used for abrupt nosing (20%)

    def __len__(self):
        return len(self.filenames)

    def get_inf(self, index):
        filename = self.filenames[index]
        info = filename.split('_')
        filename_key = info[0] + '_' + info[1]
        sample2file = self.sample2ecg_inf[filename_key][int(info[-1])]
        es, et, ss, st = sample2file['re_s'], sample2file['re_t'], sample2file['sst_s'], sample2file['sst_t']
        ecg_filename = sample2file['ecg_fn']
        sst_filename = sample2file['sst_fn']
        anchor_filename = sample2file['anchor_fn']
        return sst_filename, ecg_filename, anchor_filename, es, et, ss, st

    def __getitem__(self, index):
        pass


def get_dataloader(data_set, shuffle, batch_size, collate_fn, sw=None, num_workers=8, drop_last=False):
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers,
                                         worker_init_fn=sw,
                                         pin_memory=True,
                                         collate_fn=collate_fn, drop_last=drop_last)
    return loader


class DataSpliter:
    def __init__(self, data_root=None, rand_ref=False, train_transform=None, val_transform=None,
                 train_ratio=0.8, num_domain=4, n_fold=5):
        self.data_root = data_root
        if self.data_root is None:
            return
        self.pre_domain = -1
        self.sample_fold = []
        self.rand_ref = rand_ref
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_ratio = train_ratio
        self.num_domain = num_domain
        self.num_fold = n_fold
        self.sample2file_info = np.load(os.path.join(self.data_root, 'samples', 'radarode_sample2file_info.npy'),
                                        allow_pickle=True).item()

    @staticmethod
    def split_list(lst, n_parts=5):
        length = len(lst)
        if length == 0:
            return [[] for _ in range(n_parts)]
        k, m = divmod(length, n_parts)  # k=每份基础长度, m=余数
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]

    def organize_data(self, domain):
        pass

    def get_trails(self, domain, index):
        pass

    def get_dataset(self, index):
        samples = []
        samples.extend(self.sample_fold[index])
        # (self, sst_ecg_root, filenames, sample2ecg_info, aug_snr=100, align_length=200)
        return BaseDataset(self.data_root, samples, self.sample2file_info)

    def split_data(self, domain, train_idx=(0, 1), test_idx=(0, 1), need_val=True):
        self.organize_data(domain)
        train_data, val_data, test_data = [], [], []

        for i in train_idx:
            train_data.extend(self.sample_fold[i])
        for i in test_idx:
            test_data.extend(self.sample_fold[i])
        if need_val:
            train_data_len = int(len(train_data) * self.train_ratio)
            rand_idx = np.arange(len(train_data))
            np.random.shuffle(rand_idx)
            train_data = np.array(train_data)
            val_data.extend(train_data[rand_idx[train_data_len:]])
            train_data = train_data[rand_idx[:train_data_len]]

        return train_data, val_data, test_data


def dataset_concat(ID_selected, data_root, aug_snr=0):
    dataset = []
    for ID in ID_selected:
        ID_path = des_path_finder(ID, data_root)
        dataset = ConcatDataset([dataset, BaseDataset(sst_ecg_root=ID_path, aug_snr=aug_snr)])
    return dataset


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


def split_mmecg_raw_data(data_root='/root/autodl-tmp/dataset/mmecg'):
    rcg_ecg_path = os.path.join(data_root, 'finalPartialPublicData20221108', '{sample_id}.mat')
    ecg_fs = 200
    sst_fs = 30
    ecg_sample_len = 4 * ecg_fs
    ecg_step = 2 * ecg_fs
    sst_sample_len = 4 * sst_fs
    sst_sample_step = 2 * sst_fs
    uid_set = {}
    status_set = {}
    sample_count_set = {}
    user_count = 0
    status_count = 0
    file_sample_info = {}
    for ti in range(1, 92):
        mat_map = load_mat_auto(rcg_ecg_path.format(sample_id=ti))
        data_struct = mat_map['data']
        radar_data = data_struct['RCG'][0, 0]
        ref_data = data_struct['ECG'][0, 0]
        user_id = data_struct['id'][0, 0][0, 0]
        status = data_struct['physistatus'][0, 0][0]
        if user_id not in uid_set:
            user_count += 1
            uid_set[user_id] = user_count
        if status not in status_set:
            status_count += 1
            status_set[status] = status_count
        sample_key = f"u{uid_set[user_id]}_st{status_set[status]}"
        num_samples = (len(radar_data) - ecg_sample_len) // ecg_step + 1
        sample_count = 0
        if sample_key not in sample_count_set:
            sample_count_set[sample_key] = 0
            file_sample_info[sample_key] = []
        else:
            print(f'sample dup {sample_key}')
            sample_count = sample_count_set[sample_key]
        radar_trail_name = RADAR_TRAIL_FORMAT.format(user=uid_set[user_id], status=status_set[status], id=ti)
        ref_trail_name = REF_TRAIL_FORMAT.format(user=uid_set[user_id], status=status_set[status], id=ti)
        pos_trail_name = POS_TRAIL_FORMAT.format(user=uid_set[user_id], status=status_set[status], id=ti)
        sst_trail_name = 'SST_{sample_id}.npy'.format(sample_id=ti)
        anchor_trail_name = 'anchor_{sample_id}.npy'.format(sample_id=ti)
        for i in tqdm(range(num_samples)):
            sample_index = i + sample_count
            ecg_s = i * ecg_step
            ecg_t = ecg_s + ecg_sample_len
            sst_s = i * sst_sample_step
            sst_t = sst_s + sst_sample_len
            file_sample_info[sample_key].append({
                'radar_fn': radar_trail_name,
                'pos_fn': pos_trail_name,
                'ecg_fn': ref_trail_name,
                'sst_fn': sst_trail_name,
                'anchor_fn': anchor_trail_name,
                're_s': ecg_s,
                're_t': ecg_t,
                'sst_s': sst_s,
                'sst_t': sst_t
            })
        sample_count_set[sample_key] = sample_count + num_samples
    np.save(os.path.join(data_root, 'samples', 'radarode_sample2file_info.npy'), file_sample_info)
    print(f'status {status_set}')
    print(f'user {uid_set}')


'''
root = '/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_step/'
    ID = np.arange(3, 7)
    dataset = dataset_concat(ID, root, aug_snr=101)
    print(ID, len(dataset))
    count = 0
    for item in dataset:
        print(item[0].size(), item[1]['ECG_shape'].size(), item[1]['PPI'].size(), item[1]['Anchor'].size())
        # break

'''

if __name__ == '__main__':
    split_mmecg_raw_data()
