import os

import numpy as np
import torch
from tqdm import tqdm
import visdom

from Projects.radarODE_plus.spectrum_dataset import DataSpliter, BaseDataset, down_sample, add_gaussian_sst, \
    add_abrupt_sst, load_mat_auto

USER = [i + 1 for i in range(11)]
STATUS = [i + 1 for i in range(4)]
FILE_FORMAT = 'u{user}_st{status}_{sample}'
FILE_KEY_FORMAT = 'u{user}_st{status}'
RADAR_TRAIL_FORMAT = 'radar_u{user}_st{status}_id{id}.npy'
REF_TRAIL_FORMAT = 'ecg_u{user}_st{status}_id{id}.npy'
POS_TRAIL_FORMAT = 'pos_u{user}_st{status}_id{id}.npy'
RADAR_TRAIL_FORMAT = 'radar_u{user}_st{status}_id{id}.npy'

class MMECGDataset(BaseDataset):
    def __init__(self, sst_ecg_root, filenames, sample2ecg_info, aug_snr=100, align_length=200):
        super().__init__(sst_ecg_root, filenames, sample2ecg_info, aug_snr, align_length)

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def preprocessing_radar(radar):
        radar_max = np.max(radar, keepdims=True)
        radar_min = np.min(radar, keepdims=True)
        radar = (radar - radar_min) / (radar_max - radar_min + 1e-7)
        return radar

    @staticmethod
    def preprocessing_ref(ref):
        ref_max = np.max(ref, keepdims=True)
        ref_min = np.min(ref, keepdims=True)
        ref = (ref - ref_min) / (ref_max - ref_min + 1e-7)
        return ref[:, 0]

    @staticmethod
    def preprocessing_anchor(anchor):
        anchor = np.transpose(anchor, (1, 0))
        return anchor

    @staticmethod
    def find_median_ppi_segment_start(signal):
        r_locs = np.where(signal == 1)[0]
        # 1. 找出所有 R 波位置
        if len(r_locs) < 2:
            raise ValueError("R 峰数量不足以计算 PPI")

        # 2. 计算所有 PPI
        ppis = np.diff(r_locs)  # [r2-r1, r3-r2, ...]

        # 3. 找到中位数 PPI
        median_ppi = np.median(ppis)

        # 4. 找到与中位数最接近的 PPI 的索引
        idx = np.argmin(np.abs(ppis - median_ppi))

        return r_locs[idx], r_locs[idx + 1]

    def __getitem__(self, index):
        sst_filename, ecg_filename, anchor_filename, es, et, ss, st = self.get_inf(index)
        # sst = load_mat_auto(os.path.join(self.sst_ecg_root, 'sst/30Hz_half_01', sst_filename))
        # sst = sst['SST'][ss:st, :, :]
        sst = np.load(os.path.join(self.sst_ecg_root, 'sst/30Hz_half_01', sst_filename))[:, :, ss:st]
        ref = np.load(os.path.join(self.sst_ecg_root, 'trails', ecg_filename))
        anchor_mask = np.load(os.path.join(self.sst_ecg_root, 'anchor', anchor_filename))
        anchor = np.zeros_like(ref)
        anchor[anchor_mask] = 1
        anchor = anchor[es: et]
        ecg_s, ecg_t = self.find_median_ppi_segment_start(anchor[:, 0])
        ref = ref[es:et][ecg_s:ecg_t]
        target_len = 260
        sst, ref, anchor = self.preprocessing_radar(sst), self.preprocessing_ref(ref), self.preprocessing_anchor(anchor)

        ppi_info = np.pad(ref, (0, target_len - ref.shape[-1]), 'constant', constant_values=-10)
        ecg_target = down_sample(ref, target_len=200)[None, :]
        if self.aug_snr < 100:
            sst = add_gaussian_sst(sst, self.aug_snr)
        if self.aug_snr > 100 and index in self.index_select:
            sst = add_abrupt_sst(sst, self.aug_snr % 100)
        sst = torch.from_numpy(sst).type(torch.float32)
        ecg_target = torch.from_numpy(ecg_target).type(torch.float32)
        anchor = torch.from_numpy(anchor).type(torch.float32)
        return sst, {'ECG_shape': ecg_target, 'PPI': ppi_info, 'Anchor': anchor}


class MMECGDataSpliter(DataSpliter):
    def __init__(self, data_root=r"/root/autodl-tmp/dataset/mmecg", rand_ref=False, train_transform=None,
                 val_transform=None, train_ratio=0.8, num_domain=4, n_fold=5):
        super().__init__(data_root, rand_ref, train_transform, val_transform, train_ratio, num_domain,
                         n_fold)

        self.sample_fold.extend([[] for _ in range(max(n_fold, len(USER), len(STATUS)))])

    def organize_data(self, domain):
        if self.pre_domain == domain:
            return
        else:
            self.pre_domain = domain
        cur_domain_idx = [0 for _ in range(self.num_domain)]

        for u_id, u in enumerate(USER):
            cur_domain_idx[1] = u_id
            for s_id, s in enumerate(STATUS):
                cur_domain_idx[2] = s_id
                temp_filenames = []
                i = 0
                while True:
                    filename = FILE_FORMAT.format(user=u, status=s, sample=i)
                    sample_key = FILE_KEY_FORMAT.format(user=u, status=s)
                    if sample_key not in self.sample2file_info or len(self.sample2file_info[sample_key]) <= i:
                        break
                    temp_filenames.append(filename)
                    i += 1
                if len(temp_filenames) == 0:
                    continue
                temp_filenames = np.array(temp_filenames)
                if domain == 0:
                    rand_idx = np.arange(len(temp_filenames))
                    np.random.shuffle(rand_idx)
                    rand_idx = self.split_list(rand_idx, self.num_fold)
                    for i in range(self.num_fold):
                        self.sample_fold[i].extend(temp_filenames[rand_idx[i]])
                else:
                    self.sample_fold[cur_domain_idx[domain]].extend(temp_filenames)

    def get_trails(self, domain, index):
        # filename_list = os.listdir(os.path.join(self.data_root, 'sst/30Hz_half_01'))
        cur_domain_idx = [0 for _ in range(self.num_domain)]
        radar_trails = []
        ref_trails = []
        pos_trails = []
        for u_id, u in enumerate(USER):
            cur_domain_idx[1] = u_id
            for s_id, s in enumerate(STATUS):
                cur_domain_idx[2] = s_id
                if cur_domain_idx[domain] == index:
                    sample_key = FILE_KEY_FORMAT.format(user=u, status=s)
                    if sample_key not in self.sample2file_info:
                        continue
                    info_list = self.sample2file_info[sample_key]
                    trail_set = set()
                    for info in info_list:
                        sst_trial_filename = info['sst_fn']
                        if sst_trial_filename not in trail_set:
                            radar_trails.append(np.load(os.path.join(self.data_root, 'sst/30Hz_half_01', sst_trial_filename)))
                            trail_set.add(sst_trial_filename)
                            ref_trails.append(np.load(os.path.join(self.data_root, 'trails', info['ecg_fn'])))
        radar_trails = np.array(radar_trails)
        ref_trails = np.transpose(np.array(ref_trails), (0, 2, 1))
        return radar_trails, ref_trails

    def split_data(self, domain, train_idx=(0, 1), test_idx=(0, 1), need_val=True):
        data = super(MMECGDataSpliter, self).split_data(domain, train_idx, test_idx, need_val)
        tr, vl, te = data
        # sst_ecg_root, filenames, sample2ecg_info
        return MMECGDataset(self.data_root, tr, self.sample2file_info), \
               MMECGDataset(self.data_root, vl, self.sample2file_info), \
               MMECGDataset(self.data_root, te, self.sample2file_info)


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

def resave_sst(data_root='/root/autodl-tmp/dataset/mmecg/sst/30Hz_half_01'):
    filenames = os.listdir(data_root)
    for filename in tqdm(filenames):
        sst = load_mat_auto(os.path.join(data_root, filename))
        sst = sst['SST'][:, :, :]
        sst =np.transpose(sst, (2, 1, 0))
        new_filename = filename.split('.')[0] + '.npy'
        np.save(os.path.join(data_root, new_filename), sst)
        os.remove(os.path.join(data_root, filename))


if __name__ == '__main__':
    split_mmecg_raw_data()
    # resave_sst()