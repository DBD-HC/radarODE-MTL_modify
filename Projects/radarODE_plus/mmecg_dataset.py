import os

import numpy as np
import torch

from Projects.radarODE_plus.spectrum_dataset import DataSpliter, BaseDataset, down_sample, add_gaussian_sst, \
    add_abrupt_sst

USER = [i + 1 for i in range(11)]
STATUS = [i + 1 for i in range(4)]
FILE_FORMAT = 'u{user}_st{status}_{sample}'
FILE_KEY_FORMAT = 'u{user}_st{status}'
RADAR_TRAIL_FORMAT = 'radar_u{user}_st{status}_id{id}.npy'
REF_TRAIL_FORMAT = 'ecg_u{user}_st{status}_id{id}.npy'
POS_TRAIL_FORMAT = 'pos_u{user}_st{status}_id{id}.npy'


class MMECGDataset(BaseDataset):
    def __init__(self, sst_ecg_root, filenames, sample2ecg_info, aug_snr=100, align_length=200):
        super().__init__(sst_ecg_root, filenames, sample2ecg_info, aug_snr, align_length)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        sst_filename, ecg_filename, anchor_filename, es, et, ss, st = self.get_inf(index)
        d = np.load(os.path.join(self.sst_ecg_root, 'sst/30Hz_half_01', sst_filename))[:, :, ss:st]
        ref = np.load(os.path.join(self.sst_ecg_root, 'trails', ecg_filename))
        anchor_mask = np.load(os.path.join(self.sst_ecg_root, 'anchor', anchor_filename))
        anchor = np.zeros_like(ref)
        anchor[anchor_mask] = 1
        anchor = anchor[es: et]
        ref = ref[es:et]
        target_len = 260
        d, ref, anchor = self.preprocessing_radar(d), self.preprocessing_ref(ref), self.preprocessing_anchor(anchor)
        ecg_origin = down_sample(ref, target_len=None)
        ppi_info = np.pad(ecg_origin, (0, target_len - ecg_origin.shape[-1]), 'constant', constant_values=-10)

        ecg_target = down_sample(ref, target_len=200)[None, :]
        if self.aug_snr < 100:
            d = add_gaussian_sst(d, self.aug_snr)
        if self.aug_snr > 100 and index in self.index_select:
            d = add_abrupt_sst(d, self.aug_snr % 100)
        d = torch.from_numpy(d).type(torch.float32)
        ecg_target = torch.from_numpy(ecg_target).type(torch.float32)
        anchor = torch.from_numpy(anchor).type(torch.float32)
        return d, {'ECG_shape': ecg_target, 'PPI': ppi_info, 'Anchor': anchor}


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
        filename_list = os.listdir(os.path.join(self.data_root, 'trails'))
        num_files = len(filename_list) // 3
        cur_domain_idx = [0 for _ in range(self.num_domain)]
        radar_trails = []
        ref_trails = []
        pos_trails = []
        for u_id, u in enumerate(USER):
            cur_domain_idx[1] = u_id
            for s_id, s in enumerate(STATUS):
                cur_domain_idx[2] = s_id
                if cur_domain_idx[domain] == index:
                    trail_id = 0
                    while trail_id < num_files:
                        radar_filename = RADAR_TRAIL_FORMAT.format(user=u, status=s, id=trail_id)
                        ref_filename = REF_TRAIL_FORMAT.format(user=u, status=s, id=trail_id)
                        pos_filename = POS_TRAIL_FORMAT.format(user=u, status=s, id=trail_id)
                        trail_id += 1
                        if radar_filename not in filename_list:
                            continue
                        radar_trails.append(np.load(os.path.join(self.data_root, 'trails', radar_filename)))
                        ref_trails.append(np.load(os.path.join(self.data_root, 'trails', ref_filename)))
                        pos_trails.append(np.load(os.path.join(self.data_root, 'trails', pos_filename)))

        radar_trails = np.transpose(np.array(radar_trails), (0, 2, 1))
        ref_trails = np.transpose(np.array(ref_trails), (0, 2, 1))
        pos_trails = np.array(pos_trails)
        return radar_trails, ref_trails, pos_trails

    def split_data(self, domain, train_idx=(0, 1), test_idx=(0, 1), need_val=True):
        data = super(MMECGDataSpliter, self).split_data(domain, train_idx, test_idx, need_val)
        tr, vl, te = data
        # sst_ecg_root, filenames, sample2ecg_info
        return MMECGDataset(self.data_root, tr, self.sample2file_info), \
               MMECGDataset(self.data_root, vl, self.sample2file_info), \
               MMECGDataset(self.data_root, te, self.sample2file_info)