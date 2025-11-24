import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# for vscode
import visdom
from tqdm import tqdm

from Projects.radarODE_plus.mmecg_dataset import MMECGDataSpliter
from Projects.radarODE_plus.utils.utils import shapeMetric, shapeLoss, ppiMetric, ppiLoss, anchorMetric, anchorLoss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.utils import set_random_seed, set_device
from LibMTL.model import resnet_dilated
from LibMTL import Trainer

from spectrum_dataset import dataset_concat, get_dataloader
from nets.PPI_decoder import PPI_decoder
from nets.anchor_decoder import anchor_decoder
from nets.model import backbone, shapeDecoder
from LibMTL.config import prepare_args
import argparse


def parse_args(parser):
    parser.add_argument('--train_bs', default=32, type=int,
                        help='batch size for training')
    parser.add_argument('--test_bs', default=32, type=int,
                        help='batch size for test')
    parser.add_argument('--epochs', default=200,
                        type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/',
                        type=str, help='dataset path')
    # if True, only select 100 samples for training and testing
    parser.add_argument('--select_sample', default=False,
                        type=bool, help='select sample')
    parser.add_argument('--aug_snr', default=100, type=int, help='100 for no aug otherwise the SNR')
    return parser.parse_args()

def norm_zs(ref_data):
    ref_mean = torch.mean(ref_data, dim=-1, keepdim=True)
    ref_std = torch.std(ref_data, dim=-1, keepdim=True)
    ref_data = (ref_data - ref_mean) / (ref_std + 1e-9)
    return ref_data


def cross_domain(params, domain, train_index, test_index, data_spliter, normalize_fn=norm_zs):
    viz = visdom.Visdom(env='cross domain', port=6006)
    # viz = None
    train_dataset, val_dataset, test_dataset = data_spliter.split_data(domain, train_index, test_index, need_val=True)
    train_loader = get_dataloader(train_dataset, shuffle=True, collate_fn=None, batch_size=params.train_bs)
    val_loader = get_dataloader(val_dataset, shuffle=False, collate_fn=None, batch_size=params.test_bs)
    test_loader = get_dataloader(test_dataset, shuffle=False, collate_fn=None, batch_size=params.test_bs)

    params.mode = 'train'
    main(params, train_loader, val_loader, test_loader)
    params.mode = 'test'
    main(params, None, None, test_loader)

    long_term_pcc = []
    for idx in test_index:
        params.mode = 'long_term_val'
        radar_trails, ref_trails = data_spliter.get_trails(domain, idx)
        radar_trails = torch.from_numpy(radar_trails).type(torch.float32).cuda()
        ref_trails = torch.from_numpy(ref_trails).type(torch.float32).cuda()
        recon_ecg = main(params, None, None, None, radar_trails)
        pcc, _ = batch_max_pearson_corr(ref_trails[:, :, :recon_ecg.size(-1)], recon_ecg, dim=-1, max_lag=100)
        pcc = torch.mean(pcc, dim=0)
        long_term_pcc.append(pcc)
        print(f'[validation] long term pcc {long_term_pcc}')
        model_name = 'RadarODE'

        recon_ecg = normalize_fn(recon_ecg)
        ref_data = normalize_fn(ref_trails)
        visualize_gen_curves(recon_ecg[0, 0], ref_data[0, 0, :len(recon_ecg[0, 0])], viz,
                             win=f'gen_ecg_lt_{domain}_{idx}_{model_name}',
                             title=f'gen_ecg_lt_{domain}_{idx}_{model_name}')


def batch_max_pearson_corr(x: torch.Tensor, y: torch.Tensor, dim=-1, max_lag: int = None, eps: float = 1e-8):
    """
    批量计算每条时间序列的最大皮尔逊相关系数及对应 lag。

    参数：
        x, y      : [batch, T] Tensor
        max_lag   : int, 最大考虑的正负时延
        eps       : float, 防止除零

    返回：
        max_pcc   : [batch] Tensor, 每条序列的最大 PCC
        best_lag  : [batch] Tensor, 每条序列对应 lag
    """
    if x.shape != y.shape:
        raise ValueError("x 和 y 必须形状相同")
    batch_size, C, T = x.shape
    if max_lag is None:
        max_lag = T - 1

    max_pcc = torch.full((batch_size,), -2.0, device=x.device, dtype=x.dtype)
    best_lag = torch.zeros((batch_size,), device=x.device, dtype=torch.int)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs = x[..., :T + lag]
            ys = y[..., -lag:]
        elif lag > 0:
            xs = x[..., lag:]
            ys = y[..., :T - lag]
        else:
            xs, ys = x, y

        xm = xs - xs.mean(dim=dim, keepdim=True)
        ym = ys - ys.mean(dim=dim, keepdim=True)

        r_num = torch.sum(xm * ym, dim=dim, keepdim=True)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=dim, keepdim=True) * torch.sum(ym ** 2, dim=dim, keepdim=True)) + eps
        pcc = r_num / r_den

        mask = pcc[:, 0, 0] > max_pcc
        max_pcc = torch.where(mask, pcc[:, 0, 0], max_pcc)
        best_lag = torch.where(mask, torch.full_like(best_lag, lag), best_lag)

    return max_pcc, best_lag


def visualize_gen_curves(gen_curves, ref_curves, viz, win='curves', title='Generated vs Reference'):
    """
    使用 visdom 可视化生成曲线和参考曲线

    参数:
        gen_curves : numpy.ndarray 或 torch.Tensor, shape [N] 或 [batch, N]
        ref_curves : numpy.ndarray 或 torch.Tensor, 同 shape
        viz        : visdom.Visdom 实例
        win        : str, 窗口名字
        title      : str, 窗口标题
    """
    if viz is None:
        return

    # 转 numpy
    if hasattr(gen_curves, "detach"):
        gen_curves = gen_curves.detach().cpu().numpy()
    if hasattr(ref_curves, "detach"):
        ref_curves = ref_curves.detach().cpu().numpy()

    # 如果是 batch，取第一条或平均
    if gen_curves.ndim > 1:
        gen_curves = gen_curves.mean(axis=0)
    if ref_curves.ndim > 1:
        ref_curves = ref_curves.mean(axis=0)

    x = np.arange(len(gen_curves))

    viz.line(
        X=np.column_stack([x, x]),
        Y=np.column_stack([gen_curves, ref_curves]),
        win=win,
        opts=dict(
            title=title,
            xlabel='Time step',
            ylabel='Value',
            legend=['Generated', 'Reference']
        )
    )


def main(params, train_loader, val_loader, test_loader, trails=None):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    # define tasks
    task_dict = {'ECG_shape': {'metrics': ['norm_MSE', 'MSE', 'CE'],
                               'metrics_fn': shapeMetric(),
                               'loss_fn': shapeLoss(),
                               'weight': [0, 0, 0]},
                 'PPI': {'metrics': ['PPI_sec', 'CE'],
                         'metrics_fn': ppiMetric(),
                         'loss_fn': ppiLoss(),
                         'weight': [0, 0]},
                 'Anchor': {'metrics': ['MSE'],
                            'metrics_fn': anchorMetric(),
                            'loss_fn': anchorLoss(),
                            'weight': [0]}}

    # # define backbone and en/decoders
    def encoder_class():
        return backbone(in_channels=50)

    num_out_channels = {'PPI': 260, 'Anchor': 200}
    decoders = nn.ModuleDict({'ECG_shape': shapeDecoder(),
                              'PPI': PPI_decoder(output_dim=num_out_channels['PPI']),
                              #   'Anchor': PPI_decoder(output_dim=num_out_channels['Anchor'])})
                              'Anchor': anchor_decoder()})

    class radarODE_plus(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, modelName, **kwargs):
            super(radarODE_plus, self).__init__(task_dict=task_dict,
                                                weighting=weighting,
                                                architecture=architecture,
                                                encoder_class=encoder_class,
                                                decoders=decoders,
                                                rep_grad=rep_grad,
                                                multi_input=multi_input,
                                                optim_param=optim_param,
                                                scheduler_param=scheduler_param,
                                                modelName=modelName,
                                                **kwargs)

    radarODE_plus_model = radarODE_plus(task_dict=task_dict,
                                        weighting=params.weighting,
                                        architecture=params.arch,
                                        encoder_class=encoder_class,
                                        decoders=decoders,
                                        rep_grad=params.rep_grad,
                                        multi_input=params.multi_input,
                                        optim_param=optim_param,
                                        scheduler_param=scheduler_param,
                                        save_path=params.save_path,
                                        load_path=params.load_path,
                                        modelName=params.save_name,
                                        **kwargs)
    if params.mode == 'train':
        radarODE_plus_model.train(train_loader, val_loader, params.epochs)
    elif params.mode == 'test':
        radarODE_plus_model.test(test_loader)
    elif params.mode == 'long_term_val':
        gen_ecg = radarODE_plus_model.long_term_validation(trails)
        return gen_ecg
    else:
        raise ValueError


if __name__ == "__main__":
    print(f'cuda is available {torch.cuda.is_available()}')
    n_epochs = 50
    batch_size = 22
    learning_rate = 5e-3
    lr_scheduler = 'cos'
    optimizer = 'sgd'
    weight_decay = 5e-4
    momentum = 0.937
    eta_min = learning_rate * 0.01
    T_max = 100

    params = parse_args(LibMTL_args)
    params.gpu_id = '0'

    params.dataset_path = '/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_step/'
    params.save_path = '/root/autodl-tmp/radarode/Projects/radarODE_plus/checkpoint/'

    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    params.train_bs, params.test_bs = batch_size, batch_size
    params.epochs = n_epochs
    params.weighting = 'EGA'
    params.EGA_temp = 1
    # 100 for no noise otherwise the SNR, 6,3,0,-1,-2,-3 for SNR, 101 for 1 sec extensive abrupt noise, 111 for 1 sec mild abrupt noise
    params.aug_snr = 100
    params.rep_grad = False
    params.multi_input = False
    params.arch = 'HPS'
    params.optim = optimizer
    params.lr, params.weight_decay, params.momentum = learning_rate, weight_decay, momentum
    params.scheduler = lr_scheduler
    params.eta_min, params.T_max = eta_min, T_max
    params.mode = 'train'
    params.save_name = f'{params.weighting}'

    set_random_seed(seed=1998)
    user = [i for i in range(11)]
    pcc_list = []
    for test_user in user:
        if test_user < 0:
            continue
        train_users = [j for j in user if j != test_user]
        cross_domain(params, domain=1, train_index=train_users, test_index=[test_user],
                     data_spliter=MMECGDataSpliter(rand_ref=True))
        # pcc_list.append(temp_pcc)
        # print(f'[test] short term pcc {pcc_list}')
