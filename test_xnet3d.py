from torchvision import transforms, datasets
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
from torch.backends import cudnn
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchio as tio

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_3d
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_iit
from config.train_test_config.train_test_config import save_test_3d
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pd', '--path_dataset', default='/mnt/data1/XNet/dataset/LiTS')
    parser.add_argument('-p', '--path_model', default='/mnt/data1/XNet/pretrained_model/semi_xnet/LiTS/best_result1_Jc_0.7794.pth')
    parser.add_argument('--path_seg_results', default='/mnt/data1/XNet/seg_pred/test')
    parser.add_argument('--dataset_name', default='LiTS', help='LiTS, Atrial')
    parser.add_argument('--input1', default='L')
    parser.add_argument('--input2', default='H')
    parser.add_argument('--threshold', default=None)
    parser.add_argument('--result', default='result1', help='result1, result2')
    parser.add_argument('--patch_size', default=(112, 112, 32))
    parser.add_argument('--patch_overlap', default=(56, 56, 16))
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-n', '--network', default='xnet3d')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    # Config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # Results Save
    if not os.path.exists(args.path_seg_results) and rank == args.rank_index:
        os.mkdir(args.path_seg_results)
    path_seg_results = args.path_seg_results + '/' + str(dataset_name)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results + '/' + str(os.path.splitext(os.path.split(args.path_model)[1])[0])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)

    data_transform = data_transform_3d(cfg['NORMALIZE'])
    dataset_val = dataset_iit(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        input2=args.input2,
        transform_1=data_transform['test'],
    )

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()

    # if rank == args.rank_index:
    #     state_dict = torch.load(args.path_model, map_location=torch.device(args.local_rank))
    #     model.load_state_dict(state_dict=state_dict)
    # model = DistributedDataParallel(model, device_ids=[args.local_rank])

    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    state_dict = torch.load(args.path_model)
    model.load_state_dict(state_dict=state_dict)
    dist.barrier()

    # Test
    since = time.time()

    for i, subject in enumerate(dataset_val.dataset_1):

        grid_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap
        )

        # val_sampler = torch.utils.data.distributed.DistributedSampler(grid_sampler, shuffle=False)

        dataloaders = dict()
        dataloaders['test'] = DataLoader(grid_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
        # dataloaders['test'] = DataLoader(grid_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16, sampler=val_sampler)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

        with torch.no_grad():
            model.eval()

            for data in dataloaders['test']:

                inputs_test_1 = Variable(data['image'][tio.DATA].cuda())
                inputs_test_2 = Variable(data['image2'][tio.DATA].cuda())
                location_test = data[tio.LOCATION]

                outputs_test_1, outputs_test_2 = model(inputs_test_1, inputs_test_2)
                if args.result == 'result1':
                    outputs_test = outputs_test_1
                else:
                    outputs_test = outputs_test_2

                aggregator.add_batch(outputs_test, location_test)

        outputs_tensor = aggregator.get_output_tensor()
        save_test_3d(cfg['NUM_CLASSES'], outputs_tensor, subject['ID'], args.threshold, path_seg_results, subject['image']['affine'])


    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)