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

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_itn
from config.train_test_config.train_test_config import print_test_eval, save_test_2d
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

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pd', '--path_dataset', default='dataset/GlaS')
    parser.add_argument('-p', '--path_model', default='checkpoints/sup_xnet/GlaS/Glas_unsup_xnet-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-20-80-L-H/best_result1_Jc_0.8080.pth')
    parser.add_argument('--path_seg_results', default='seg_pred/sup_xnet')
    parser.add_argument('--dataset_name', default='GlaS', help='CREMI')
    parser.add_argument('--input1', default='image')
    #parser.add_argument('--input2', default='H')
    parser.add_argument('--if_mask', default=True)
    parser.add_argument('--threshold', default=0.5400, help='0.5600, 5400')
    parser.add_argument('-ds', '--deep_supervision', default=False)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-n', '--network', default='xnet')
    #parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    args = parser.parse_args()

    #local_rank
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    print(f"Using local_rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    #rank = torch.distributed.get_rank()
    rank = dist.get_rank()
    print(f"Rank of current process: {rank}")
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
    # print(path_seg_results)

    if args.input1 == 'image':
        input1_mean = 'MEAN'
        input1_std = 'STD'
    else:
        input1_mean = 'MEAN_' + args.input1
        input1_std = 'STD_' + args.input1

    # Dataset
    data_transforms = data_transform_2d()
    data_normalize = data_normalize_2d(cfg[input1_mean], cfg[input1_std])

    dataset_val = imagefloder_itn(
        data_dir=args.path_dataset + '/val',
        input1=args.input1,
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        sup=True,
        num_images=None
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16, sampler=val_sampler)

    num_batches = {'val': len(dataloaders['val'])}

    # Model
    model = get_network(args.network, in_channels=1, num_classes=2)
    model = model.cuda()

    # if rank == args.rank_index:
    #     state_dict = torch.load(args.path_model, map_location=torch.device(args.local_rank))
    #     model.load_state_dict(state_dict=state_dict)
    # model = DistributedDataParallel(model, device_ids=[args.local_rank])

    model = DistributedDataParallel(model, device_ids=[local_rank])
    state_dict = torch.load(args.path_model)
    model.load_state_dict(state_dict=state_dict)
    dist.barrier()

    # Test
    since = time.time()

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(dataloaders['val']):
            inputs_test = data['image']
            inputs_test = Variable(inputs_test.cuda(non_blocking=True))
            name_test = data['ID']
            if args.if_mask:
                mask_test = data['mask']
                mask_test = Variable(mask_test.cuda(non_blocking=True))

            outputs_test = model(inputs_test)
            if args.deep_supervision:
                outputs_test = outputs_test[0]

            if args.if_mask:
                if i == 0:
                    score_list_test = outputs_test
                    name_list_test = name_test
                    mask_list_test = mask_test
                else:
                # elif 0 < i <= num_batches['val'] / 16:
                    score_list_test = torch.cat((score_list_test, outputs_test), dim=0)
                    name_list_test = np.append(name_list_test, name_test, axis=0)
                    mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
                torch.cuda.empty_cache()
            else:
                save_test_2d(cfg['NUM_CLASSES'], outputs_test, name_test, args.threshold, path_seg_results, cfg['PALETTE'])
                torch.cuda.empty_cache()

        if args.if_mask:
            score_gather_list_test = [torch.zeros_like(score_list_test) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_test, score_list_test)
            score_list_test = torch.cat(score_gather_list_test, dim=0)

            mask_gather_list_test = [torch.zeros_like(mask_list_test) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_test, mask_list_test)
            mask_list_test = torch.cat(mask_gather_list_test, dim=0)

            name_gather_list_test = [None for _ in range(ngpus_per_node)]
            torch.distributed.all_gather_object(name_gather_list_test, name_list_test)
            name_list_test = np.concatenate(name_gather_list_test, axis=0)

        if args.if_mask and rank == args.rank_index:
            print('=' * print_num)
            test_eval_list = print_test_eval(cfg['NUM_CLASSES'], score_list_test, mask_list_test, print_num_minus)
            save_test_2d(cfg['NUM_CLASSES'], score_list_test, name_list_test, test_eval_list[0], path_seg_results, cfg['PALETTE'])
            torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)
    cleanup()