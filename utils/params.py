import argparse


def set_params():
    parser = argparse.ArgumentParser(description='Depth Estimation using Monocular Images')
    parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'kitti'])
    parser.add_argument('--data_path', type=str, default='data/')

    # train
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default=['0'], help='IDs of GPUs to use')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='w-decay (default: 5e-4)')
    parser.add_argument('--min_depth', type=float, default=10.0, help='Minimum of input depths')
    parser.add_argument('--max_depth', type=float, default=1000.0, help='Maximum of input depths')

    # model
    parser.add_argument('--model_name', type=str, default='ResDeep', choices=['DenseDepth', 'ResDeep'])
    parser.add_argument('--pretrained_net', type=str, default='models/pretrained/nyu.h5')
    parser.add_argument('--resume', type=str, default=None, help='Start training from an existing model.')
    parser.add_argument('--save_path', type=str, default='train_results/')

    args = parser.parse_args()
    return args






