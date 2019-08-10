from torch.utils.data import DataLoader

from dataloader.datasets import NYU2Dataset


def data_loader(args, **kwargs):
    if args.dataset == 'nyu':
        train_set = NYU2Dataset(args.data_path, args.crop_size)
        val_set = NYU2Dataset(args.data_path, train=False)
        print('Train_size: {}. Validation size: {}'.format(len(train_set), len(val_set)))

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader

    else:
        raise NotImplementedError

