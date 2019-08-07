import os
import time
import numpy as np

import torch
import torch.nn as nn

from models import build_model
from utils.loss import ssim
from dataloader import data_loader
from utils.helper import AverageMeter, DepthNorm, set_random_seed, keras2torch_weights
from utils.params import set_params
from utils.keeper import Keeper


def main(args):
    log.info("Minion has spawn.")

    model = build_model(args.model_name)

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    l1_criterion = nn.L1Loss()

    log.info("loading data...")
    train_loader, val_loader = data_loader(args)

    # Whether using checkpoint
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise RuntimeError("=> no checkpoint found")
        checkpoint = torch.load(args.resume)
        # if args.use_cuda:
        #     model.module.load_state_dict(checkpoint['state_dict'])
        # else:
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        args.start_epoch = checkpoint['epoch'] + 1
    else:
        best_loss = np.inf

    # whether using pretrained model
    if args.pretrained_net is not None and args.resume is None:
        pretrained_w = keras2torch_weights(args.pretrained_net)
        model_dict = model.state_dict()
        pretrained_dict = {k: torch.from_numpy(v) for k, v in pretrained_w.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.cuda() if args.use_cuda else model

    # --------------- Start training ---------------
    for epoch in range(args.epochs):
        e_time = time.time()
        log.info('training: epoch {}/{} \n'.format(epoch+1, args.epochs))

        model.train()
        train_losses = AverageMeter()

        for i, (train_image, train_depth) in enumerate(train_loader):
            # print('train image size: {}'.format(train_image.size()))
            # print('train depth size: {}'.format(train_depth.size()))
            if args.use_cuda:
                train_image = train_image.cuda()
                train_depth = train_depth.cuda()

            optimizer.zero_grad()

            # Normalize depth
            depth_n = DepthNorm(train_depth, args.max_depth)
            # add channel dimension
            depth_n = depth_n.unsqueeze(1)

            train_out = model(train_image)

            train_loss_depth = l1_criterion(train_out, depth_n)
            train_loss_ssim = torch.clamp((1 - ssim(train_out, depth_n, val_range=args.max_depth / args.min_depth)) * 0.5, 0, 1)

            train_loss = (1.0 * train_loss_ssim) + (0.1 * train_loss_depth)

            train_losses.update(train_loss.data.item(), train_image.size(0))
            train_loss.backward()
            optimizer.step()

        log.info('Train loss of epoch {} is {}'.format(epoch,  train_losses.avg))
        keeper.save_loss([epoch, train_losses.count, train_losses.avg], 'train_loss.csv')
        keeper.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
        })

        # --------------- validation ---------------
        log.info('validation: epoch {}/{} \n'.format(epoch + 1, args.epochs))
        model.eval()
        val_losses = AverageMeter()

        for i, (val_image, val_depth) in enumerate(val_loader):
            if args.use_cuda:
                val_image = val_image.cuda()
                val_depth = val_depth.cuda()

            with torch.no_grad():
                val_depth_n = DepthNorm(val_depth, args.max_depth)
                val_out = model(val_image)

            if i % 100 == 1:
                keeper.save_img(val_image[1], val_depth[1], val_out[1], img_name='val_{}_{}'.format(epoch, i))

            val_loss_depth = l1_criterion(val_out, val_depth_n)
            val_loss_ssim = torch.clamp((1 - ssim(val_out, val_depth_n, val_range=args.max_depth / args.min_depth)) * 0.5, 0, 1)

            val_loss = (1.0 * val_loss_ssim) + (0.1 * val_loss_depth)

            val_losses.update(val_loss.data.item(), val_image.size(0))

        log.info('Validation loss of epoch {} is {}'.format(epoch, val_losses.avg))
        keeper.save_loss([epoch, val_losses.count, val_losses.avg], 'validation_loss.csv')

        if val_losses.avg < best_loss:
            keeper.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, 'best_model.pth')

        log.info('training time of epoch [%d/%d] is %d \n' % (epoch+1, args.epochs, time.time()-e_time))


if __name__ == '__main__':
    print('Welcome to summoner\'s rift')
    set_random_seed()
    args = set_params()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
    args.use_cuda = torch.cuda.is_available()

    start_time = time.time()

    keeper = Keeper(args)
    keeper.save_experiment_config()
    log = keeper.setup_logger()

    log.info("Thirty seconds until minion spawn!")

    main(args)

    log.info('Victory! Total game time is: {}'.format(time.time()-start_time))
