import os
import time
import torch
import glob
import logging
import pickle
import matplotlib.pyplot as plt
import pandas as pd


class Keeper(object):
    def __init__(self, args):
        self.args = args
        start_time = time.time()
        str_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(start_time))
        run_id = str_time + '-e' + str(args.epochs) + '-bs' + str(
            args.batch_size) + '-lr' + str(args.lr) + '-' + args.dataset

        self.experiment_dir = os.path.join(self.args.save_path, run_id)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def save_experiment_config(self):
        config_file = os.path.join(self.experiment_dir, 'parameters.txt')

        with open(config_file, 'wb') as f:
            pickle.dump(self.args, f)

    def save_img(self, input, target, pred, img_name='test.jpg'):
        img_path = os.path.join(self.experiment_dir, 'val_img')
        if self.args.use_cuda:
            input, target, pred = input.cpu().numpy(), target.cpu().numpy(), pred.cpu().numpy()
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img_file = os.path.join(img_path, img_name)
        plt.subplot(221)
        plt.imshow(input)
        plt.subplot(222)
        plt.imshow(target)
        plt.subplot(223)
        plt.savefig(img_file)
        plt.clf()

    def save_loss(self, loss_list, file_name='train_loss.csv'):
        loss_file = os.path.join(self.experiment_dir, file_name)
        loss_dict = {}
        for i, loss in enumerate(loss_list):
            loss_dict[i] = [loss]
        all_loss = pd.DataFrame(loss_dict)
        all_loss.to_csv(loss_file, mode='a', header=False)

    def setup_logger(self, file_name='train.log'):
        log_file = os.path.join(self.experiment_dir, file_name)
        file_formatter = logging.Formatter(
            "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        logger = logging.getLogger('example')
        handler = logging.StreamHandler()
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)

        file_handle_name = "file"
        if file_handle_name in [h.name for h in logger.handlers]:
            return
        file_handle = logging.FileHandler(filename=log_file, mode="a")
        file_handle.set_name(file_handle_name)
        file_handle.setFormatter(file_formatter)
        logger.addHandler(file_handle)
        logger.setLevel(logging.DEBUG)
        return logger
