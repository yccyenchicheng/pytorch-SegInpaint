"""
usage: python main.py --gpu_ids 0 --batch_size 2
"""
import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
import util.util as util
from dataset.cityscapes_dataset import CityscapesDataset
from util.visualizer import Visualizer
from trainers.seg_inpaint_trainer import SegInpaintTrainer

current_time = datetime.now().strftime("%m%d-%H%M")

def get_opt():
    parser = argparse.ArgumentParser()
    ### base options ###
    parser.add_argument('--name', type=str, default='exp1-%s' % current_time, help="name of this experiment")
    parser.add_argument('--phase', type=str, default='train', help="'train' or 'test'")
    parser.add_argument('--gpu_ids', type=str, default='0,2', help="0,1,2 corresponds to GPU 2,0,1 (weird)")
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')

    # input/output sizes
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')

    # for setting input
    parser.add_argument('--dataset', type=str, default='cityscapes') # dataroot will be at: 'server'_data/cityscapes
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--mask', type=int, default=3, choices=[1, 2, 3, 4, 5], help='1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)')

    # for instance-wise features
    parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
    #####################

    # for deeplab
    parser.add_argument('--deeplab_backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--deeplab_output_stride', type=int, default=16, help='network output stride (default: 8)')

    ### train options ###
    # for displays
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
    parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

    # for training
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')

    # for discriminator
    parser.add_argument('--ndf', type=int, default=64, help='how many D to use in multiscale discriminator')
    parser.add_argument('--num_D', type=int, default=3, help='how many D to use in multiscale discriminator')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    #####################

    parser.add_argument('--test', action='store_true')

    # copy from original code. add more command_line_option
    dataset_mode = 'cityscapes'
    isTrain = True
    dataset_option_setter = dataset.get_option_setter(dataset_mode)
    parser = dataset_option_setter(parser, isTrain)

    opt = parser.parse_args()

    opt.isTrain = isTrain

    return opt

with open('latest_cmd.txt', 'w') as f:
    cmd = ' '.join(sys.argv) + '\n'
    f.write(cmd)

opt = get_opt()
opt.dataroot = os.path.join('data', opt.dataset)
opt.total_epochs = opt.niter + opt.niter_decay

# logs and checkpoing
if not os.path.exists('logs'):
    os.mkdir('logs')

log_root = os.path.join('logs', 'seg_inpaint_logs')
if not os.path.exists(log_root):
    os.mkdir(log_root)

opt.log_root = log_root
exp_dir = os.path.join(log_root, opt.name)
util.mkdir(exp_dir)
ckpt_dir = os.path.join(log_root, opt.name, 'checkpoint')
util.mkdir(ckpt_dir)

dataset = CityscapesDataset()
dataset.initialize(opt)
dataloader = DataLoader(dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=opt.isTrain)

# setup GPU, optimizer.
# borrow from SPADE
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
    "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
    % (opt.batch_size, len(opt.gpu_ids))

trainer = SegInpaintTrainer(opt)

# create tool for visualization
visualizer = Visualizer(opt)

total_steps_so_far = 0

for epoch in range(opt.total_epochs):

    for i, data_i in tqdm(enumerate(dataloader), total=len(dataloader)):
        current_step = epoch*len(dataloader) + i

        # Training
        if not opt.test or i == 0:
            trainer.run_generator_one_step(data_i)
            trainer.run_discriminator_one_step(data_i)
        else:
            pass

        if current_step % 10 == 0:
            if i != 0:
                sys.stdout.write("\033[F") # back to previous line
                sys.stdout.write("\033[K") # clear line

            loss_str = trainer.get_loss_str()
            print("[%d/%d] %d %s" % (epoch, opt.total_epochs, current_step, loss_str))
            with open(visualizer.log_name, "a") as log_file:
                log_file.write('%s\n' % loss_str)
 

        if current_step % 50 == 0:
            real_img, corruped_img, generated_seg, generated_img = \
                trainer.get_latest_results()
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', generated_img),
                                   ('synthesized_segmentation', generated_seg),
                                   ('real_image', real_img),                                  
                                   ('corruped_image', corruped_img)],
                                   )
            visualizer.display_current_results(visuals, epoch, current_step)

        if opt.test:
            break

    trainer.update_learning_rate(epoch)

    model_path = os.path.join(ckpt_dir, 'model_%d.pth' % epoch)
    trainer.save(model_path, epoch)

print('Training complete.')
