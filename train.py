import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import os
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter

from models import *
from data import Provider, SRBenchmark
from utils import PSNR, _rgb2ycbcr, seed_everything


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser("Training Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int,  default=8)
    parser.add_argument("--train-dir", type=str, default='./data/train/DIV2K',
                        help="Training images")
    parser.add_argument("--val-dir", type=str, default='./data/test/',
                        help="Validation images")
    parser.add_argument("--i-display", type=int, default=500,
                        help="display info every N iteration")
    parser.add_argument("--i-validate", type=int, default=500,
                        help="validation every N iteration")
    parser.add_argument("--i-save", type=int, default=2000,
                        help="save checkpoints every N iteration")

    parser.add_argument("--upscale", nargs='+', type=int, default=[2, 2],
                        help="upscaling factors")
    parser.add_argument("--crop-size", type=int, default=48,
                        help="input LR training patch size")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="training batch size")
    parser.add_argument("--start-iter", type=int, default=0,
                        help="Set 0 for from scratch, else will load saved params and trains further")
    parser.add_argument("--train-iter", type=int, default=200000,
                        help="number of training iterations")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--wd', type=float, default=0,  help='weight decay')

    parser.add_argument('--msb', type=str, default='hdb', choices=['hdb', 'hd'])
    parser.add_argument('--lsb', type=str, default='hd', choices=['hdb', 'hd'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb:{}-lsb:{}-act:{}-nf:{}-{}".format(args.msb, args.lsb, args.act_fn, args.n_filters, factors)

    act_fn_dict = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': nn.LeakyReLU, 'starrelu': StarReLU}
    args.act_fn = act_fn_dict[args.act_fn]

    return args


def SaveCheckpoint(models, opt_G, i, args, best=False):
    if best:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), 'checkpoint/{}/model_G_S{}_best.pth'.format(args.exp_name, stage))
            else:
                torch.save(model.state_dict(), 'checkpoint/{}/model_G_S{}_best.pth'.format(args.exp_name, stage))
        torch.save(opt_G.state_dict(), 'checkpoint/{}/opt_G_best.pth'.format(args.exp_name))
        print("Best checkpoint saved")
    else:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), 'checkpoint/{}/model_G_S{}_i{:06d}.pth'.format(args.exp_name, stage, i))
            else:
                torch.save(model.state_dict(), 'checkpoint/{}/model_G_S{}_i{:06d}.pth'.format(args.exp_name, stage, i))
        torch.save(opt_G.state_dict(), 'checkpoint/{}/opt_G_i{:06d}.pth'.format(args.exp_name, i))
        print("Checkpoint saved {}".format(str(i)))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    seed_everything(args.seed)

    ### Tensorboard for monitoring ###
    writer = SummaryWriter(log_dir='./log/{}'.format(args.exp_name))

    models = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)
    
    for s in args.upscale:
        models.append(HKNet(msb=args.msb, lsb=args.lsb, nf=args.n_filters, upscale=s, act=args.act_fn).to(device))


    ## Optimizers
    opt_G = optim.Adam([{'params': list(filter(lambda p: p.requires_grad, model.parameters()))} for model in models], 
                       lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd, eps=1e-8, amsgrad=False)

    scheduler = optim.lr_scheduler.MultiStepLR(opt_G, milestones=[100000, 150000], gamma=0.1)

    ## Load saved params
    if args.start_iter > 0:
        for stage in range(n_stages):
            lm = torch.load('checkpoint/{}/model_G_S{}_i{:06d}.pth'.format(args.exp_name, stage, args.start_iter))
            models[0].load_state_dict(lm, strict=True)

        lm = torch.load('checkpoint/{}/opt_G_i{:06d}.pth'.format(args.exp_name, args.start_iter))
        opt_G.load_state_dict(lm)
    
    if torch.cuda.device_count() > 1:
        models = [nn.DataParallel(model) for model in models]
    
    # Training dataset
    train_loader = Provider(args.batch_size, args.n_workers, sr_scale, args.train_dir, args.crop_size)

    # Validation dataset
    valid_loader = SRBenchmark(args.val_dir, scale=sr_scale)
    valid_datasets = ['Set5']

    ## Prepare directories
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/{}'.format(args.exp_name)):
        os.mkdir('checkpoint/{}'.format(args.exp_name))
    if not os.path.isdir('log'):
        os.mkdir('log')

    l_accum = [0.,0.,0.]
    dT = 0.
    rT = 0.
    accum_samples = 0


    ### TRAINING
    best_psnr = 0.0
    for i in tqdm(range(args.start_iter+1, args.train_iter+1)):

        for model in models:
            model.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H = train_loader.next()
        batch_H = batch_H.to(device)      # BxCxHxW (32, 3, 192, 192), range [0,1]
        batch_L = batch_L.to(device)      # BxCxHxW (32, 3, 48, 48), range [0,1]
    
        dT += time.time() - st


        ## TRAIN G
        st = time.time()
        opt_G.zero_grad()

        x = batch_L
        for model in models:
            x = model(x)
        pred = torch.clamp(x, 0, 1)  # [-2, 2] -> [0, 1]
        loss_G = F.mse_loss(pred, batch_H)

        # Update
        loss_G.backward()
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += args.batch_size
        l_accum[0] += loss_G.item()


        ## Show information
        if i % args.i_display == 0:
            writer.add_scalar('loss_Pixel', l_accum[0]/args.i_display, i)
            print("{}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                args.exp_name, i, accum_samples, l_accum[0]/args.i_display, dT/args.i_display, rT/args.i_display))
            l_accum = [0.,0.,0.]
            dT = 0.
            rT = 0.


        ## Save models
        if i % args.i_save == 0:
            SaveCheckpoint(models, opt_G, i, args)


        ## Validation
        if i % args.i_validate == 0:
            with torch.no_grad():
                for model in models:
                    model.eval()


                for j in range(len(valid_datasets)):
                    psnrs = []
                    files = valid_loader.files[valid_datasets[j]]

                    for k in range(len(files)):
                        key = valid_datasets[j] + '_' + files[k][:-4]

                        img_gt = valid_loader.ims[key] # (512, 512, 3) range [0, 255]
                        input_im = valid_loader.ims[key + 'x%d' % sr_scale] # (128, 128, 3) range [0, 255]

                        input_im = input_im.astype(np.float32) / 255.0  
                        val_L = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).to(device) # (1, 3, 128, 128)

                        x = val_L
                        for model in models:
                            x = model(x)


                        # Output 
                        image_out = (x).cpu().data.numpy() # (1, 3, 512, 512)
                        image_out = np.transpose(np.clip(image_out[0], 0. , 1.), [1,2,0]) # BxCxHxW -> HxWxC
                        image_out = ((image_out)*255).astype(np.uint8)

                        # PSNR on Y channel
                        psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], sr_scale))

                    mean_psnr = np.mean(np.asarray(psnrs))

                    # save best psnr for set5
                    if mean_psnr > best_psnr:
                        best_psnr = np.mean(np.asarray(psnrs))
                        SaveCheckpoint(models, opt_G, i, args, best=True)
                    
                    print('Iter {} | Dataset {} | AVG Val PSNR: {:02f}'.format(i, valid_datasets[j], mean_psnr))
                    writer.add_scalar('PSNR_valid/{}'.format(valid_datasets[j]), mean_psnr, i)
                    writer.flush()

    print(f'Best PSNR: {best_psnr}')