"""
adapted from SPADE
"""
import torch
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.seg_inpaint_model import SegInpaintModel

class SegInpaintTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.seg_inpaint_model = SegInpaintModel(opt)
        if len(opt.gpu_ids) > 0:
            self.seg_inpaint_model = DataParallelWithCallback(self.seg_inpaint_model,
                                                              device_ids=opt.gpu_ids)
            self.seg_inpaint_model_on_one_gpu = self.seg_inpaint_model.module
        else:
            self.seg_inpaint_model_on_one_gpu = self.seg_inpaint_model

        self.generated = None
        
        self.optimizer_SPNet, self.optimizer_SGNet, self.optimizer_D_seg, self.optimizer_D_img = \
            self.seg_inpaint_model_on_one_gpu.create_optimizers(opt)

        self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        # first update SPNet
        self.optimizer_SPNet.zero_grad()
        spn_losses, generated_seg = self.seg_inpaint_model(data, mode='spn')
        spn_loss = sum(spn_losses.values()).mean()
        spn_loss.backward()
        self.optimizer_SPNet.step()
        
        self.spn_losses = spn_losses
        self.generated_seg = generated_seg

        # then udpate SGNet
        self.optimizer_SGNet.zero_grad()
        sgn_losses, generated_img = self.seg_inpaint_model(data, mode='sgn', fake_seg=generated_seg)
        sgn_loss = sum(sgn_losses.values()).mean()
        sgn_loss.backward()
        self.optimizer_SGNet.step()

        self.sgn_losses = sgn_losses
        self.generated_img = generated_img

    def run_discriminator_one_step(self, data):
        # first D_seg
        self.optimizer_D_seg.zero_grad()
        d_seg_losses = self.seg_inpaint_model(data, mode='d_seg')
        d_seg_loss = sum(d_seg_losses.values()).mean()
        d_seg_loss.backward()
        self.optimizer_D_seg.step()
        self.d_seg_losses = d_seg_losses
        
        self.optimizer_D_img.zero_grad()
        #d_img_losses = self.seg_inpaint_model(data, mode='d_img', fake_seg=self.generated_seg)
        d_img_losses, real_img, corruped_img = self.seg_inpaint_model(data, mode='d_img', fake_seg=self.generated_seg)
        d_img_loss = sum(d_img_losses.values()).mean()
        d_img_loss.backward()
        self.optimizer_D_img.step()
        self.d_img_losses = d_img_losses

        # NOTE: for display current results
        self.real_img = real_img
        self.corruped_img = corruped_img

    def get_latest_results(self):
        return self.real_img, self.corruped_img, self.generated_seg, self.generated_img

    def get_loss_str(self):
        def gather_str(name, errors):
            msg = '%s: ' % name
            for k, v in errors.items():
                v = v.mean().float()
                msg += '%s: %.3f ' % (k, v)
            msg += '| '
            return msg
        spn_l, sgn_l, d_seg_l, d_img_l = self.spn_losses, self.sgn_losses, self.d_seg_losses, self.d_img_losses
        
        return gather_str('SPN', spn_l) + gather_str('SGN', sgn_l) + \
               gather_str('DSeg', d_seg_l) + gather_str('DImg', d_img_l) 

    def save(self, path, epoch):
        self.seg_inpaint_model_on_one_gpu.save(path, epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.seg_inpaint_model_on_one_gpu.optimizer_SPNet.param_groups:
                param_group['lr'] = new_lr_G
            for param_group in self.seg_inpaint_model_on_one_gpu.optimizer_SGNet.param_groups:
                param_group['lr'] = new_lr_G
            for param_group in self.seg_inpaint_model_on_one_gpu.optimizer_D_seg.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.seg_inpaint_model_on_one_gpu.optimizer_D_img.param_groups:
                param_group['lr'] = new_lr_D
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
