from collections import namedtuple
from cv2.ximgproc import guidedFilter
from net import *
from net.losses import StdLoss
from net.vae_model import VAE
from utils.imresize import np_imresize
from utils.image_io import *
from utils.dcp import get_atmosphere
from skimage.measure import compare_psnr, compare_ssim
import torch
import torch.nn as nn
import numpy as np

DehazeResult_psnr = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr'])
DehazeResult_ssim = namedtuple("DehazeResult", ['learned', 't', 'a', 'ssim'])


class Dehaze(object):
    def __init__(self, image_name, image, gt_img, num_iter=500, clip=True, output_path="output/"):
        self.image_name = image_name
        self.image = image
        self.gt_img = gt_img
        self.num_iter = num_iter
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None
        self.output_path = output_path

        self.clip = clip
        self.blur_loss = None
        self.best_result_psnr = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 3
        self.post = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        self.images_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.image_net = image_net.type(data_type)

        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(data_type)

    def _init_ambient(self):
        ambient_net = VAE(self.gt_img.shape)
        self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)

        atmosphere = get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                        requires_grad=False)
        self.at_back = atmosphere

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()]
        parameters += [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _init_inputs(self):
        self.image_net_inputs = np_to_torch(self.image).cuda()
        self.mask_net_inputs = np_to_torch(self.image).cuda()
        self.ambient_net_input = np_to_torch(self.image).cuda()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure()
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self):
        self.image_out = self.image_net(self.image_net_inputs)
        self.ambient_out = self.ambient_net(self.ambient_net_input)

        self.mask_out = self.mask_net(self.mask_net_inputs)

        self.blur_out = self.blur_loss(self.mask_out)
        self.mseloss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                     self.images_torch)

        vae_loss = self.ambient_net.getLoss()
        self.total_loss = self.mseloss + vae_loss
        self.total_loss += 0.005 * self.blur_out

        dcp_prior = torch.min(self.image_out.permute(0, 2, 3, 1), 3)[0]
        self.dcp_loss = self.mse_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.05
        self.total_loss += self.dcp_loss

        self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
        self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))

        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):
        if step % 5 == 0:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
            mask_out_np = self.t_matting(mask_out_np)

            post = np.clip((self.image - ((1 - mask_out_np) * ambient_out_np)) / mask_out_np, 0, 1)

            psnr = compare_psnr(self.gt_img, post)
            ssims = compare_ssim(self.gt_img.transpose(1, 2, 0), post.transpose(1, 2, 0), multichannel=True)

            self.current_result = DehazeResult_psnr(learned=image_out_np, t=mask_out_np, a=ambient_out_np, psnr=psnr)
            self.temp = DehazeResult_ssim(learned=image_out_np, t=mask_out_np, a=ambient_out_np, ssim=ssims)

            if self.best_result_psnr is None or self.best_result_psnr.psnr < self.current_result.psnr:
                self.best_result_psnr = self.current_result

            if self.best_result_ssim is None or self.best_result_ssim.ssim < self.temp.ssim:
                self.best_result_ssim = self.temp

    def _plot_closure(self, step):
        """
         :param step: the number of the iteration

         :return:
         """
        print('Iteration %05d    Loss %f  %f cur_ssim %f max_ssim: %f cur_psnr %f max_psnr %f\n' % (
                                                                            step, self.total_loss.item(),
                                                                            self.blur_out.item(),
                                                                            self.temp.ssim,
                                                                            self.best_result_ssim.ssim,
                                                                            self.current_result.psnr,
                                                                            self.best_result_psnr.psnr), '\r', end='')

    def finalize(self):
        self.final_image = np_imresize(self.best_result_psnr.learned, output_shape=self.original_image.shape[1:])
        self.final_t_map = np_imresize(self.best_result_psnr.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.best_result_psnr.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.final_t_map
        self.post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        save_image(self.image_name + "_psnr", self.post, self.output_path)

        self.final_t_map = np_imresize(self.best_result_ssim.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.best_result_ssim.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.final_t_map
        post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)

        save_image(self.image_name + "_ssim", post, self.output_path)

        self.final_t_map = np_imresize(self.current_result.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.current_result.a, output_shape=self.original_image.shape[1:])
        mask_out_np = self.t_matting(self.final_t_map)
        post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)

        save_image(self.image_name + "_run_final", post, self.output_path)

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehaze(image_name, image, gt_img, num_iter=500, output_path="output/"):
    dh = Dehaze(image_name, image, gt_img, num_iter, clip=True, output_path=output_path)

    dh.optimize()
    dh.finalize()

    save_image(image_name + "_original", np.clip(image, 0, 1), dh.output_path)

    psnr = dh.best_result_psnr.psnr
    ssim = dh.best_result_ssim.ssim
    return psnr, ssim


if __name__ == "__main__":
    torch.cuda.set_device(0)

    hazy_add = 'data/hazy.png'
    gt_add = 'data/gt.png'
    name = "1400_3"
    print(name)

    hazy_img = prepare_hazy_image(hazy_add)
    gt_img = prepare_gt_img(gt_add, SOTS=True)

    psnr, ssim = dehaze(name, hazy_img, gt_img, output_path="output/")
    print(psnr, ssim)
