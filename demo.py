import os
from network.network_pro import Inpaint as Pro
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from utils import load_checkpoint
import warnings
import glob, argparse
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

warnings.simplefilter('ignore', UserWarning)

parser = argparse.ArgumentParser(description="Official PyTorch Code for Error-Mask-Adaptive Dynamic Filtring for Image Inpainting, IEEE Access, vol. 13, pp. 18403--18417 2025", 
                                 usage='use "%(prog)s --help" for more information', 
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--ckpt', required=True, help='Path for the pretrained model')
parser.add_argument('--path', default="./example/CelebA", help='''Path for directory of pairs of image and mask.''')
parser.add_argument('--output_path', default="./results", help='Path for saving inpainted images')


args = parser.parse_args()

assert os.path.exists(args.path), "Please check image path"

device = torch.device('cuda')

os.makedirs(args.output_path, exist_ok=True)

proposed = Pro(input_size=256).to(device)
proposed = load_checkpoint(args.ckpt, proposed)[0]
proposed.eval()

lst = sorted(glob.glob(f'{args.path}/*_mask.*'))
PSNR = []
SSIM = []
for step, fn in enumerate(lst):
    img = cv2.imread(fn.replace('_mask', '_img')) / 255.
    img = torch.Tensor(img[None]).permute(0, 3, 1, 2) * 2. - 1.
    h, w = img.shape[:2]
    img = TF.resize(img, [256, 256])
    mask = cv2.imread(fn, 0) / 255.
    mask = torch.Tensor(mask[None, None])
    gt = img.to(device, dtype=torch.float32)
    mask = mask.to(device, dtype=torch.float32)
    with torch.no_grad():
        _, out_pro = proposed(gt, mask)

    out_pro = torch.clip(out_pro * mask + gt * (1 - mask), -1., 1.)
    out_pro = np.uint8((out_pro[0].permute(1, 2, 0).cpu().detach().numpy() + 1) / 2. * 255.)

        
    gt = np.uint8((gt[0].permute(1, 2, 0).cpu().detach().numpy() + 1) / 2. * 255.)
    PSNR.append(psnr(gt, out_pro, data_range=255))
    SSIM.append(ssim(gt, out_pro, data_range=255, channel_axis=-1))
    out_fn = os.path.basename(fn.replace('_mask', '_out'))
    cv2.imwrite(f'{args.output_path}/{out_fn}', out_pro)

print(f"\nResults: \t{np.mean(PSNR):.3f}\t{np.mean(SSIM):.5f}")
