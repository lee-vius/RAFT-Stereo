import sys
from torch._C import device

from tqdm.cli import main
sys.path.append('core')

import numpy as np
import cv2 as cv
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
print("Using hardware: {DEVICE}")

def load_image(imfile):
    img = np.array(cv.imread(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

class raftArgs():
    def __init__(self) -> None:
        self.hidden_dims = [128]*3
        self.corr_implementation = 'reg'
        self.corr_levels = 4
        self.corr_radius = 4
        self.shared_backbone = False
        self.n_downsample = 2
        self.n_gru_layers = 3
        self.slow_fast_gru = False

        self.mixed_precision = False
        self.valid_iters = 32
        self.restore_ckpt = './models/raftstereo-eth3d.pth'
        

if __name__ == '__main__':
    args = raftArgs()
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids = [0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        leftfile = './datasets/MyData/meditest1/left/img0.jpg'
        rightfile = './datasets/MyData/meditest1/right/img0.jpg'

        left = load_image(leftfile)
        right = load_image(rightfile)

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        _, flow_up = model(left, right, iters=args.valid_iters, test_mode=True)
        disp = flow_up.cpu().numpy().squeeze()
        plt.imshow(-disp, cmap='jet')
        plt.show()