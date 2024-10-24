import warnings
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import logging
from utils.my_data_parallel import MyDataParallel
import network
from optimizer import restore_snapshot
import argparse
from datasets import teeth_mask
from config import assert_and_infer_cfg
from torch.backends import cudnn

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--snapshot', required=True, type=str, default='')
parser.add_argument('--arch', type=str, default='', required=True)
parser.add_argument('--dataset_cls', type=str, default='cityscapes', help='cityscapes')
args = parser.parse_args()

args.dataset_cls = teeth_mask
assert_and_infer_cfg(args, train_mode=False)
args.apex = False  # No support for apex eval
cudnn.benchmark = False
img_name_list = ["2018_000110", "2018_000149", "2018_000190", "2018_000192", "2018_000243", "2018_000405"]


def get_net():
    """
    Get Network for evaluation
    """
    logging.info('Load model file: %s', args.snapshot)
    net = network.get_net(args, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    net, _ = restore_snapshot(net, optimizer=None,
                              snapshot=args.snapshot, restore_optimizer_bool=False)
    net.eval()
    return net


for img_name in img_name_list:
    img_path = "./tmp/grad_cam/pick/raw/" + img_name + ".png"
    mask_path = "./tmp/grad_cam/pick/mask/" + img_name + ".png"
    image = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))
    rgb_img = np.float32(image) / 255
    rgb_mask = np.float32(mask) / 255
    tensor_img = preprocess_image(rgb_img,
                                    mean=[0.667, 0.602, 0.530],
                                    std=[0.168, 0.197, 0.189])
    tensor_mask = preprocess_image(rgb_mask,
                                    mean=[0.673],
                                    std=[0.461])
    input_tensor = torch.cat((tensor_img, tensor_mask), dim=0).unsqueeze(0)

    # Taken from the torchvision tutorial
    # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html

    model = get_net()

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()


    class SegmentationModelOutputWrapper(torch.nn.Module):
        def __init__(self, model):
            super(SegmentationModelOutputWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)


    model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor)

    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = [
        'background', 'teeth', 'plaque'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    plaque_category = sem_class_to_idx["plaque"]
    plaque_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    plaque_mask_uint8 = 255 * np.uint8(plaque_mask == plaque_category)
    plaque_mask_float = np.float32(plaque_mask == plaque_category)

    both_images = np.hstack((image, np.repeat(plaque_mask_uint8[:, :, None], 3, axis=-1)))
    Image.fromarray(both_images)


    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()

        def __call__(self, model_output):
            return (model_output[self.category, :, :] * self.mask).sum()


    # target_layers = [model.model.backbone.layer4]
    target_layers = [model.model.module.final_seg]
    targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]
    with GradCAM(model=model,
                 target_layers=target_layers,
                 use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    img = Image.fromarray(cam_image)
    img.save("./tmp/grad_cam/final/" + img_name + ".png")
