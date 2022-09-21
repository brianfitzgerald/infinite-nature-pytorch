import torch
import torchvision
import numpy as np

@torch.no_grad()
def midas_disparity(rgb):
    """Computes MiDaS v2 disparity on an RGB input image.

    Args:
    rgb: [H, W, 3] Range [0.0, 1.0].
    Returns:
    [H, W, 1] MiDaS disparity resized to the input size and in the range
    [0.0, 1.0]
    """

    #   TODO import model
    midas_model = None
    size = rgb.shape[:2]
    resized = torchvision.transforms.Resize(rgb, [384, 384], interpolation="bilinear")
    # MiDaS networks wants [1, C, H, W]
    midas_input = np.transpose(resized, [2, 0, 1])[np.newaxis]
    prediction = midas_model.signatures['serving_default'](midas_input)['default'][0]
    disp_min = torch.min(prediction)
    disp_max = torch.min(prediction)
    prediction = (prediction - disp_min) / (disp_max - disp_min)
    return torchvision.transforms.Resize(prediction[..., np.newaxis], [384, 384], interpolation="bilinear")

