import numpy as np
from PIL import Image

def preserve_colors(original_image, stylized_image):
    """
    Perform color preservation.

    Inputs:
     - original image: content image
     - stylized_image: image obtained after stylization
     - dtype: uint8

    Output:
     - img_out: image with style of stylized_image but color of original image
    """

    original_image = np.clip(original_image, 0, 255)
    styled_image = np.clip(stylized_image, 0, 255)

    # Luminosity transfer steps:
    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
    # 2. Convert stylized grayscale into YUV (YCbCr)
    # 3. Convert original image into YUV (YCbCr)
    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
    # 5. Convert recombined image from YUV back to RGB

    # 1
    styled_grayscale = rgb2gray(styled_image)
    styled_grayscale_rgb = gray2rgb(styled_grayscale)

    # 2
    styled_grayscale_yuv = np.array(
        Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

    # 3
    original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

    # 4
    w, h, _ = original_image.shape
    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
    combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
    combined_yuv[..., 1] = original_yuv[..., 1]
    combined_yuv[..., 2] = original_yuv[..., 2]

    # 5
    img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
    return img_out

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


