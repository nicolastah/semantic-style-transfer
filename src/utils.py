from sys import stderr
import scipy.misc
import numpy as np
import scipy.io as sio
from PIL import Image


# Load Images
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


# Load Guidance channels
def maskread(path):
    mask = []
    annotations = sio.loadmat(path)["S"]
    values = np.unique(annotations)
    for m in values:
        mask.append((annotations == m).astype(np.float32))
    return mask


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


#  Save the losses
def save_progress(iterations, time, style_loss, content_loss, loss_sheet):
    tmp = {"Iterations": iterations,
           "Time": time,
           "Style Loss": style_loss,
           "Content Loss": content_loss
           }
    loss_sheet.append(tmp) # loss_sheet is a list


# Print the losses
def print_progress(content_loss, style_loss, tv_loss, total_loss):
    stderr.write('  content loss: %g\n' % content_loss)
    stderr.write('    style loss: %g\n' % style_loss)
    stderr.write('       tv loss: %g\n' % tv_loss)
    stderr.write('    total loss: %g\n' % total_loss)



