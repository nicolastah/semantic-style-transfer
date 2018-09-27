import tensorflow as tf
import numpy as np
import skimage.morphology


def down_sample_guidance_channels(mask, auto_tuning, erosion, net, STYLE_LAYERS, guidance_maps, ratio):
    """
    Down sample the guidance channels for each considered layers

    Inputs:
     - mask: list containing segmentation masks of each regions
     - auto_tuning: bool
     - erosion: bool
     - net: VGG (needed to get dimension of the feature maps)
     - STYLE_LAYERS: tuple
     - guidance_maps: empty dictionnary (see output)
     - ratio: empty list (see output)

    Output (no return type, because list and dictionnary are passed as reference):
     - ratio: pixel ratio of each region in the content image
     - guidance_maps: guidance channel of each region for each considered layers
    """


    mask_placeholder = tf.placeholder(tf.float32, shape=(1, None, None, None))
    mask_tensor = tf.nn.avg_pool(mask_placeholder,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME")

    for idx in range(len(mask)):

        mask_obj = mask[idx]  # Current mask
        mask_layers = {}  # Current mask for all layers

        if auto_tuning:
            ratio.append(np.count_nonzero(mask_obj) / float(mask_obj.size))
        else:
            ratio.append(1.0)

        mask_obj = np.expand_dims(mask_obj, axis=0)  # Batch
        mask_obj = np.expand_dims(mask_obj, axis=3)  # Number of maps

        # Loop over the layers used to define Lstyle
        for style_layer in STYLE_LAYERS:

            # Normalisation from Gatys
            mask_copy = np.copy(mask_obj[0, :, :, 0])
            if np.count_nonzero(mask_obj[0, :, :, 0]) > 0:
                mask_copy /= (np.count_nonzero(mask_obj[0, :, :, 0]))

            mask_copy = np.expand_dims(mask_copy, axis=0)
            mask_copy = np.expand_dims(mask_copy, axis=3)

            channels = net[style_layer].shape[3]
            width = mask_obj.shape[1]
            height = mask_obj.shape[2]
            mask_shape = (1, width, height, int(channels))
            mask_layers[style_layer] = np.broadcast_to(mask_copy, mask_shape)

            # There is pooling between each layer we chose
            mask_obj = mask_tensor.eval(feed_dict={mask_placeholder: mask_obj})

            # Erosion/dilation
            if erosion:
                mask_obj[0, :, :, 0] = skimage.morphology.erosion(mask_obj[0, :, :, 0])

        guidance_maps[idx] = mask_layers


def compute_style_layers_weight(weight_scheme, STYLE_LAYERS, STYLE_LAYER_WEIGHT_EXP):
    """
    Different style layers have different weights.

    Inputs:
     - weight scheme: 1: Anyshatalie, 2: Geometric Scheme
     - STYLE_LAYERS: layers used to represent the style
     - STYLE_LAYER_WEIGHTS_EXP: by how much the weights are multiplied between successive layers. (use when weight scheme = 1)

    Output:
     - style_layers_weights: weights associated to each layer
    """

    style_layers_weights = {}

    if weight_scheme == 1:
        layer_weight = 1.0
        for style_layer in STYLE_LAYERS:
            style_layers_weights[style_layer] = layer_weight
            layer_weight *= STYLE_LAYER_WEIGHT_EXP
    else:
        exp = len(STYLE_LAYERS)
        layer_weight = 2.0
        for style_layer in STYLE_LAYERS:
            style_layers_weights[style_layer] = layer_weight ** exp
            exp -= 1

    # Normalize style layers weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    return style_layers_weights


def regularization_loss(image, tv_weight):
    tv_y_size = _tensor_size(image[:, 1:, :, :])
    tv_x_size = _tensor_size(image[:, :, 1:, :])
    tv_loss = tv_weight * 2 * (
        (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :tf.shape(image)[1] - 1, :, :]) /
         tv_y_size) +
        (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :tf.shape(image)[2] - 1, :]) /
         tv_x_size))
    return tv_loss


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)



