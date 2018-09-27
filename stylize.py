# Copyright (c) 2015-2018 Chung Nicolas. Released under GPLv3.

# Deep learning framework
import sys
sys.path.insert(0, "./src")
import tensorflow as tf
import numpy as np
import vgg
import colors
import ops
import utils

from sys import stderr
import time

# If need to record loss in CVS
import pyexcel
SAVE_ITERATIONS = None

# Layers used for features representation
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

CONTENT_WEIGHT_BLEND = 1                # content weight blend, conv4_2 * blend + conv5_2 * (1-blend)
INITIAL_NOISEBLEND = 0.0

# Style weight scheme
# 1 = Anishatyle: higher layer higher weight
# 2 = Geometric scheme: lower layer, higher weight
weight_scheme = 2
STYLE_LAYER_WEIGHT_EXP = 1

# Adam optimization
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
POOLING = 'max'                         # max or avg

try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, semantic_transfer,
            initial, content, style,
            mask, sem_style_images, gradient_capping, capped_objs, auto_tuning, erosion,
            preserve_colors, iterations,
            content_weight, style_weight, tv_weight,
            learning_rate,
            print_iterations=None, checkpoint_iterations=None):

    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """

    t = time.time()

    # Load network
    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    # Dictionaries = features maps for each considered layers
    content_features = {}
    if semantic_transfer:
        style_semantic_features = [{} for _ in sem_style_images]
        guidance_maps = [{} for _ in mask]
        ratio = []                                                                # Auto tuning
        net_gradient = []                                                         # For Gradient Capping
    else:
        style_features = {}

    # Batch
    shape = (1,) + content.shape

    # To vizualize the loss curves
    if SAVE_ITERATIONS:
        loss_sheet = []

    style_layers_weights = ops.compute_style_layers_weight(weight_scheme, STYLE_LAYERS, STYLE_LAYER_WEIGHT_EXP)

    # Content features of content image
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, POOLING)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])

        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # Style features of style images
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, POOLING)

        # Guided Gram Matrices (Semantic style transfer)
        if semantic_transfer:
            # Downsample guidance channels
            ops.down_sample_guidance_channels(mask, auto_tuning, erosion, net, STYLE_LAYERS, guidance_maps, ratio)
            for idx, img in enumerate(sem_style_images):
                style_pre = np.array([vgg.preprocess(img, vgg_mean_pixel)])

                for layer in STYLE_LAYERS:
                    features = net[layer].eval(feed_dict={image: style_pre})
                    features = features * guidance_maps[idx][layer]
                    features = np.reshape(features, (-1, features.shape[3]))
                    features = features - 1
                    gram = np.matmul(features.T, features)
                    style_semantic_features[idx][layer] = gram

        # Gram Matrices (Whole style transfer)
        else:
            style_pre = np.array([vgg.preprocess(style, vgg_mean_pixel)])

            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                features = features - 1
                gram = np.matmul(features.T, features) / features.size
                style_features[layer] = gram

    # Initial noise
    initial_content_noise_coeff = 1.0 - INITIAL_NOISEBLEND

    # Optimization
    with tf.Graph().as_default():

        # Initialisation
        if initial is None:
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            initial = initial * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (
                      1.0 - initial_content_noise_coeff)

        image = tf.Variable(initial)

        # Content loss
        net = vgg.net_preloaded(vgg_weights, image, POOLING)
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = CONTENT_WEIGHT_BLEND
        content_layers_weights['relu5_2'] = 1.0 - CONTENT_WEIGHT_BLEND
        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight
                                  * (2 * tf.nn.l2_loss(net[content_layer] - content_features[content_layer])
                                  / content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)


        style_loss = 0
        style_losses = []

        # Semantic Style Loss
        if semantic_transfer:

            for i in range(len(sem_style_images)):

                segmented_obj = guidance_maps[i]

                if gradient_capping:
                    if capped_objs[i] == 1:
                        mask_tmp = np.expand_dims(mask[i], axis=0)
                        mask_tmp = np.expand_dims(mask_tmp, axis=3)
                        image_tmp = image * tf.stop_gradient(tf.convert_to_tensor(mask_tmp, dtype=tf.float32))
                        net_gradient.append(vgg.net_preloaded(vgg_weights, image_tmp, POOLING))
                    else:
                        net_gradient.append(net)

                for idx, style_layer in enumerate(STYLE_LAYERS):
                    if gradient_capping:
                        layer = net_gradient[i][style_layer]
                    else:
                        layer = net[style_layer]

                    _, height, width, number = map(lambda i: i.value, layer.get_shape())
                    size = number
                    feats = layer * segmented_obj[style_layer]

                    # Gram of the stylized image
                    feats = tf.reshape(feats, (-1, number))
                    feats = feats - 1
                    gram = tf.matmul(tf.transpose(feats), feats)

                    # Precomputed Gram of the style image
                    style_gram = style_semantic_features[i][style_layer]
                    style_losses.append(style_layers_weights[style_layer] * 2
                                        * tf.nn.l2_loss(gram - style_gram)
                                         / (2 * size**2))

                style_loss += style_weight * reduce(tf.add, style_losses) * ratio[i]

        # Full Style Loss
        else:
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number                                           # Ml * Nl
                feats = tf.reshape(layer, (-1, number))
                feats = feats - 1
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[style_layer]
                style_losses.append(style_layers_weights[style_layer] * 2
                                        * tf.nn.l2_loss(gram - style_gram)
                                        / style_gram.size)
            style_loss += style_weight * reduce(tf.add,style_losses)

        # Regularization Loss
        tv_loss = ops.regularization_loss(image, tv_weight)

        # Total Loss
        loss = content_loss + style_loss + tv_loss

        # Optimizer
        train_step = tf.train.AdamOptimizer(learning_rate, BETA1, BETA2, EPSILON).minimize(loss)

        # best is the image returned after optimization
        best_loss = float('inf')
        best = None

        # Optimization
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')

            # Print the progress for every print_iterations
            if (print_iterations and print_iterations != 0):
                utils.print_progress(content_loss.eval(), style_loss.eval(), tv_loss.eval(), loss.eval())

            # Optimize + print loss + return final image
            for i in range(iterations):
                train_step.run()
                last_step = (i == iterations - 1)

                if print_iterations and i % print_iterations == 0:
                    utils.print_progress(content_loss.eval(), style_loss.eval(), tv_loss.eval(), loss.eval())

                if SAVE_ITERATIONS and i % SAVE_ITERATIONS == 0:
                    utils.save_progress(i, time.time() - t, style_loss.eval(), content_loss.eval(), loss_sheet)

                if last_step:
                    utils.print_progress(content_loss.eval(), style_loss.eval(), tv_loss.eval(), loss.eval())
                    if SAVE_ITERATIONS:
                        utils.save_progress(i, time.time() - t, style_loss.eval(), content_loss.eval(), loss_sheet)
                        pyexcel.save_as(records = loss_sheet, dest_file_name = "loss.csv")

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:

                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    # Color preservation
                    if preserve_colors and preserve_colors == True:
                        img_out = colors.preserve_colors(content, img_out)

                    yield (
                        (None if last_step else i),
                        img_out
                    )



