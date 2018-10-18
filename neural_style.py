# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

# os function
import os
from argparse import ArgumentParser

import sys
sys.path.insert(0, "./src")

# Algebra/math
import scipy.misc

# Image processing
import utils

# Style transfer
from stylize import stylize

# Default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e3
LEARNING_RATE = 1e1
ITERATIONS = 500
VGG_PATH = './model/imagenet-vgg-verydeep-19.mat'

# To run only on a specific gpu
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--semantic-transfer', type=bool,
                        dest='semantic_transfer', help='if true perform semantic style transfer, else perform full style transfer',
                        metavar='SEM_TRANSFER', default=False)

    parser.add_argument('--content',
                        dest='content', help='content image',
                        metavar='CONTENT', required=True)
    parser.add_argument('--initial',
                        dest='initial', help='initial image to begin optimization, if none use gaussian noise',
                        metavar='INITIAL')

    parser.add_argument('--style',
                        dest='style', help='one style image for full style transfer',
                        metavar='STYLE')

    parser.add_argument('--semantic-styles',
                        dest='semantic_styles',
                        nargs='+', help='two or more style images for the semantic style transfer',
                        metavar='SEM_STYLE')
    parser.add_argument('--masks',
                        dest='masks', help='segmentation masks of the content image',
                        metavar='MASK')

    parser.add_argument('--gradient-capping', type=bool,
                        dest='gradient_capping', help='True: use gradient capping, False: use only guided Gram Matrices',
                        metavar='GRADIENT_CAPPING', default=False)
    parser.add_argument('--capped-objs',
                        dest='capped_objs', type=int,
                        nargs='+', help='To use gradient capping on object i use 1, 0. otherwise',
                        metavar='CAPPED_OBJS')

    parser.add_argument('--output',
                        dest='output', help='output path',
                        metavar='OUTPUT', required=True)

    parser.add_argument('--auto-tuning', type=bool,
                        dest='auto_tuning', help='To use auto-tuning or not',
                        metavar='AUTO_TUNING', default=True)
    parser.add_argument('--erosion', type=bool,
                        dest='erosion', help='To erode the segmentation mask or not',
                        metavar='EROSION', default=False)

    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='iterations (default %(default)s)',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='statistics printing frequency',
                        metavar='PRINT_ITERATIONS')

    parser.add_argument('--checkpoint-output',
                        dest='checkpoint_output', help='checkpoint output format, e.g. output%s.jpg',
                        metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS')

    parser.add_argument('--network',
                        dest='network', help='path to network parameters (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate', help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--preserve-colors', action='store_true',
                        dest='preserve_colors',
                        help='style-only transfer (preserving colors) - if color transfer is not needed')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    # Check VGG path
    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    # Load Content and styles images
    #  And resize the style to the content size
    content = utils.imread(options.content)
    if options.semantic_transfer:
        masks = utils.maskread(options.masks)
        sem_styles = [utils.imread(s) for s in options.semantic_styles]
        for idx, img in enumerate(sem_styles):
            sem_styles[idx] = scipy.misc.imresize(img, content.shape)
        style = None
    else:
        style = utils.imread(options.style)
        style = scipy.misc.imresize(style, content.shape)
        masks = None
        sem_styles = None

    # Image initialisation: noise or content image?
    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(utils.imread(initial), content.shape,'bilinear')
    else:
        initial = None

    # Checkpoint format
    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    # Begin optimization
    for iteration, image in stylize(
        network=options.network,
        semantic_transfer=options.semantic_transfer,
        initial=initial,
        content=content,
        style=style,
        mask=masks,
        sem_style_images=sem_styles,
        gradient_capping=options.gradient_capping,
        capped_objs=options.capped_objs,
        auto_tuning=options.auto_tuning,
        erosion=options.erosion,
        preserve_colors=options.preserve_colors,
        iterations=options.iterations,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        tv_weight=options.tv_weight,
        learning_rate=options.learning_rate,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations
    ):
        output_file = None
        combined_rgb = image

        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output
        if output_file:
            utils.imsave(output_file, combined_rgb)


if __name__ == '__main__':
    main()
