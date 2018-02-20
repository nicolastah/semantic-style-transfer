# semantic-style-transfer
 :art: :art:
**Tensorflow** implementation of **semantic style transfer** based on **guided Gram Matrices**.
Say that it uses the slow model (imqge optimisation problem)

In construction...

## Contents
1. [Examples](#examples)
2. [Implementation Details](#implementation-details)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Acknowledgements](#acknowledgements)
5. [Citation](#citation)
5. [License](#license)

## Examples

## Implementation Details

## Installation

Let's get started! **Installation** will be pretty **easy**. First, **clone a copy** of this repository:

```
git clone https://github.com/nicolastah/semantic-style-transfer
```

The next step will be to **create a new project** using your favorite python IDE. If you have never used Python before, I would recommend [Pycharm](https://www.jetbrains.com/pycharm/). Its integrated debugger is just awesome! Last step is to install all the depedencies and you will be ready to go  :stuck_out_tongue:!

### Dependencies
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [TensorFlow](https://www.tensorflow.org)
- [SciPy](https://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)
- [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (**GPU**) -- Recommended
- [CUDNN](https://developer.nvidia.com/cudnn) (**GPU**) -- Recommended

I recommend creating **isolated Python environments** using [Virtualenv](https://virtualenv.pypa.io/en/stable/). **Vital** to **avoid dependencies conflicts** when working on different projects. For those who are not familiar with Virtualenv, here are 2 tutorials that should help you get started, [tuto1](http://thepythonguru.com/python-virtualenv-guide/) and [tuto2](http://www.simononsoftware.com/virtualenv-tutorial-part-2/). Lastly, I' m using [Ubuntu16.04](https://www.ubuntu.com/download/desktop), but the code should run on Windows and macOs.

**Remark:** make sure that your **project interpreter** is **properly configured**. In Pycharm you can check which interpreter is used by clicking: ```File -> Settings -> Project```. If you have created a virtual environment and want to use it as the project interpreter click on the small gear, ```Add Local``` and locate your virtual environment.

### Model Weigths
Recall that  **style transfer** is based on **perceptual losses**. Thoses losses, based on 
**high level features representation**, allows to **separate style and content**. As in the orignal work [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), we have used a [VGG19](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-very) pretrained on image classification. Download and copy it to the folder X.

### Speed
Talk about the speed with GPU. Do it with different mage size (similar as Titus). Then do the same without GPU. Say that it is also possbile but it is slower
Can use emojie here, turtle = slow (cpu), rabbit = fast (gpu)

## Usage
### Full Transfer
Show how to run the code
Say how to pgive param script using pycharm
Put link to titu Kera implementation, say that can check his work for the tips

### Semantic Transfer
Show how to run the code
Explain guidance guided matrices = semantic mask
Can put some links to some semantic segmentation algorithms
Explain how to put the mask in my programm

## Acknowledgements
- Guided Gram Matrices is based on Gatys' paper [Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/abs/1611.07865)
- Our implementation is based on [anishathalye/neural-style](https://github.com/anishathalye/neural-style)
- Our work is an implement of XX

## Citation
```
@misc{nchung2018_semantic_style,
  author = {Chung Nicolas},
  title = {Semantic Style Transfer},
  year = {2018},
  howpublished = {\url{https://github.com/nicolastah/semantic-style-transfer}},
  note = {commit xxxxxxx}
}
```

## License
Copyright (c) 2018 Chung Nicolas. Released under GPLv3. See [LICENSE.txt](./LICENSE) for details.
