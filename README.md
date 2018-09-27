# semantic-style-transfer
:art: :art:
**Tensorflow** implementation of **semantic style transfer** based on **guided Gram Matrices**. Our model is based on the image optimisation problem defined in [Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/abs/1611.07865). We proposed 2 simple solutions to improve the quality of the generated images:
- Gradient capping
- Auto-tuning

## Contents
1. [Examples](#examples)
2. [Implementation Details](#implementation-details)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Acknowledgements](#acknowledgements)
5. [Citation](#citation)
5. [License](#license)

## Examples
#### Full style transfer
<div align='center'>
 <img src='content/cat.jpg' height="209px">
 <img src='style/rain_princess.jpg' height="209px">
 <img src='output/cat_full.png' height="209px">
</div>

#### Semantic style transfer
<div align='center'>
 <img src='mask/cat.png' height="157.75px">
 <img src='style/wave.jpg' height="157.75px" width="157.17px">
 <img src='style/rain_princess.jpg' height="157.75px" width="157.75px">
 <img src='style/ashville.jpg' height="157.75px" width="157.75px">
</div>

<div align='center'>
 <img src='content/cat.jpg' height="315.5px">
 <img src='output/cat_sem.png' height="315.5px">
</div>

<div align='center'>
 <img src='mask/elephant.png' height="123px" width="123px">
 <img src='style/wave.jpg' height="123px" width="123px">
 <img src='style/night.jpg' height="123px" width="123px">
 <img src='style/machine.jpg' height="123px" width="123px">
 <img src='style/prayer.jpg' height="123px" width="123px">
</div>

<div align='center'>
 <img src='content/elephant.jpg' height="315.5px">
 <img src='output/elephant.png' height="315.5px">
</div>

## Implementation Details

## Installation

Let's get started! **Installation** will be pretty **easy**. First, **clone a copy** of this repository:

```
git clone https://github.com/nicolastah/semantic-style-transfer
```

The next step will be to **create a new project** using your favorite python IDE. I'm personnally using [Pycharm](https://www.jetbrains.com/pycharm/). I find its integrated debugger just awesome! Last step is to install all the depedencies and you will be ready to go  :stuck_out_tongue:!

### Dependencies
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [TensorFlow](https://www.tensorflow.org)
- [SciPy](https://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org)
- [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (**GPU**) -- Recommended
- [CUDNN](https://developer.nvidia.com/cudnn) (**GPU**) -- Recommended

I recommend creating **isolated Python environments** using [Virtualenv](https://virtualenv.pypa.io/en/stable/). **Vital** to **avoid dependencies conflicts**  :dizzy_face: when working on different projects. For those who are not familiar with Virtualenv, here are 2 tutorials that should help you get started, [tuto1](http://thepythonguru.com/python-virtualenv-guide/) and [tuto2](http://www.simononsoftware.com/virtualenv-tutorial-part-2/). The code was tested on [Ubuntu16.04](https://www.ubuntu.com/download/desktop).

### Model Weigths: VGG19
- **Download** the weights [here]( http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat).
- Recall that **style transfer** is based on **perceptual losses**. Those losses allow to **separate style and content**. They are computed from **high level features**. As in the orignal work [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), we extract those features using a **VGG19** pretrained on image classification.  

## Usage
Check the wiki for code usage.

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
