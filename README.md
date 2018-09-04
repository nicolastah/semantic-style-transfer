In construction...

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
 <img src='content/cat.jpg' height="225px">
 <img src='style/rain_princess.jpg' height="225px">
 <img src='output/cat_full.png' height="225px">
</div>

#### Semantic style transfer
- 3 style combination
<div align='center'>
 <img src='mask/cat.png' height="123px">
 <img src='style/wave.jpg' height="123px" width="123px">
 <img src='style/rain_princess.jpg' height="123px" width="123px">
 <img src='style/ashville.jpg' height="123px" width="123px">
</div>

<div align='center'>
 <img src='content/cat.jpg' height="250px">
 <img src='output/cat_sem.png' height="250px">
</div>

 - 4 style combination
<div align='center'>
 <img src='mask/elephant.png' height="123px" width="123px">
 <img src='style/wave.jpg' height="123px" width="123px">
 <img src='style/night.jpg' height="123px" width="123px">
 <img src='style/machine.jpg' height="123px" width="123px">
 <img src='style/prayer.jpg' height="123px" width="123px">
</div>

<div align='center'>
 <img src='content/elephant.jpg' height="311.5px">
 <img src='output/elephant.png' height="311.5px">
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

I recommend creating **isolated Python environments** using [Virtualenv](https://virtualenv.pypa.io/en/stable/). **Vital** to **avoid dependencies conflicts**  :dizzy_face: when working on different projects. For those who are not familiar with Virtualenv, here are 2 tutorials that should help you get started, [tuto1](http://thepythonguru.com/python-virtualenv-guide/) and [tuto2](http://www.simononsoftware.com/virtualenv-tutorial-part-2/). Lastly, I' m using [Ubuntu16.04](https://www.ubuntu.com/download/desktop), but the code should run on Windows and macOs.

 :warning: **Remarks**: 
- make sure that your **project interpreter** is **properly configured**. 
- In Pycharm you can check which interpreter is used by clicking: ```File -> Settings -> Project```. Pycharm lists all the interpreters that you have already used. If you have created a new virtual environment and want to set it as the project interpreter click on the small gear, ```Add Local``` and locate it.

### Model Weigths: VGG19
- **Download** the weights [here]( http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat).
- Recall that **style transfer** is based on **perceptual losses**. Those losses allow to **separate style and content**. They are computed from **high level features**. As in the orignal work [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), we extract those features using a **VGG19** pretrained on image classification.  

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
- Greats tips to obtain good results from [titu1994/Neural-Style-Transfer](https://github.com/titu1994/Neural-Style-Transfer)
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
