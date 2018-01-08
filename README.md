# semantic-style-transfer
Tensorflow implementation of semantic style transfer based on guided Gram Matrices
Say that it uses the slow model (imqge optimisation problem)

In construction...

## Contents
Put hyperlink to the different section

## Examples

## Implementation Details

## Installation
Say that is pretty easy( no need to recomplie the project)
Copy this repository using the git clone
Configure a new project using a python IDE (say that I personnaly use Pycharm), useful to debugg
Configure project interpreter (chose the virtual env)

```
git clone https://github.com/nicolastah/semantic-style-transfer
```

### Dependencies
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [TensorFlow](https://www.tensorflow.org)
- [SciPy](https://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)
- [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)
- CUDA (GPU) -- Recommended
- CUDNN (GPU) -- Recommended

I recommend creating **isolated Python environments** using [Virtualenv](https://virtualenv.pypa.io/en/stable/). **Vital** to **avoid dependencies conflicts** when working on different projects. Here are 2 tutorials that should help you get started, [tuto1](http://thepythonguru.com/python-virtualenv-guide/) and [tuto2](http://www.simononsoftware.com/virtualenv-tutorial-part-2/). Say that using Linux is easier than windows to install everything

### Model Weigths
VGG19, put the link of where you cn find it.  Explained briefly what is it and why we are using such model
talk about perceptual losses, say that need to used VGG trained on image classification, can make a reference to paper Exploring neural style

### Speed
Talk about the speed with GPU. Do it with different mage size (similar as Titus). Then do the same without GPU. Say that it is also possbile but it is slower

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

## Acknowledgement
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
