# Single Shot MultiBox Detector Implementation in Tensorflow 2.0

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). 
The implementation is a rewrite of [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) so all credit for implementation details go to them.
The design goal is modularity and extensibility.

Currently, it has MobileNetV1, MobileNetV2, and VGG based SSD/SSD-Lite implementations. 

Added features include TFRecord support as well as a data generator that augments a directory of masked
images and pastes them on top of provided background images.