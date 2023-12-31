# LeNet MNIST

This project is a simple implementation of the LeNet architecture, written in PyTorch, proposed in: LeCun, Y.; Bottou, L.; Bengio, Y. & Haffner, P. (1998). _Gradient-based learning applied to document recognition_. The network is trained on the MNIST handrawn numbers dataset.

## Loss function

The loss function of choice is the Cross Entropy loss function

## Optimizer

The algorithm used for optimization is the Adam optimizer

## Architecture

![image](https://github.com/AlexanderJernstrom/lenet-mnist/assets/46424392/1563ffc8-8a95-456f-bf77-d7bf5db755ab)
Image comes from wikipedia:
https://en.wikipedia.org/wiki/LeNet#/media/File:Comparison_image_neural_networks.svg

The code implements the architecture highlighted in LeNet part of the image.

## Results
This architecture reached on an M2 Pro 97.3% accuracy on the MNIST dataset
