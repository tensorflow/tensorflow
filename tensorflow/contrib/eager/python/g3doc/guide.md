# Eager execution

Eager execution is a feature that makes TensorFlow execute operations
immediately: concrete values are returned, instead of creating a computational
graph that is executed later.

A user guide is available: https://www.tensorflow.org/programmers_guide/eager
([source file](../../../../docs_src/programmers_guide/eager.md))

We welcome feedback through [GitHub issues](https://github.com/tensorflow/tensorflow/labels/comp:eager).

Sample code is available, including benchmarks for some:

- [Linear Regression](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/linear_regression)
- [MNIST handwritten digit classifier](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist)
- [ResNet50 image classification](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/resnet50)
- [RNN to generate colors](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/rnn_colorbot)
- [RNN language model](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/rnn_ptb)
