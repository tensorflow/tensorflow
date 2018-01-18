# Eager Execution

> *WARNING*: This is a preview/pre-alpha version. The API and performance
> characteristics are subject to change.

Eager execution is an experimental interface to TensorFlow that provides an
imperative programming style (à la [NumPy](http://www.numpy.org)). When you
enable eager execution, TensorFlow operations execute immediately; you do not
execute a pre-constructed graph with
[`Session.run()`](https://www.tensorflow.org/api_docs/python/tf/Session).

For example, consider a simple computation in TensorFlow:

```python
x = tf.placeholder(tf.float32, shape=[1, 1])
m = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(m, feed_dict={x: [[2.]]}))

# Will print [[4.]]
```

Eager execution makes this much simpler:

```python
x = [[2.]]
m = tf.matmul(x, x)

print(m)
```

## Caveats

This feature is in early stages and work remains to be done in terms of smooth
support for distributed and multi-GPU training and CPU performance.

- [Known issues](https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue%20is%3Aopen%20label%3Acomp%3Aeager)
- Feedback is welcome, please consider
  [filing an issue](https://github.com/tensorflow/tensorflow/issues/new) to provide it.

## Installation

Since eager execution is not yet part of a TensorFlow release, using it requires
either [building from source](https://www.tensorflow.org/install/install_sources)
or the latest nightly builds. The nightly builds are available as:

- [`pip` packages](https://github.com/tensorflow/tensorflow/blob/master/README.md#installation) and

- [docker](https://hub.docker.com/r/tensorflow/tensorflow/) images.

For example, to run the latest nightly docker image:

```sh
# If you have a GPU, use https://github.com/NVIDIA/nvidia-docker
nvidia-docker pull tensorflow/tensorflow:nightly-gpu
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-gpu

# If you do not have a GPU, use the CPU-only image
docker pull tensorflow/tensorflow:nightly
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly
```

And then visit http://localhost:8888 in your browser for a Jupyter notebook
environment. Try out the notebooks below.

## Documentation

For an introduction to eager execution in TensorFlow, see:

- [User Guide](python/g3doc/guide.md)
- Notebook: [Basic Usage](python/examples/notebooks/1_basics.ipynb)
- Notebook: [Gradients](python/examples/notebooks/2_gradients.ipynb)
- Notebook: [Importing Data](python/examples/notebooks/3_datasets.ipynb)

## Changelog

- 2017/10/31: Initial preview release.
- 2017/12/01: Example of dynamic neural network:
  [SPINN: Stack-augmented Parser-Interpreter Neural Network](https://arxiv.org/abs/1603.06021).
  See [README.md](python/examples/spinn/README.md) for details.
