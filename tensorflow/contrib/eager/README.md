# Eager Execution

Eager execution provides an imperative interface to TensorFlow (similar to
[NumPy](http://www.numpy.org)). When you enable eager execution, TensorFlow
operations execute immediately; you do not execute a pre-constructed graph with
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
support for distributed and multi-GPU training and performance.

- [Known issues](https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue%20is%3Aopen%20label%3Acomp%3Aeager)
- Feedback is welcome, please consider
  [filing an issue](https://github.com/tensorflow/tensorflow/issues/new) to provide it.

## Installation

For eager execution, we recommend using TensorFlow version 1.8 or newer.
Installation instructions at https://www.tensorflow.org/install/

## Documentation

For an introduction to eager execution in TensorFlow, see:

- [User Guide](https://www.tensorflow.org/get_started/eager) ([source](../../docs_src/guide/eager.md))
- Notebook: [Basic Usage](python/examples/notebooks/1_basics.ipynb)
- Notebook: [Gradients](python/examples/notebooks/2_gradients.ipynb)
- Notebook: [Importing Data](python/examples/notebooks/3_datasets.ipynb)
