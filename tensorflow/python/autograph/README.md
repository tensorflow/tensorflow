# AutoGraph

IMPORTANT: AutoGraph is beta software, and under active development. Expect rough edges and bugs, but if you try it, we appreciate early feedback! We'd also love contributions ([please see our contributing guidelines](CONTRIBUTING.md) and our [style guide](STYLE_GUIDE.md)).

AutoGraph is a Python to TensorFlow compiler.

With AutoGraph, you can write [Eager style](https://www.tensorflow.org/guide/eager) code in a concise manner, and run it as a TensorFlow graph. AutoGraph uses source code transformation and partial evaluation to generate Python code that builds an equivalent TensorFlow subgraph. The result is code that behaves like ops and can be freely combined with other TensorFlow ops.  [Please see this file for which parts of the Python language we currently support](LIMITATIONS.md).

For example, this Python function:

```
def f(x):
  if x < 0:
    x = -x
  return x
```

would be converted to this:

```
def graph_mode_f(x):
  with tf.name_scope('f'):

    def if_true():
      with tf.name_scope('if_true'):
        x_1, = x,
        x_1 = tf.negative(x_1)
        return x_1,

    def if_false():
      with tf.name_scope('if_false'):
        x_1, = x,
        return x_1,
    x = ag__.utils.run_cond(tf.greater(x, 0), if_true, if_false)
    return x
```

so you can use it like an op:

```
with tf.Graph().as_default():
  x = tf.constant(-1.0)

  converted_f = autograph.to_graph(f)
  y = converted_f(x)

  with tf.Session() as sess:
    print(sess.run(y))
    # Output: 1
```

# Getting started

Use AutoGraph in one of the following ways, described below:

 1. Annotations (simpler)
 2. Functional API (more flexible)

To get started, install the latest nightly TensorFlow build:

```shell
pip install -U tf-nightly
```

Then import the `autograph` module from `tf.contrib`:

```
from tensorflow.python import autograph as ag
```

### Related links

Articles:

 * [TensorFlow blog post](https://medium.com/tensorflow/autograph-converts-python-into-tensorflow-graphs-b2a871f87ec7)

Interactive notebooks:

 * [Quick guide](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/guide/autograph.ipynb)
 * [RNN trained using Keras and Estimators](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/autograph/examples/notebooks/rnn_keras_estimator.ipynb)
 * [Demo from the TF Dev Summit 2018](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/autograph/examples/notebooks/dev_summit_2018_demo.ipynb)
 * [Basic control flow speed test](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/autograph/examples/notebooks/ag_vs_eager_collatz_speed_test.ipynb)
 * [MNIST training speed test](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/autograph/examples/notebooks/ag_vs_eager_mnist_speed_test.ipynb)
 * [Basic algorithm samples](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/autograph/examples/notebooks/algorithms.ipynb)
 * [Introductory workshop support notebook](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/autograph/examples/notebooks/workshop.ipynb)

## Using with annotations

Annotating a function or class with `@convert` converts it in place:

```
@ag.convert()
def f(x):
  if x < 0:
    x = -x
  return x
```

... so that it always outputs TensorFlow code:

```
with tf.Graph().as_default():
  x = tf.constant(-1)

  y = f(x)

  with tf.Session() as sess:
    print(sess.run(y))
    # Output: 1
```

## Using the functional API

The functional API allows you to convert an existing function, class or object after it was defined:

```
converted_f = ag.to_graph(f)

print(converted_f(tf.constant(-1)))
# Output: Tensor

print(f(-1))
# Output: 1
```

You can use the functional API to inspect the generated code as well:

```
print(ag.to_code(f))
# Output: <Python and TensorFlow code>
```

## Filing bugs and feature requests

### Reporting a bug

 - If AutoGraph-generated code is compiling and running, but producing an incorrect result, send us a minimal reproduction case that includes the original Eager code, the inputs and if possible, the outputs or the error message.
 - If AutoGraph-generated code is compiling, but not running, send us a minimal reproduction case that includes the original Eager code, the inputs and if possible, the outputs or the error message.
 - If AutoGraph-generated code is not compiling, send us two minimal pieces of code. First, the Eager code that you would like to write, and second, the Graph code that you would like AutoGraph to have generated for you.

### Requesting a feature

If you’d like AutoGraph to convert a feature of Python or TF that we currently don’t handle, please let us know by filing a bug. We’ll make it as easy as possible to interact with us through there.
