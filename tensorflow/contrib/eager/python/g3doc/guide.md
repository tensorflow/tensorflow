# TensorFlow Eager Execution

## What is this?

Eager execution is a feature that makes TensorFlow execute operations
immediately: concrete values are returned, instead of a computational graph to
be executed later.

As a result, enabling eager execution provides:

-   A [NumPy](http://www.numpy.org/)-like library for numerical computation with
    support for GPU acceleration and automatic differentiation.
-   A flexible platform for machine learning research and experimentation.

Eager execution is under active development. This guide walks through an
alpha/preview release. In particular, not all TensorFlow APIs currently work
with eager execution enabled, and some models may be slow to execute, compared
to models defined without using eager execution.

## Installation

Eager execution is **not** included in the latest release (version 1.4) of
TensorFlow. To use it, you will need to [build TensorFlow from
source](https://www.tensorflow.org/install/install_sources) or install the
nightly builds.

For example, the nightly builds can be installed using `pip`:

-   `pip install tf-nightly` (for CPU-only TensorFlow)
-   `pip install tf-nightly-gpu` (for GPU-enabled TensorFlow)

Or using `docker`, with [Jupyter Notebook](http://jupyter.org/) support:

```sh
# For CPU-only TensorFlow
docker pull tensorflow/tensorflow:nightly
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly

# For GPU-enabled TensorFlow:
# (Requires https://github.com/NVIDIA/nvidia-docker)
nvidia-docker pull tensorflow/tensorflow:nightly-gpu
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-gpu
```

## Getting Started

With TensorFlow installed, eager execution is enabled via a single call:

```python
import tensorflow as tf

import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()
```

Enabling eager execution changes how TensorFlow functions behave (in particular,
`Tensor` objects will reference concrete values instead of being symbolic
handles to nodes in a computational graph). As a result, eager execution should
be enabled at the beginning of a program and cannot be disabled afterwards in
the same program.

Code examples in the rest of this guide assume that eager execution has been
enabled.

## A library for numerical computation

A significant fraction of the [TensorFlow
API](https://www.tensorflow.org/api_docs/python/) consists of numerical
operations:
[arithmetic operations](https://www.tensorflow.org/api_guides/python/math_ops#Arithmetic_Operators),
[matrix operations](https://www.tensorflow.org/api_guides/python/math_ops#Matrix_Math_Functions),
[linear algebra operations](https://www.tensorflow.org/versions/master/api_docs/python/tf/linalg),
etc.

With eager execution enabled, these operations consume and return
multi-dimensional arrays as `Tensor` objects, similar to NumPy
[`ndarray`s](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html).
For example:

```python
# Multiply two 2x2 matrices
x = tf.matmul([[1, 2],
               [3, 4]],
              [[4, 5],
               [6, 7]])
# Add one to each element
# (tf.add supports broadcasting)
y = tf.add(x, 1)

# Create a random random 5x3 matrix
z = tf.random_uniform([5, 3])

print(x)
print(y)
print(z)
```

Output:

```
tf.Tensor(
[[16 19]
 [36 43]], shape=(2, 2), dtype=int32)
tf.Tensor(
[[17 20]
 [37 44]], shape=(2, 2), dtype=int32)
tf.Tensor(
[[ 0.25058532  0.0929395   0.54113817]
 [ 0.3108716   0.93350542  0.84909797]
 [ 0.53081679  0.12788558  0.01767385]
 [ 0.29725885  0.33540785  0.83588314]
 [ 0.38877153  0.39720535  0.78914213]], shape=(5, 3), dtype=float32)
```

For convenience, these operations can also be triggered via operator overloading
of the `Tensor` object. For example, the `+` operator is equivalent to `tf.add`,
`-` to `tf.subtract`, `*` to `tf.multiply`, etc.:

```python
x = (tf.ones([1], dtype=tf.float32) + 1) * 2 - 1
print(x)
```

Output:

```
tf.Tensor([ 3.], shape=(1,), dtype=float32)
```

### Converting to and from NumPy

The operations above automatically convert Python objects (like lists of
numbers) and NumPy arrays to `Tensor` objects. `Tensor` objects can also be used
as NumPy arrays by numpy operations.

```python
import numpy as np

x = tf.add(1, 1)                     # tf.Tensor with a value of 2
y = tf.add(np.array(1), np.array(1)) # tf.Tensor with a value of 2
z = np.multiply(x, y)                # numpy.int64 with a value of 4
```

Alternatively, they can be explicitly converted using
[`tf.constant`](https://www.tensorflow.org/api_docs/python/tf/constant), as
shown in the next example.

Conversely, you can call the `numpy()` method of a `Tensor` object' to obtain
its NumPy `ndarray` value. For example:

```python
import numpy as np

np_x = np.array(2., dtype=np.float32)
x = tf.constant(np_x)

py_y = 3.
y = tf.constant(py_y)

z = x + y + 1

print(z)
print(z.numpy())
```

Output:

```
tf.Tensor(6.0, shape=(), dtype=float32)
6.0
```

### GPU acceleration

Many TensorFlow operations support GPU acceleration. With eager execution
enabled, [computation is *not* automatically
offloaded](https://www.tensorflow.org/tutorials/using_gpu) to GPUs. Instead, you
must explicitly specify when GPUs should be used.

The simplest way to do this is to enclose your computation in a `with
tf.device('/gpu:0')` block. Also of interest is the `tfe.num_gpus()` function,
which returns the number of available GPUs.

For example, consider this snippet to measure the time to multiply two 1000x1000
matrices on CPU:

```python
import time

def measure(x):
  # The very first time a GPU is used by TensorFlow, it is initialized.
  # So exclude the first run from timing.
  tf.matmul(x, x)

  start = time.time()
  for i in range(10):
    tf.matmul(x, x)
  end = time.time()

  return "Took %s seconds to multiply a %s matrix by itself 10 times" % (end - start, x.shape)

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: %s" % measure(tf.random_normal([1000, 1000])))

# If a GPU is available, run on GPU:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: %s" % measure(tf.random_normal([1000, 1000])))
```

Output (exact numbers will depend on the characteristics of the hardware):

```python
CPU: Took 0.145531892776 seconds to multiply a (1000, 1000) matrix by itself 10 times
GPU: Took 0.000458955764771 seconds to multiply a (1000, 1000) matrix by itself 10 times
```

Alternatively, methods on the `Tensor` object can be used to explicitly copy the
`Tensor` to a different device. Operations are typically executed on the device
on which the inputs are placed. For example:

```python
x = tf.random_normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)  # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

if tfe.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1
```

### Automatic Differentiation

[Automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is
very useful when implementing many machine learning algorithms (e.g.,
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) for training
neural networks). For this purpose, TensorFlow eager execution provides an
[autograd](https://github.com/HIPS/autograd)-style API for automatic
differentiation. Specifically, the functions:

-   `tfe.gradients_function(f)`: Returns a Python function that computes the
    derivatives of the Python function `f` with respect to its arguments. `f`
    must return a scalar value. When the returned function is invoked, it
    returns a list of `Tensor` objects (one element for each argument of `f`).
-   `tfe.value_and_gradients_function(f)`: Similar to `tfe.gradients_function`,
    except that when the returned function is invoked, it returns the value of
    `f` in addition to the list of derivatives of `f` with respect to its
    arguments.

These functions naturally apply to higher order differentiation as well. For
example:

```python
def f(x):
  return tf.multiply(x, x)  # Or x * x
assert 9 == f(3.).numpy()

df = tfe.gradients_function(f)
assert 6 == df(3.)[0].numpy()

# Second order deriviative.
d2f = tfe.gradients_function(lambda x: df(x)[0])
assert 2 == d2f(3.)[0].numpy()

# Third order derivative.
d3f = tfe.gradients_function(lambda x : d2f(x)[0])
assert 0 == d3f(3.)[0].numpy()
```

These functions can be used to train models. For example, consider the following
simple linear regression model:

```python
def prediction(input, weight, bias):
  return input * weight + bias

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# A loss function: Mean-squared error
def loss(weight, bias):
  error = prediction(training_inputs, weight, bias) - training_outputs
  return tf.reduce_mean(tf.square(error))

# Function that returns the the derivative of loss with respect to
# weight and bias
grad = tfe.gradients_function(loss)

# Train for 200 steps (starting from some random choice for W and B, on the same
# batch of data).
W = 5.
B = 10.
learning_rate = 0.01
print("Initial loss: %f" % loss(W, B).numpy())
for i in range(200):
  (dW, dB) = grad(W, B)
  W -= dW * learning_rate
  B -= dB * learning_rate
  if i % 20 == 0:
    print("Loss at step %d: %f" % (i, loss(W, B).numpy()))
print("Final loss: %f" % loss(W, B).numpy())
print("W, B = %f, %f" % (W.numpy(), B.numpy()))
```

Output: (the exact numbers may vary depending on the randomness in noise)

```
Initial loss: 66.730003
Loss at step 0: 64.200096
Loss at step 20: 29.872814
Loss at step 40: 14.233772
Loss at step 60: 7.090570
Loss at step 80: 3.819887
Loss at step 100: 2.318821
Loss at step 120: 1.628385
Loss at step 140: 1.310142
Loss at step 160: 1.163167
Loss at step 180: 1.095162
Final loss: 1.064711
W, B = 3.094944, 2.161383
```

To utilize the GPU, place the code above within a `with tf.device("/gpu:0"):`
block. (However, this particular model, with only two floating point parameters,
is unlikely to benefit from GPU acceleration.)

### Customizing gradients

One may want to define custom gradients for an operation, or for a function.
This may be useful for multiple reasons, including providing a more efficient
or more [numerically stable](https://en.wikipedia.org/wiki/Numerical_stability)
gradient for a sequence of operations.

For example, consider the function `log(1 + e^x)`, which commonly occurs in the
computation of cross entropy and log likelihoods.

```python
def log1pexp(x):
  return tf.log(1 + tf.exp(x))
grad_log1pexp = tfe.gradients_function(log1pexp)

# Works fine at x = 0.
assert 0.5 == float(grad_log1pexp(0.)[0])

# Returns a `nan` at x = 100 due to numerical instability.
import math
assert math.isnan(float(grad_log1pexp(100.)[0]))
```

We can define a custom gradient for the above function that analytically
simplifies the gradient expression.

```python
@tfe.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad
grad_log1pexp = tfe.gradients_function(log1pexp)

# Works as before at x = 0.
assert 0.5 == float(grad_log1pexp(0.)[0])

# But now works at x = 100 as well.
assert 1.0 == float(grad_log1pexp(100.)[0])
```
Also notice how the gradient function implementation reuses an expression
(`tf.exp(x)`) computed during the forward pass, hence making the gradient
computation more efficient by avoiding redundant computation.

## Building and training models

In practice, your computation may have many parameters to be optimized (by
computing derivatives). Encapsulating them into re-usable classes/objects
makes the code easier to follow than writing a single top-level function with
many arguments.

In fact, eager execution encourages use of the [Keras](https://keras.io)-style
"Layer" classes in the
[`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers)
module.

Furthermore, you may want to apply more sophisticated techniques to compute
parameter updates, such as those in
[`tf.train.Optimizer`](https://www.tensorflow.org/api_guides/python/train#Optimizers)
implementations.

This next section walks through using the same `Optimizer` and `Layer` APIs used
to build trainable TensorFlow graphs in an environment where eager execution is
enabled.

### Variables and Optimizers

`tfe.Variable` objects store mutable `Tensor` values that can be accessed during
training, making automatic differentiation easier. In particular, parameters of
a model can be encapsulated in Python classes as variables.

`tfe.gradients_function(f)` introduced earlier computes the derivatives of `f`
with respect to its arguments. However, it requires all parameters of interest
to be arguments of `f`, which becomes cumbersome when `f` depends on a large
number of trainable parameters.

`tfe.implicit_gradients` is an alternative function with some useful properties:

-   It computes the derivatives of `f` with respect to all the `tfe.Variable`s
    used by `f`.
-   When the returned function is invoked, it returns a list of
    (gradient value, Variable object) tuples.

Representing model parameters as `Variable` objects, along with the use of
`tfe.implicit_gradients`, typically results in better encapsulation. For
example, the linear regression model described above can be written into a
class:

```python
class Model(object):
  def __init__(self):
    self.W = tfe.Variable(5., name='weight')
    self.B = tfe.Variable(10., name='bias')

  def predict(self, inputs):
    return inputs * self.W + self.B


# The loss function to be optimized
def loss(model, inputs, targets):
  error = model.predict(inputs) - targets
  return tf.reduce_mean(tf.square(error))

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# Define:
# 1. A model
# 2. Derivatives of a loss function with respect to model parameters
# 3. A strategy for updating the variables based on the derivatives
model = Model()
grad = tfe.implicit_gradients(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# The training loop
print("Initial loss: %f" %
      loss(model, training_inputs, training_outputs).numpy())
for i in range(201):
  optimizer.apply_gradients(grad(model, training_inputs, training_outputs))
  if i % 20 == 0:
    print("Loss at step %d: %f" %
          (i, loss(model, training_inputs, training_outputs).numpy()))
print("Final loss: %f" % loss(model, training_inputs, training_outputs).numpy())
print("W, B = %s, %s" % (model.W.numpy(), model.B.numpy()))
```

Output:

```
Initial loss: 69.693184
Loss at step 0: 66.987854
Loss at step 20: 30.553387
Loss at step 40: 14.250237
Loss at step 60: 6.955020
Loss at step 80: 3.690550
Loss at step 100: 2.229739
Loss at step 120: 1.576032
Loss at step 140: 1.283496
Loss at step 160: 1.152584
Loss at step 180: 1.093999
Final loss: 1.067780
W, B = 3.0114281, 2.0865183
```

Using `implicit_gradients` avoids the need to provide all the trainable
parameters of the model as arguments to the `loss` function.

### Using Keras and the Layers API

[Keras](https://keras.io) is a popular API for defining model structures. The
[`tf.keras.layers`](https://www.tensorflow.org/api_docs/python/tf/keras/layers)
module provides a set of building blocks for models and is implemented using the
`tf.layers.Layer` subclasses in the
[`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers)
module. We encourage the use of these same building blocks when using
TensorFlow's eager execution feature. For example, the very same linear
regression model can be built using `tf.layers.Dense`:

```python
class Model(object):
  def __init__(self):
    self.layer = tf.layers.Dense(1)

  def predict(self, inputs):
    return self.layer(inputs)
```

The `tf.layers` API makes it more convenient to define more sophisticated
models. For example, the following will train an MNIST model:

```python
class MNISTModel(object):
  def __init__(self, data_format):
    # 'channels_first' is typically faster on GPUs
    # while 'channels_last' is typically faster on CPUs.
    # See: https://www.tensorflow.org/performance/performance_guide#data_formats
    if data_format == 'channels_first':
      self._input_shape = [-1, 1, 28, 28]
    else:
      self._input_shape = [-1, 28, 28, 1]
    self.conv1 = tf.layers.Conv2D(32, 5,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  data_format=data_format)
    self.max_pool2d = tf.layers.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
    self.conv2 = tf.layers.Conv2D(64, 5,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  data_format=data_format)
    self.dense1 = tf.layers.Dense(1024, activation=tf.nn.relu)
    self.dropout = tf.layers.Dropout(0.5)
    self.dense2 = tf.layers.Dense(10)

  def predict(self, inputs):
    x = tf.reshape(inputs, self._input_shape)
    x = self.max_pool2d(self.conv1(x))
    x = self.max_pool2d(self.conv2(x))
    x = tf.layers.flatten(x)
    x = self.dropout(self.dense1(x))
    return self.dense2(x)

def loss(model, inputs, targets):
  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          logits=model.predict(inputs), labels=targets))


# Load the training and validation data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("./mnist_data", one_hot=True)

# Train
device = "gpu:0" if tfe.num_gpus() else "cpu:0"
model = MNISTModel('channels_first' if tfe.num_gpus() else 'channels_last')
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
grad = tfe.implicit_gradients(loss)
for i in range(20001):
  with tf.device(device):
    (inputs, targets) = data.train.next_batch(50)
    optimizer.apply_gradients(grad(model, inputs, targets))
    if i % 100 == 0:
      print("Step %d: Loss on training set : %f" %
            (i, loss(model, inputs, targets).numpy()))
print("Loss on test set: %f" % loss(model, data.test.images, data.test.labels).numpy())
```

For a more complete example, see
[`tensorflow/contrib/eager/python/examples/mnist.py`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist/mnist.py)

### Checkpointing trained variables

TensorFlow Variables (`tfe.Variable`) provides a way to represent shared,
persistent state of your model. The `tfe.Saver` class (which is a thin wrapper
over the
[`tf.train.Saver`](https://www.tensorflow.org/api_docs/python/tf/train/Saver)
class) provides a means to save and restore variables to and from _checkpoints_.

For example:

```python
# Create variables.
x = tfe.Variable(10., name='x')
y = tfe.Variable(5., name='y')

# Create a Saver.
saver = tfe.Saver([x, y])

# Assign new values to the variables and save.
x.assign(2.)
saver.save('/tmp/ckpt')

# Change the variable after saving.
x.assign(11.)
assert 16. == (x + y).numpy()  # 11 + 5

# Restore the values in the checkpoint.
saver.restore('/tmp/ckpt')

assert 7. == (x + y).numpy()  # 2 + 5
```

### `tfe.Network`

You may often want to organize your models using classes, like the `MNISTModel`
class described above. We recommend inheriting from the `tfe.Network` class as
it provides conveniences like keeping track of all model variables and methods
to save and restore from checkpoints.

Sub-classes of `tfe.Network` may register `Layer`s (like classes in
[`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers),
or [Keras
layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers))
using a call to `self.track_layer()` and define the computation in an
implementation of `call()`.

Note that `tf.layers.Layer` objects (like `tf.layers.Dense`) create variables
lazily, when the first input is encountered.

For example, consider the following two-layer neural network:

```python
class TwoLayerNet(tfe.Network):
  def __init__(self):
    super(TwoLayerNet, self).__init__()
    self.layer1 = self.track_layer(
      tf.layers.Dense(2, activation=tf.nn.relu, use_bias=False))
    self.layer2 = self.track_layer(tf.layers.Dense(3, use_bias=False))

  def call(self, x):
    return self.layer2(self.layer1(x))

net = TwoLayerNet()

# No variables created yet
assert 0 == len(net.variables)

# They are created on first input:
inp = tf.constant([[1.]])

# Since input is a 1x1 matrix, net.l1 has 2 units and net.l2 has 3 units,
# the output is the product of a 1x1 matrix with a 1x2 matrix with a 2x3
# matrix.
assert [1, 3] == net(inp).shape.as_list()  # Invoke net; get output shape.
assert 1 == len(net.layer1.variables)
assert 1 == len(net.layer2.variables)
assert 2 == len(net.variables)  # weights for each layer.
assert [1, 2] == net.variables[0].shape.as_list()  # weights of layer1.
assert [2, 3] == net.variables[1].shape.as_list()  # weights of layer2.
```

The `tfe.Network` class is itself a sub-class of `tf.layers.Layer`. This allows
instances of `tfe.Network` to be embedded in other networks. For example:

```python
class ThreeLayerNet(tfe.Network):
  def __init__(self):
    super(ThreeLayerNet, self).__init__()
    self.a = self.track_layer(TwoLayerNet())
    self.b = self.track_layer(tf.layers.Dense(4, use_bias=False))

  def call(self, x):
    return self.b(self.a(x))

net = ThreeLayerNet()

assert [1, 4] == net(inp).shape.as_list()
assert 3 == len(net.variables)
assert [1, 2] == net.variables[0].shape.as_list()
assert [2, 3] == net.variables[1].shape.as_list()
assert [3, 4] == net.variables[2].shape.as_list()
```

See more examples in
[`tensorflow/contrib/eager/python/examples`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples).

`tfe.Saver` in combination with `tfe.restore_variables_on_create` provides a
convenient way to save and load checkpoints without changing the program once
the checkpoint has been created. For example, we can set an objective for the
output of our network, choose an optimizer, and a location for the checkpoint:

```python
objective = tf.constant([[2., 3., 4., 5.]])
optimizer = tf.train.AdamOptimizer(0.01)
checkpoint_directory = '/tmp/tfe_example'
checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
net = ThreeLayerNet()
```

Note that variables have not been created yet. We want them to be restored from
a checkpoint, if one exists, so we create them inside a
`tfe.restore_variables_on_create` context manager. Then our training loop is the
same whether starting training or resuming from a previous checkpoint:

```python
with tfe.restore_variables_on_create(
    tf.train.latest_checkpoint(checkpoint_directory)):
  global_step = tf.train.get_or_create_global_step()
  for _ in range(100):
    loss_fn = lambda: tf.norm(net(inp) - objective)
    optimizer.minimize(loss_fn, global_step=global_step)
    if tf.equal(global_step % 20, 0):
      print("Step %d, output %s" % (global_step.numpy(),
                                    net(inp).numpy()))
      all_variables = (
          net.variables
          + optimizer.variables()
          + [global_step])
      # Save the checkpoint.
      tfe.Saver(all_variables).save(checkpoint_prefix, global_step=global_step)
```

The first time it runs, `Network` variables are initialized randomly. Then the
output is trained to match the objective we've set:

```
Step 20, output [[ 0.03575622  0.29863232  0.03474367  0.24735749]]
Step 40, output [[ 0.40646029  0.9856872   0.46851286  0.95358551]]
Step 60, output [[ 1.74541104  2.800704    1.79055595  2.74783421]]
Step 80, output [[ 2.14977384  3.44340849  3.96120024  5.16242075]]
Step 100, output [[ 1.99943113  3.02364397  3.93500996  4.9610076 ]]
```

In subsequent iterations, variables are initialized with the values read from
the latest checkpoint. Running the same code again, we continue from where we
left off:

```
Step 120, output [[ 1.99234128  3.0271616   3.98732996  4.96401167]]
Step 140, output [[ 2.00133467  3.01270437  4.00616646  5.00406504]]
Step 160, output [[ 1.99647415  2.9956708   3.99064088  4.99632359]]
Step 180, output [[ 2.00699997  3.00904822  4.00706148  5.01193142]]
Step 200, output [[ 1.98334622  2.98249531  3.97375059  4.97123432]]
```


### Summaries, metrics and TensorBoard

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
is a popular tool for understanding, debugging and optimizing the model training
process. To benefit from the visualizations offered by TensorBoard, summary
events need to be written during the course of execution of your program. You
might find many Tensorflow programs that include the
[`tf.summary`](https://www.tensorflow.org/api_guides/python/summary) operations
during graph construction.

`tf.summary` operations are *not* compatible with eager execution, but an
equivalent alternative exists in
[`tf.contrib.summary`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/summary)
that is compatible with both eager execution and graph construction.

During model construction simply insert summary operations like
`tf.contrib.summary.scalar`. These operations do nothing by default, unless a
summary writer is currently active and a writing policy is set.

For example, to record summaries once every 100 global steps, use:

```python
tf.train.get_or_create_global_step()  # Ensuring the global step variable exists
writer = tf.contrib.summary.create_summary_file_writer(logdir)

for _ in range(iterations):
  with writer.as_default():
    with tf.contrib.summary.record_summaries_every_n_global_steps(100):
      # your model code goes here
      tf.contrib.summary.scalar('loss', loss)
      # ...
```

See the full mnist example in
[`tensorflow/contrib/eager/python/examples/mnist`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist)
for a full model using `tf.contrib.summary`.

Similarly to summaries, the metrics in `tf.metrics` are currently not compatible
with eager execution. We instead provide object-oriented metrics in the
`tfe.metrics` package, which are compatible with graph construction as well.

Metrics in the `tfe.metrics`, such as `tfe.metrics.Mean` and
`tfe.Metrics.Accuracy`, all implement an intuitive object-oriented
interface. Here's an example of how to use the `tfe.metrics.Mean` metric:

```python
# Metrics are objects, which can be created and destroyed.
my_mean = tfe.metrics.Mean(name='my_mean')
# While a metric is active, you can call it as a function to accumulate into its
# internal state.
my_mean(0.0)
my_mean(10.0)
# Once you've finished updating the metric, you can get its result. In this case
# a simple average over all the calls to it. If a summary writer is active the
# metric will write the appropriate summaries using the metric name.
assert 5.0 == my_mean.result().numpy()
```

For a full example of a model using metrics for evaluation, see the mnist
example in
[`tensorflow/contrib/eager/python/examples/mnist`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist).

### Input Pipelines

The discussion above has been centered around the computation executed by your
model. The
[`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)
module provides APIs to build complex input pipelines from simple, reusable
pieces.

If you're familiar with constructing `tf.data.Dataset` objects when building
TensorFlow graphs, the same API calls are used when eager execution is enabled.
However, the process of iterating over elements of the dataset differs between
eager execution and graph construction. When eager execution is enabled, the
discussion on iterator creation using `make_one_shot_iterator()` and
`get_next()` in the
[Programmer's Guide](https://www.tensorflow.org/programmers_guide/datasets) is
*not* applicable. Instead, a more Pythonic `Iterator` class is available.

For example:

```python
# Create a source Dataset from in-memory numpy arrays.
# For reading from files on disk, you may want to use other Dataset classes
# like the TextLineDataset or the TFRecordDataset.
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Apply transformations, shuffling, batching etc.
dataset = dataset.map(tf.square).shuffle(2).batch(2)

# Use tfe.Iterator to iterate over the dataset.
for x in tfe.Iterator(dataset):
  print(x)
```

Output:

```
tf.Tensor([4 9], shape=(2,), dtype=int32)
tf.Tensor([16 25], shape=(2,), dtype=int32)
tf.Tensor([36  1], shape=(2,), dtype=int32)
```

## Interoperating with Graphs

Eager execution improves the process of model development in Python; however,
because it is in its earliest stages, it does not yet support some features
available to [TensorFlow
graphs](https://www.tensorflow.org/get_started/get_started#the_computational_graph)
that are desirable when deploying models in production. In particular, eager
execution does not yet support distributed training, exporting models (to other
[programming languages](https://www.tensorflow.org/api_docs/), [TensorFlow
serving](https://www.tensorflow.org/serving/), and mobile applications), and
various memory and computation optimizations that are applied to TensorFlow's
dataflow graphs.

That said, the APIs used to build modes are exactly the same whether executing
eagerly or constructing graphs. This means that you can iteratively develop your
model with eager execution enabled and later, if needed, use the same code to
reap the benefits of representing models as computational graphs.

For example,
[`mnist.py`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist/mnist.py)
defines a model that is eagerly executed. That same code is used to construct
and execute a graph in
[`mnist_graph_test.py`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist/mnist_graph_test.py).

Other models in the [examples
directory](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/)
demonstrate this as well.

Some differences worth noting:

-   There is no notion of a `tf.placeholder` or a `tf.Session` when eager
    execution is enabled.
-   Many properties on the `tf.Tensor` object, like `tf.Tensor.name`,
    `tf.Tensor.op`, `tf.Tensor.inputs` are not meaningful when eager execution
    is enabled and their use will raise an `AttributeError`.
-   To use `tfe.implicit_gradients` in graph construction, variables must be
    created with [`use_resource=True`] provided to
    [`tf.get_variable()`](https://www.tensorflow.org/api_docs/python/tf/get_variable)
    or
    [`tf.variable_scope()`](https://www.tensorflow.org/api_docs/python/tf/variable_scope).
-   Some API calls (such as the functional-style `tf.layers.dense`,
    `tf.layers.conv2d`) are not compatible with eager execution. Use of such
    methods should raise an error indicating the alternative (e.g., the
    `tf.layers.Dense` and `tf.layers.Conv2D` classes).

## What next?

Please give eager execution a spin. This feature is in early stages and is
evolving, so we welcome your feedback via issues on GitHub (see [known
issues](https://github.com/tensorflow/tensorflow/labels/comp:eager)).

You may want to browse through some sample code, including benchmarks for some:

-   [Linear Regression](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/linear_regression)
-   [MNIST handwritten digit classifier](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist)
-   [ResNet50 image classification](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/resnet50)
-   [RNN to generate colors](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/rnn_colorbot)
-   [RNN language model](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/rnn_ptb)

