# Eager Execution

TensorFlow's eager execution is an imperative programming environment that
evaluates operations immediately, without building graphs: operations return
concrete values instead of constructing a computational graph to run later. This
makes it easy to get started with TensorFlow and debug models, and it
reduces boilerplate as well. To follow along with this guide, run the code
samples below in an interactive `python` interpreter.

Eager execution is a flexible machine learning platform for research and
experimentation, providing:

* *An intuitive interface*—Structure your code naturally and use Python data
  structures. Quickly iterate on small models and small data.
* *Easier debugging*—Call ops directly to inspect running models and test
  changes. Use standard Python debugging tools for immediate error reporting.
* *Natural control flow*—Use Python control flow instead of graph control
  flow, simplifying the specification of dynamic models.

Eager execution supports most TensorFlow operations and GPU acceleration. For a
collection of examples running in eager execution, see:
[tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples).

Note: Some models may experience increased overhead with eager execution
enabled. Performance improvements are ongoing, but please
[file a bug](https://github.com/tensorflow/tensorflow/issues) if you find a
problem and share your benchmarks.

## Setup and basic usage

Upgrade to the latest version of TensorFlow:

```
$ pip install --upgrade tensorflow
```

To start eager execution, add `tf.enable_eager_execution()` to the beginning of
the program or console session. Do not add this operation to other modules that
the program calls.

```py
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()
```

Now you can run TensorFlow operations and the results will return immediately:

```py
tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))  # => "hello, [[4.]]"
```

Enabling eager execution changes how TensorFlow operations behave—now they
immediately evaluate and return their values to Python. `tf.Tensor` objects
reference concrete values instead of symbolic handles to nodes in a computational
graph. Since there isn't a computational graph to build and run later in a
session, it's easy to inspect results using `print()` or a debugger. Evaluating,
printing, and checking tensor values does not break the flow for computing
gradients.

Eager execution works nicely with [NumPy](http://www.numpy.org/). NumPy
operations accept `tf.Tensor` arguments. TensorFlow
[math operations](https://www.tensorflow.org/api_guides/python/math_ops) convert
Python objects and NumPy arrays to `tf.Tensor` objects. The
`tf.Tensor.numpy` method returns the object's value as a NumPy `ndarray`.

```py
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]
#               [3 4]], shape=(2, 2), dtype=int32)

# Broadcasting support
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]
#               [4 5]], shape=(2, 2), dtype=int32)

# Operator overloading is supported
print(a * b)
# => tf.Tensor([[ 2  6]
#               [12 20]], shape=(2, 2), dtype=int32)

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]

# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]
```

The `tf.contrib.eager` module contains symbols available to both eager and graph execution
environments and is useful for writing code to [work with graphs](#work_with_graphs):

```py
tfe = tf.contrib.eager
```

## Dynamic control flow

A major benefit of eager execution is that all the functionality of the host
language is available while your model is executing. So, for example,
it is easy to write [fizzbuzz](https://en.wikipedia.org/wiki/Fizz_buzz):

```py
def fizzbuzz(max_num):
  counter = tf.constant(0)
  for num in range(max_num):
    num = tf.constant(num)
    if num % 3 == 0 and num % 5 == 0:
      print('FizzBuzz')
    elif num % 3 == 0:
      print('Fizz')
    elif num % 5 == 0:
      print('Buzz')
    else:
      print(num)
    counter += 1
  return counter
```

This has conditionals that depend on tensor values and it prints these values
at runtime.

## Build a model

Many machine learning models are represented by composing layers. When
using TensorFlow with eager execution you can either write your own layers or
use a layer provided in the `tf.keras.layers` package.

While you can use any Python object to represent a layer,
TensorFlow has `tf.keras.layers.Layer` as a convenient base class. Inherit from
it to implement your own layer:

```py
class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    self.output_units = output_units

  def build(self, input):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence remove the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.kernel = self.add_variable(
      "kernel", [input.shape[-1], self.output_units])

  def call(self, input):
    # Override call() instead of __call__ so we can perform some bookkeeping.
    return tf.matmul(input, self.kernel)
```

Use `tf.keras.layers.Dense` layer instead  of `MySimpleLayer` above as it has
a superset of its functionality (it can also add a bias).

When composing layers into models you can use `tf.keras.Sequential` to represent
models which are a linear stack of layers. It is easy to use for basic models:

```py
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)
])
```

Alternatively, organize models in classes by inheriting from `tf.keras.Model`.
This is a container for layers that is a layer itself, allowing `tf.keras.Model`
objects to contain other `tf.keras.Model` objects.

```py
class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result

model = MNISTModel()
```

It's not required to set an input shape for the `tf.keras.Model` class since
the parameters are set the first time input is passed to the layer.

`tf.keras.layers` classes create and contain their own model variables that
are tied to the lifetime of their layer objects. To share layer variables, share
their objects.


## Eager training

### Computing gradients

[Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
is useful for implementing machine learning algorithms such as
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) for training
neural networks. During eager execution, use `tf.GradientTape` to trace
operations for computing gradients later.

`tf.GradientTape` is an opt-in feature to provide maximal performance when
not tracing. Since different operations can occur during each call, all
forward-pass operations get recorded to a "tape". To compute the gradient, play
the tape backwards and then discard. A particular `tf.GradientTape` can only
compute one gradient; subsequent calls throw a runtime error.

```py
w = tfe.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
```

Here's an example of `tf.GradientTape` that records forward-pass operations
to train a simple model:

```py
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def prediction(input, weight, bias):
  return input * weight + bias

# A loss function using mean-squared error
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))

# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
  with tf.GradientTape() as tape:
    loss_value = loss(weights, biases)
  return tape.gradient(loss_value, [weights, biases])

train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tfe.Variable(5.)
B = tfe.Variable(10.)

print("Initial loss: {:.3f}".format(loss(W, B)))

for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))
```

Output (exact numbers may vary):

```
Initial loss: 71.204
Loss at step 000: 68.333
Loss at step 020: 30.222
Loss at step 040: 13.691
Loss at step 060: 6.508
Loss at step 080: 3.382
Loss at step 100: 2.018
Loss at step 120: 1.422
Loss at step 140: 1.161
Loss at step 160: 1.046
Loss at step 180: 0.996
Final loss: 0.974
W = 3.01582956314, B = 2.1191945076
```

Replay the `tf.GradientTape` to compute the gradients and apply them in a
training loop. This is demonstrated in an excerpt from the
[mnist_eager.py](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py)
example:

```py
dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
                                              data.train.labels))
...
for (batch, (images, labels)) in enumerate(dataset):
  ...
  with tf.GradientTape() as tape:
    logits = model(images, training=True)
    loss_value = loss(logits, labels)
  ...
  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
```


The following example creates a multi-layer model that classifies the standard
[MNIST handwritten digits](https://www.tensorflow.org/tutorials/layers). It
demonstrates the optimizer and layer APIs to build trainable graphs in an eager
execution environment.

### Train a model

Even without training, call the model and inspect the output in eager execution:

```py
# Create a tensor representing a blank image
batch = tf.zeros([1, 1, 784])
print(batch.shape)  # => (1, 1, 784)

result = model(batch)
# => tf.Tensor([[[ 0.  0., ..., 0.]]], shape=(1, 1, 10), dtype=float32)
```

This example uses the
[dataset.py module](https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py)
from the
[TensorFlow MNIST example](https://github.com/tensorflow/models/tree/master/official/mnist);
download this file to your local directory. Run the following to download the
MNIST data files to your working directory and prepare a `tf.data.Dataset`
for training:

```py
import dataset  # download dataset.py file
dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)
```

To train a model, define a loss function to optimize and then calculate
gradients. Use an optimizer to update the variables:

```py
def loss(model, x, y):
  prediction = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

x, y = iter(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))

# Training loop
for (i, (x, y)) in enumerate(dataset_train):
  # Calculate derivatives of the input function with respect to its parameters.
  grads = grad(model, x, y)
  # Apply the gradient to the model
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

print("Final loss: {:.3f}".format(loss(model, x, y)))
```

Output (exact numbers may vary):

```
Initial loss: 2.674
Loss at step 0000: 2.593
Loss at step 0200: 2.143
Loss at step 0400: 2.009
Loss at step 0600: 2.103
Loss at step 0800: 1.621
Loss at step 1000: 1.695
...
Loss at step 6600: 0.602
Loss at step 6800: 0.557
Loss at step 7000: 0.499
Loss at step 7200: 0.744
Loss at step 7400: 0.681
Final loss: 0.670
```

And for faster training, move the computation to a GPU:

```py
with tf.device("/gpu:0"):
  for (i, (x, y)) in enumerate(dataset_train):
    # minimize() is equivalent to the grad() and apply_gradients() calls.
    optimizer.minimize(lambda: loss(model, x, y),
                       global_step=tf.train.get_or_create_global_step())
```

### Variables and optimizers

`tfe.Variable` objects store mutable `tf.Tensor` values accessed during
training to make automatic differentiation easier. The parameters of a model can
be encapsulated in classes as variables.

Better encapsulate model parameters by using `tfe.Variable` with
`tf.GradientTape`. For example, the automatic differentiation example above
can be rewritten:

```py
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tfe.Variable(5., name='weight')
    self.B = tfe.Variable(10., name='bias')
  def predict(self, inputs):
    return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model.predict(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                            global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
```

Output (exact numbers may vary):

```
Initial loss: 69.066
Loss at step 000: 66.368
Loss at step 020: 30.107
Loss at step 040: 13.959
Loss at step 060: 6.769
Loss at step 080: 3.567
Loss at step 100: 2.141
Loss at step 120: 1.506
Loss at step 140: 1.223
Loss at step 160: 1.097
Loss at step 180: 1.041
Loss at step 200: 1.016
Loss at step 220: 1.005
Loss at step 240: 1.000
Loss at step 260: 0.998
Loss at step 280: 0.997
Final loss: 0.996
W = 2.99431324005, B = 2.02129220963
```

## Use objects for state during eager execution

With graph execution, program state (such as the variables) is stored in global
collections and their lifetime is managed by the `tf.Session` object. In
contrast, during eager execution the lifetime of state objects is determined by
the lifetime of their corresponding Python object.

### Variables are objects

During eager execution, variables persist until the last reference to the object
is removed, and is then deleted.

```py
with tf.device("gpu:0"):
  v = tfe.Variable(tf.random_normal([1000, 1000]))
  v = None  # v no longer takes up GPU memory
```

### Object-based saving

`tfe.Checkpoint` can save and restore `tfe.Variable`s to and from
checkpoints:

```py
x = tfe.Variable(10.)

checkpoint = tfe.Checkpoint(x=x)  # save as "x"

x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')

x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(save_path)

print(x)  # => 2.0
```

To save and load models, `tfe.Checkpoint` stores the internal state of objects,
without requiring hidden variables. To record the state of a `model`,
an `optimizer`, and a global step, pass them to a `tfe.Checkpoint`:

```py
model = MyModel()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = ‘/path/to/model_dir’
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())

root.save(file_prefix=checkpoint_prefix)
# or
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

### Object-oriented metrics

`tfe.metrics` are stored as objects. Update a metric by passing the new data to
the callable, and retrieve the result using the `tfe.metrics.result` method,
for example:

```py
m = tfe.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5
```

#### Summaries and TensorBoard

@{$summaries_and_tensorboard$TensorBoard} is a visualization tool for
understanding, debugging and optimizing the model training process. It uses
summary events that are written while executing the program.

`tf.contrib.summary` is compatible with both eager and graph execution
environments. Summary operations, such as `tf.contrib.summary.scalar`, are
inserted during model construction. For example, to record summaries once every
100 global steps:

```py
writer = tf.contrib.summary.create_file_writer(logdir)
global_step=tf.train.get_or_create_global_step()  # return global step var

writer.set_as_default()

for _ in range(iterations):
  global_step.assign_add(1)
  # Must include a record_summaries method
  with tf.contrib.summary.record_summaries_every_n_global_steps(100):
    # your model code goes here
    tf.contrib.summary.scalar('loss', loss)
     ...
```

## Advanced automatic differentiation topics

### Dynamic models

`tf.GradientTape` can also be used in dynamic models. This example for a
[backtracking line search](https://wikipedia.org/wiki/Backtracking_line_search)
algorithm looks like normal NumPy code, except there are gradients and is
differentiable, despite the complex control flow:

```py
def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # Variables are automatically recorded, but manually watch a tensor
    tape.watch(init_x)
    value = fn(init_x)
  grad = tape.gradient(value, init_x)
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value
```

### Additional functions to compute gradients

`tf.GradientTape` is a powerful interface for computing gradients, but there
is another [Autograd](https://github.com/HIPS/autograd)-style API available for
automatic differentiation. These functions are useful if writing math code with
only tensors and gradient functions, and without `tfe.Variables`:

* `tfe.gradients_function` —Returns a function that computes the derivatives
  of its input function parameter with respect to its arguments. The input
  function parameter must return a scalar value. When the returned function is
  invoked, it returns a list of `tf.Tensor` objects: one element for each
  argument of the input function. Since anything of interest must be passed as a
  function parameter, this becomes unwieldy if there's a dependency on many
  trainable parameters.
* `tfe.value_and_gradients_function` —Similar to
  `tfe.gradients_function`, but when the returned function is invoked, it
  returns the value from the input function in addition to the list of
  derivatives of the input function with respect to its arguments.

In the following example, `tfe.gradients_function` takes the `square`
function as an argument and returns a function that computes the partial
derivatives of `square` with respect to its inputs. To calculate the derivative
of `square` at `3`, `grad(3.0)` returns `6`.

```py
def square(x):
  return tf.multiply(x, x)

grad = tfe.gradients_function(square)

square(3.)  # => 9.0
grad(3.)    # => [6.0]

# The second-order derivative of square:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
gradgrad(3.)  # => [2.0]

# The third-order derivative is None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.)  # => [None]


# With flow control:
def abs(x):
  return x if x > 0. else -x

grad = tfe.gradients_function(abs)

grad(3.)   # => [1.0]
grad(-3.)  # => [-1.0]
```

### Custom gradients

Custom gradients are an easy way to override gradients in eager and graph
execution. Within the forward function, define the gradient with respect to the
inputs, outputs, or intermediate results. For example, here's an easy way to clip
the norm of the gradients in the backward pass:

```py
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn
```

Custom gradients are commonly used to provide a numerically stable gradient for a
sequence of operations:

```py
def log1pexp(x):
  return tf.log(1 + tf.exp(x))
grad_log1pexp = tfe.gradients_function(log1pexp)

# The gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]

# However, x = 100 fails because of numerical instability.
grad_log1pexp(100.)  # => [nan]
```

Here, the `log1pexp` function can be analytically simplified with a custom
gradient. The implementation below reuses the value for `tf.exp(x)` that is
computed during the forward pass—making it more efficient by eliminating
redundant calculations:

```py
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad

grad_log1pexp = tfe.gradients_function(log1pexp)

# As before, the gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]

# And the gradient computation also works at x = 100.
grad_log1pexp(100.)  # => [1.0]
```

## Performance

Computation is automatically offloaded to GPUs during eager execution. If you
want control over where a computation runs you can enclose it in a
`tf.device('/gpu:0')` block (or the CPU equivalent):

```py
import time

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
    _ = x.numpy()  # Make sure to execute op and not just enqueue it
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))

# Run on GPU, if available:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random_normal(shape), steps)))
else:
  print("GPU: not found")
```

Output (exact numbers depend on hardware):

```
Time to multiply a (1000, 1000) matrix by itself 200 times:
CPU: 4.614904403686523 secs
GPU: 0.5581181049346924 secs
```

A `tf.Tensor` object can be copied to a different device to execute its
operations:

```py
x = tf.random_normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

if tfe.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1
```

### Benchmarks

For compute-heavy models, such as
[ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50)
training on a GPU, eager execution performance is comparable to graph execution.
But this gap grows larger for models with less computation and there is work to
be done for optimizing hot code paths for models with lots of small operations.


## Work with graphs

While eager execution makes development and debugging more interactive,
TensorFlow graph execution has advantages for distributed training, performance
optimizations, and production deployment. However, writing graph code can feel
different than writing regular Python code and more difficult to debug.

For building and training graph-constructed models, the Python program first
builds a graph representing the computation, then invokes `Session.run` to send
the graph for execution on the C++-based runtime.  This provides:

* Automatic differentiation using static autodiff.
* Simple deployment to a platform independent server.
* Graph-based optimizations (common subexpression elimination, constant-folding, etc.).
* Compilation and kernel fusion.
* Automatic distribution and replication (placing nodes on the distributed system).

Deploying code written for eager execution is more difficult: either generate a
graph from the model, or run the Python runtime and code directly on the server.

### Write compatible code

The same code written for eager execution will also build a graph during graph
execution. Do this by simply running the same code in a new Python session where
eager execution is not enabled.

Most TensorFlow operations work during eager execution, but there are some things
to keep in mind:

* Use `tf.data` for input processing instead of queues. It's faster and easier.
* Use object-oriented layer APIs—like `tf.keras.layers` and
  `tf.keras.Model`—since they have explicit storage for variables.
* Most model code works the same during eager and graph execution, but there are
  exceptions. (For example, dynamic models using Python control flow to change the
  computation based on inputs.)
* Once eager execution is enabled with `tf.enable_eager_execution`, it
  cannot be turned off. Start a new Python session to return to graph execution.

It's best to write code for both eager execution *and* graph execution. This
gives you eager's interactive experimentation and debuggability with the
distributed performance benefits of graph execution.

Write, debug, and iterate in eager execution, then import the model graph for
production deployment. Use `tfe.Checkpoint` to save and restore model
variables, this allows movement between eager and graph execution environments.
See the examples in:
[tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples).

### Use eager execution in a graph environment

Selectively enable eager execution in a TensorFlow graph environment using
`tfe.py_func`. This is used when `tf.enable_eager_execution()` has *not*
been called.

```py
def my_py_func(x):
  x = tf.matmul(x, x)  # You can use tf ops
  print(x)  # but it's eager!
  return x

with tf.Session() as sess:
  x = tf.placeholder(dtype=tf.float32)
  # Call eager function in graph!
  pf = tfe.py_func(my_py_func, [x], tf.float32)
  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]
```
