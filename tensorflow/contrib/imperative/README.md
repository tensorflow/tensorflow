## Imperative programming in TensorFlow

In the standard TensorFlow library, the specification of the computation is done
statically in terms of a computation graph, and is separate from the execution
of the graph. This model of programming is referred to as *lazy*, *deferred*,
*dynamic*, or, *asynchronous*. This library brings imperative style programming (à
la [NumPy](http://www.numpy.org)) to TensorFlow. Using this library, you can:

* Write code in an imperative style: the results of the computation are available
  right after the execution of a line of code.
* Use TensorFlow operations on tensors, and get all the benefits of GPU
  acceleration.
* Include any Python control flow statements like `while` and `if` when
  specifying the computation.
* Perform automatic differentiation on your code with the
  standard
  [`tf.gradients`](https://www.tensorflow.org/api_docs/python/train/gradient_computation#gradients) function.

### Getting started

This library is a thin wrapper over the standard TensorFlow Python library. The
source code is
available
[here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/imperative). You
can get started on Linux by installing the nightly PIP package linked off 
[the main page](https://github.com/tensorflow/tensorflow). Please
consult [this](https://github.com/tensorflow/tensorflow#installation) document for other platforms and the PIP package including GPU
support.


### Write your first imperative TensorFlow program

```shell
$ python
```

```python
>>> import tensorflow.contrib.imperative as tf
>>> x = tf.constant([[7.], [6]])
>>> y = tf.constant([[6., 7]])
>>> tf.matmul(x, y)
array([[ 42.,  49.],
       [ 36.,  42.]], dtype=float32)
```

Note that this code is identical in terms of the programmer's mental model to
the following NumPy code:

```python
>>> import numpy as np
>>> x = np.array([[7.], [6]])
>>> y = np.array([[6., 7]])
>>> x.dot(y)
array([[ 42.,  49.],
       [ 36.,  42.]])
```

The library can be imported as `import tensorflow.contrib.imperative as tf`
(contrast with importing standard TensorFlow, which is done as `import
tensorflow as tf`). This import statement makes all of standard TensorFlow
available in the `tf` symbol. However, it is not necessary to create a session
object and set it up to run and fetch tensors.


### Features

The library provides the following additional features on top of standard
TensorFlow:

* Tensors are automatically fetched when used in contexts that expect their
  value.

  - Printing

  ```python
  x = tf.constant(10)
  y = tf.constant(32)
  print(x + y)
  42
  ```

  - Use in conditionals

  ```python
  x = tf.constant(30)
  if x > 4:
    print('Greater than 4')
  Greater than 4

  x = tf.random_normal([3])
  y = x * 2
  while tf.global_norm([y]) < 1000:
    y = y * 2
  print(y)
  [ -213.2868042   -511.02456665  1026.66882324]
  ```

* Variables are automatically initialized, no need to run the
  [`tf.global_variables_initializer()`](https://www.tensorflow.org/api_docs/python/state_ops/variable_helper_functions#global_variables_initializer) operation.

  ```python
  x = tf.Variable(np.random.normal(size=[2, 2]), dtype=tf.float32)
  y = tf.constant([[1, 2.]])
  z = tf.matmul(y, x)
  print(z)
  array([[-1.231673  ,  3.14744973]], dtype=float32)
  ```

* Gradients work as expected using the standard `tf.gradients` function.

   ```python
   x = tf.Variable(np.random.rand(1, 3))
   y = tf.exp(x)
   dy = tf.gradients(y, x)
   # dy/dx should be equal to y (= exp(x))
   print(y, dy)
   (array([[ 1.79997761,  2.00581881,  2.37302414]]), [array([[ 1.79997761,  2.00581881,  2.37302414]])])
   ```

### Caveats

The library is implemented on top of standard TensorFlow. It still constructs a
graph in the background and defers op execution. But when an op executes for the
first time, its results are cached and the cached value is returned for future
executions, thus providing imperative semantics. Because of this implementation
choice, this library comes with the following caveats:

* **Use inside Python loops:** A graph is constructed and kept around in
  the background, both for just executing using the standard TensorFlow runtime,
  and also for allowing automatic differentiation via `tf.gradients`. This means
  that the graph keeps growing when TensorFlow functions are called inside a
  Python loop. This library provides a `tf.new_step` method that clears the
  graph as well as the cached tensors that have been kept around for gradient
  computation. `tf.new_step` can be used as a context manager around, say, a
  training loop to clear the graph after each training step.

  ```python
  x = tf.Variable(constant_op.constant(1.0))
  for i in range(10):
    # Create a new training step
    with tf.new_step() as step:
      # Perform computation and variable updates
      step.run(tf.assign_sub(x, 0.1))
      self.assertAllClose(tf.identity(x), 1.0 - (i + 1) * 0.1)
      # The graph within this context is cleared at this point.
  ```

* **Speed:** Redundant graph construction and caching of tensor values adds
  overheads that are not present in standard TensorFlow, where typically the
  graph is constructed once and executed multiple times. This library is
  intended as a vehicle to prototype the imperative programming model in
  TensorFlow. The runtime overheads can be alleviated with various optimizations
  to the runtime that would equally benefit the deferred execution mode as
  well.

