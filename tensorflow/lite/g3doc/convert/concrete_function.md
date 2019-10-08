# Generate a concrete function

In order to convert TensorFlow 2.0 models to TensorFlow Lite, the model needs to
be exported as a concrete function. This document outlines what a concrete
function is and how to generate one for an existing model.

[TOC]

## Background

In TensorFlow 2.0, eager execution is on by default. TensorFlow's eager
execution is an imperative programming environment that evaluates operations
immediately, without building graphs. Operations return concrete values instead
of constructing a computational graph to run later. A detailed guide on eager
execution is available
[here](https://www.tensorflow.org/guide/eager).

While running imperatively with eager execution makes development and debugging
more interactive, it doesn't allow for deploying on-device. The `tf.function`
API makes it possible to save models as graphs, which is required to run
TensorFlow Lite in 2.0. All operations wrapped in the `tf.function` decorator
can be exported as a graph which can then be converted to the TensorFlow Lite
FlatBuffer format.

## Terminology

The following terminology is used in this document:

*   **Signature** - The inputs and outputs for a set of operations.
*   **Concrete function** - Graph with a single signature.
*   **Polymorphic function** - Python callable that encapsulates several
    concrete function graphs behind one API.

## Methodology

This section describes how to export a concrete function.

### Annotate functions with `tf.function`

Annotating a function with `tf.function` generates a *polymorphic function*
containing those operations. All operations that are not annotated with
`tf.function` will be evaluated with eager execution. The examples below show
how to use `tf.function`.

```python
@tf.function
def pow(x):
  return x ** 2
```

```python
tf.function(lambda x : x ** 2)
```

### Create an object to save

The `tf.function` can be optionally stored as part of a `tf.Module` object.
Variables should only be defined once within the `tf.Module`. The examples below
show two different approaches for creating a class that derives `Checkpoint`.

```python
class BasicModel(tf.Module):

  def __init__(self):
    self.const = None

  @tf.function
  def pow(self, x):
    if self.const is None:
      self.const = tf.Variable(2.)
    return x ** self.const

root = BasicModel()
```

```python
root = tf.Module()
root.const = tf.Variable(2.)
root.pow = tf.function(lambda x : x ** root.const)
```

### Exporting the concrete function

The concrete function defines a graph that can be converted to TensorFlow Lite
model or be exported to a SavedModel. In order to export a concrete function
from the polymorphic function, the signature needs to be defined. The signature
can be defined the following ways:

*   Define `input_signature` parameter in `tf.function`.
*   Pass in `tf.TensorSpec` into `get_concrete_function`: e.g.
    `tf.TensorSpec(shape=[1], dtype=tf.float32)`.
*   Pass in a sample input tensor into `get_concrete_function`: e.g.
    `tf.constant(1., shape=[1])`.

The follow example shows how to define the `input_signature` parameter for
`tf.function`.

```python
class BasicModel(tf.Module):

  def __init__(self):
    self.const = None

  @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
  def pow(self, x):
    if self.const is None:
      self.const = tf.Variable(2.)
    return x ** self.const

# Create the tf.Module object.
root = BasicModel()

# Get the concrete function.
concrete_func = root.pow.get_concrete_function()
```

The example below passes in a sample input tensor into `get_concrete_function`.

```python
# Create the tf.Module object.
root = tf.Module()
root.const = tf.Variable(2.)
root.pow = tf.function(lambda x : x ** root.const)

# Get the concrete function.
input_data = tf.constant(1., shape=[1])
concrete_func = root.pow.get_concrete_function(input_data)
```

## Example program

```python
import tensorflow as tf

# Initialize the tf.Module object.
root = tf.Module()

# Instantiate the variable once.
root.var = None

# Define a function so that the operations aren't computed in advance.
@tf.function
def exported_function(x):
  # Each variable can only be defined once. The variable can be defined within
  # the function but needs to contain a reference outside of the function.
  if root.var is None:
    root.var = tf.Variable(tf.random.uniform([2, 2]))
  root.const = tf.constant([[37.0, -23.0], [1.0, 4.0]])
  root.mult = tf.matmul(root.const, root.var)
  return root.mult * x

# Save the function as part of the tf.Module object.
root.func = exported_function

# Get the concrete function.
concrete_func = root.func.get_concrete_function(
  tf.TensorSpec([1, 1], tf.float32))
```

## Common Questions

### How do I save a concrete function as a SavedModel?

Users who want to save their TensorFlow model before converting it to TensorFlow
Lite should save it as a SavedModel. After getting the concrete function, call
`tf.saved_model.save` to save the model. The example above can be saved using
the following instruction.

```python
tf.saved_model.save(root, export_dir, concrete_func)
```

Reference the
[SavedModel guide](https://www.tensorflow.org/guide/saved_model)
for detailed instructions on using SavedModels.

### How do I get a concrete function from the SavedModel?

Each concrete function within a SavedModel can be identified by a signature key.
The default signature key is `tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY`.
The example below shows how to get the concrete function from a model.

```python
model = tf.saved_model.load(export_dir)
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
```

### How do I get a concrete function for a `tf.Keras` model?

There are two approaches that you can use:

1.  Save the model as a SavedModel. A concrete function will be generated during
    the saving process, which can be accessed upon loading the model.
2.  Annotate the model with `tf.function` as seen below.

```python
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x=[-1, 0, 1, 2, 3, 4], y=[-3, -1, 1, 3, 5, 7], epochs=50)

# Get the concrete function from the Keras model.
run_model = tf.function(lambda x : model(x))

# Save the concrete function.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
```
