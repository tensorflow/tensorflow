# Keras

## What's Keras?

Keras is a high-level API specification for building and training deep learning
models, suitable for fast prototyping, advanced research, and production.
It offers three key advantages:

- **User friendliness.** Keras follows best practices for reducing
    cognitive load: it offers consistent & simple interfaces,
    it minimizes the number of user actions required for common use cases,
    and it provides clear and actionable feedback upon user error.
- **Modularity and composability.** A Keras model is composed of
    fully-configurable building blocks that can be plugged together
    with as few restrictions as possible -- like Lego bricks.
- **Easy extensibility.** You can easily write your own building blocks
    (such as new layers, new loss functions, new models where you write
    the forward pass from scratch). This allows for total expressiveness,
    making Keras suitable for advanced research.


## What's tf.keras?

`tf.keras` is TensorFlow's implementation of the Keras API specification, that
serves as the TensorFlow high-level API: it's how you build models in TensorFlow.
`tf.keras` seamlessly integrates with the rest of the TensorFlow API
(such as `tf.data` input pipelines), bringing you the full power and flexibility
of TensorFlow through an easy-to-use interface.

You can import `tf.keras` via:

```python
from tensorflow import keras
```

What follows is a quick introduction to the basics of `tf.keras`.


## Table of contents

- [Getting started: the Sequential model](#getting-started-the-sequential-model)
- [Configuring layers](#configuring-layers)
- [Configuring training](#configuring-training)
- [Training and evaluation](#training-and-evaluation)
- [Building advanced models: the functional API](#building-advanced-models-the-functional-api)
- [Building fully-customizable research models: the Model subclassing API](#building-fully-customizable-research-models-the-model-subclassing-api)
- [Callbacks](#callbacks)
- [Saving and serialization](#saving-and-serialization)
- [Developing custom layers](#developing-custom-layers)
- [Eager execution](#eager-execution)
- [Further reading](#further-reading)
- [FAQ](#faq)


---

## Getting started: the Sequential model

In `tf.keras`, you're assembling together **layers** to build **models**.
A model is generally a graph of layers.
The most common type of model is just a stack of layers: the `Sequential` class.

Here's how to build a simple fully-connected network (multi-layer perceptron):

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
# This adds to the model a densely-connected layer with 64 units:
model.add(Dense(64, activation='relu'))
# Another one:
model.add(Dense(64, activation='relu'))
# This adds a softmax layer with 10 output units:
model.add(Dense(10, activation='softmax'))
```

---

## Configuring layers

Each layer may have unique constructor arguments, but some common arguments include:

- `activation`: the activation function to be used.
    It could be specified by name, as a string (for built-in functions)
    or as a callable object. By default, no activation is applied.
- `kernel_initializer` and `bias_initializer`: the initialization schemes to use
    to create the layer's weights (kernel and bias).
    Likewise, they may be passed either by name or by specifying a callable.
    By default, the "Glorot uniform" initializer is used.
- `kernel_regularizer` and `bias_regularizer`: the regularization schemes to
    apply to the layer's weights (kernel and bias), such as L1
    or L2 regularization. By default, no regularization is applied.


### Examples

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

# A sigmoid layer:
Dense(64, activation='sigmoid')
# Another way to define the same sigmoid layer:
Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01
# applied to the kernel matrix:
Dense(64, kernel_regularizer=regularizers.l1(0.01))
# A linear layer with L2 regularization of factor 0.01
# applied to the bias vector:
Dense(64, bias_regularizer=regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
Dense(64, kernel_initializer='orthogonal')
# A linear layer with a bias vector initialized to 2.0s:
Dense(64, bias_initializer=initializers.constant(2.0))
```

---

## Configuring training

Once your model looks good, configure its learning process by calling `compile`:

```python
import tensorflow as tf

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

There are three key arguments that you need to specify:

- An `optimizer`: this object specifies the training procedure.
    We recommend that you pass instances of optimizers from the `tf.train` module
    (such as [`AdamOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer),
    [`RMSPropOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer),
    or [`GradientDescentOptimizer`](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)).
- A `loss` function to minimize: this specifies the optimization objective.
    Common choices include mean square error (`mse`), `categorical_crossentropy`
    and `binary_crossentropy`. Loss functions may be specified by name
    or by passing a callable (e.g. from the `tf.keras.losses` module).
- Some `metrics` to monitor during training: again, you can pass these as either
    string names or callables (e.g. from the `tf.keras.metrics` module).


### Examples

```python
# Configures a model to do mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',  # mean squared error
              metrics=['mae'])  # mean absolute error
```
```python
# Configures a model to do categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
```

---

## Training and evaluation

### From Numpy data

When running locally on small datasets, the easiest way to do training and
evaluation is to pass data to your model as Numpy arrays of inputs and targets.
You can "fit" your model to some training data using the `model.fit()` method:

```python
import numpy as np

data = np.random.random(shape=(1000, 32))
targets = np.random.random(shape=(1000, 10))

model.fit(data, targets, epochs=10, batch_size=32)
```

Here are some key arguments you can pass to the `fit` method:

- `epochs`: Training is structured into **epochs**. An epoch is one iteration
    over the entire input data (which is done in smaller batches).
- `batch_size`: when passing Numpy data, the model will slice the data into
    smaller batches and iterate over these batches during training.
    This integer specifies the size of each batch
    (the last batch may be smaller if the total number of samples is not
    divisible by the batch size).
- `validation_data`: when prototyping a model, you want to be able to quickly
    monitor its performance on some validation data.
    When you pass this argument (it expects a tuple of inputs and targets),
    the model will display the loss and metrics in inference mode on the data
    you passed, at the end of each epoch.

Here's an example using `validation_data`:

```python
import numpy as np

data = np.random.random(shape=(1000, 32))
targets = np.random.random(shape=(1000, 10))

val_data = np.random.random(shape=(100, 32))
val_targets = np.random.random(shape=(100, 10))

model.fit(data, targets, epochs=10, batch_size=32,
          validation_data=(val_data, val_targets))
```

### From tf.data datasets

When you need to scale to large datasets or multi-device training,
training from Numpy arrays in memory will not be ideal.
In such cases, you should use [the `tf.data` API](https://www.tensorflow.org/programmers_guide/datasets).
You can pass a `tf.data.Dataset` instance to the `fit` method:

```python
import tensorflow as tf

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, targets)).batch(32)

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
```

When doing so, the dataset itself will yield batches of data,
so the model does not need to be passed `batch_size` information.
Instead, the model needs to know for how many steps (or batches of data)
it should run at each epoch.
You specify this with the `steps_per_epoch` argument: it's the number of
training steps the model will run before moving on the next epoch.

You can also pass datasets for validation:

```python
dataset = tf.data.Dataset.from_tensor_slices((data, targets)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_targets)).batch(32)

model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=val_dataset, validation_steps=3)
```

### Evaluate and predict

In addition, you get access to the following methods
(both with Numpy data and dataset instances):

- `model.evaluate(x, y, batch_size=32)` or `model.evaluate(dataset, steps=30)`
    will return the inference-mode loss and metrics for the data provided.
- `model.predict(x, y, batch_size=32)` or `model.predict(dataset, steps=30)`
    will return the output(s) of the last layer(s) in inference on the data
    provided, as Numpy array(s).

---

## Building advanced models: the functional API

The `Sequential` model cannot represent arbitrary models -- only simple stacks
of layers. If you need to use more complex model topologies,
such as multi-input models, multi-output models,
models with a same layer called several times (shared layers),
or models with non-sequential data flows (e.g. residual connections),
you can use the 'functional API'.

Here's how it works:

- A layer instance is callable (on a tensor), and it returns a tensor.
- Input tensor(s) and output tensor(s) can then be used to define a `Model` instance.
- Such a model can be trained just like the `Sequential` model.

Here's a basic example showing the same model we previously defined,
built using the functional API:


```python
from tensorflow import keras
from tensorflow.keras import layers

# This returns a placeholder tensor:
inputs = keras.Input(shape=(784,))

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

# Instantiates the model given inputs and outputs.
model = keras.Model(inputs=inputs, outputs=predictions)

# The "compile" step specifies the training configuration.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
```

This API enables you to create models with multiple inputs and outputs,
and to "share" layers across different inputs
(i.e. to reuse a same instance multiple times).
For examples of these use cases,
please see [this guide to the functional API in Keras](https://keras.io/getting-started/functional-api-guide/).

---

## Building fully-customizable research models: the Model subclassing API

Besides `Sequential` and the functional API, one last, more flexible way to
define models is to directly subclass the `Model` class and define your own
forward pass manually.

In this API, you instante layers in `__init__` and set them as attribute of the
class instance. Then you specify the forward pass in `call`.
This API is particularly valuable when using TensorFlow with [eager execution](https://www.tensorflow.org/programmers_guide/eager),
since eager execution allows you to write your forward pass in an
imperative fashion (as if you were writing Numpy code, for instance).

```python
import tensorflow as tf
from tensorflow import keras


class MyModel(keras.Model):

  def __init__(self, num_classes=2):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = keras.layers.Dense(32, activation='relu')
    self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


# Instantiates the subclassed model.
model = MyModel(num_classes=2)

# The "compile" step specifies the training configuration.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
```

**Remember:** use the right API for the right job.
Using the `Model` subclassing API offers more flexibility,
but at the cost of greater complexity and a larger potential user error surface.
Prefer using the functional API when possible.

---

## Callbacks

Callbacks are objects that you can pass to your model that customize and extend
its behavior during training.
There are callbacks for saving checkpoints of your model at regular intervals
(`tf.keras.callbacks.ModelCheckpoint`),
to dynamically change the learning rate (`tf.keras.callbacks.LearningRateScheduler`)
or to interrupt training when validation performance has stopped improving
(`tf.keras.callbacks.EarlyStopping`).
You can also use a callback to monitor your model's behavior using
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
(`tf.keras.callbacks.TensorBoard`).
You can also write your own custom callbacks.

Different built-in callback are found in `tf.keras.callbacks`.
You use them by passing a `Callback` instance to `fit`:

```python
from tensorflow import keras

callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # Write TensorBoard logs to `./logs` directory
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks)
```

---

## Saving and serialization

### Weights-only saving

You can save the weight values of a model via `model.save_weights(filepath)`:

```python
# Saves weights to a SavedModel file.
model.save_weights('my_model')

# Restores the model's state
# (this requires a model that has the same architecture).
model.load_weights('my_model')
```

By default, this saves the weight in the TensorFlow
[`SavedModel`](https://www.tensorflow.org/programmers_guide/saved_model) format.
You could also save them in the Keras HDF5 format
(which is the default in the multi-backend implementation of Keras):

```python
# Saves weights to a HDF5 file.
model.save_weights('my_model.h5', format='h5')

# Restores the model's state.
model.load_weights('my_model.h5')
```

### Configuration-only saving (serialization)

You can also save the model's configuration
(its architecture, without any weight values),
which allows you to recreate the same model later (freshly initialized) even if
you don't have the code that defined it anymore.
Two possible serialization formats are JSON and YAML:

```python
from tensorflow.keras import models

# Serializes a model to JSON.
json_string = model.to_json()
# Recreates the model (freshly initialized).
fresh_model = models.from_json(json_string)

# Serializes a model to YAML.
yaml_string = model.to_yaml()
# Recreates the model.
fresh_model = models.from_yaml(yaml_string)
```

Note that this feature is not available with subclassed models,
because they are simply not serializable:
their architecture is defined as Python code
(the body of the `call` method of the model).

### Whole-model saving

Finally, you can also save a model wholesale, to a file that will contain both
the weight values, the model's configuration,
and even the optimizer's configuration.
The allows you to checkpoint a model and resume training later --
from the exact same state -- even if you don't have access to the original code.

```python
from tensorflow.keras import models

model.save('my_model.h5')

# Recreates the exact same model, complete with weights and optimizer.
model = models.load_model('my_model.h5')
```

---

## Developing custom layers

You can write your own custom layers by subclassing the class
`tf.keras.layers.Layer`. You will need to implement the following three methods:

- `build`: Creates the weights of the layer.
    Weights should be added via the `add_weight` method.
- `call`: Specifies the forward pass.
- `compute_output_shape`: Specifies how to compute the output shape of the layer 
    given the input shape.

Optionally, you may also implement the method `get_config()` and the
class method `from_config()` if you want your layer to be serializable.

Here's a simple example of a custom layer that implements a `matmul`
of an input with a kernel matrix:

```python
import tensorflow as tf
from tensorflow.keras import layers

class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim

    @classmethod
    def from_config(cls, config):
        return cls(**config)
```

---

## Eager execution

[Eager execution](https://www.tensorflow.org/programmers_guide/eager)
is a way to write TensorFlow code imperatively.

All three `tf.keras` model-building APIs
(`Sequential`, the functional API `Model(inputs, outputs)`,
and the subclassing API `MyModel(Model)`) are compatible with eager execution.
When using `Sequential` or the functional API, it makes no difference to the
user experience whether the model is executing eagerly or not.
Eager execution is most beneficial when used with the `Model` subclassing API,
or when prototyping a custom layer -- that is to say, in APIs that require you
to *write a forward pass as code*, rather than in APIs that allow you to create
models by assembling together existing layers.

While the same training and evaluating APIs presented in this guide work
as usual with eager execution, you can in addition
write custom training loops using the eager `GradientTape`
and define-by-run autodifferentiation:

```python
import tensorflow as tf
from tensorflow.contrib import eager as tfe

# This call begins the eager execution session.
tf.enable_eager_execution()

model = ...  # Defines a Keras model (we recommend Model subclassing in this case).
dataset = ...  # Defines a `tf.data` dataset.

optimizer = tf.train.AdamOptimizer(0.01)

for data, labels in dataset:
    # Runs the forward pass and loss computation under a `GradientTape` scope,
    # which will record all operations in order to prepare for the backward pass.
    with tfe.GradientTape() as tape:
      predictions = model(data)
      loss = loss_function(labels, predictions)

    # Runs the backward pass manually using the operations recorded
    # by the gradient tape.
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights),
                              global_step=tf.train.get_or_create_global_step())
```

---

## Further reading

### Documentation

- [tf.keras documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [keras.io](https://keras.io/)

### tf.keras tutorials and examples

- [Fashion-MNIST with tf.Keras](https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a)
- [Predicting the price of wine with the Keras Functional API and TensorFlow](
    https://medium.com/tensorflow/predicting-the-price-of-wine-with-the-keras-functional-api-and-tensorflow-a95d1c2c1b03)


---

## FAQ

### What are the differences between tf.keras and the multi-backend Keras implementation?

`tf.keras` includes first-class support for important TensorFlow-specific
functionality not found in other Keras implementations, in particular:

- Support for eager execution.
- Support for the `tf.data` API.
- Integration with the
    [`tf.estimator` API](https://www.tensorflow.org/programmers_guide/estimators),
    via `tf.keras.estimator.model_to_estimator`.

In terms of API differences: `tf.keras` is a full implementation of the
Keras API, so any code targeting the Keras API will run on `tf.keras`.
However, keep in mind that:

- The `tf.keras` API version in the latest TensorFlow release might not be the
    same as the latest `keras` version from PyPI.
    Check out `tf.keras.__version__` if in doubt.
- In `tf.keras`, the default file format saved by `model.save_weights` is the
    TensorFlow `SavedModel` format.
    To use HDF5, you can pass the `format='h5'` argument.


### What is the relationship between tf.keras and tf.estimator?

The [`tf.estimator` API](https://www.tensorflow.org/programmers_guide/estimators)
is a high-level TensorFlow API for training "estimator" models,
in particular in distributed settings.
This API targets industry use cases, such as distributed training
on large datasets with a focus on eventually exporting a production model.

If you have a `tf.keras` model that would like to train with the `tf.estimator`
API, you can convert your model to an `Estimator` object via the
`model_to_estimator` utility](https://www.tensorflow.org/programmers_guide/estimators#creating_estimators_from_keras_models):


```python
estimator = tf.keras.estimator.model_to_estimator(model)
```

When using `model_to_estimator`, enabling eager execution is helpful for
developing and debugging your `input_fn`
(as it allows you to easily print your data).


### How can I run tf.keras models on multiple GPUs?

You can run tf.keras models on multiple GPUs using the
[`DistributionStrategy API`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/DistributionStrategy).
The `DistributionStrategy` API allow you to distribute training on multiple GPUs
with almost no changes to your existing code.

Currently [`MirroredStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/MirroredStrategy)
is the only supported strategy.
`MirroredStrategy` allows you to do in-graph replication with synchronous
training using all-reduce on a single machine.
To use `DistributionStrategy` with a `tf.keras` model,
you can use the `model_to_estimator` utility to convert a `tf.keras` model to
an `Estimator` and then train the estimator.

Here is a simple example of distributing a `tf.keras` model across multiple GPUs
on a single machine.

Let's first define a simple model:

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
optimizer = tf.train.GradientDescentOptimizer(0.2)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()
```

Let's use `model_to_estimator` to create an `Estimator` instance from the
`tf.keras` model defined above.

```python
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model,
    config=config,
    model_dir='/tmp/model_dir')
```

We'll use `tf.data.Datasets` to define our input pipeline.
Our `input_fn` returns a `tf.data.Dataset` object that we then use to distribute
the data across multiple devices with each device processing
a slice of the input batch.

```python
def input_fn():
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(10)
    dataset = dataset.batch(32)
    return dataset
```

The next step is to create a `RunConfig` and set the train_distribute argument
to the new `MirroredStrategy` instance.
You can specify a list of devices or the `num_gpus` argument when creating
a `MirroredStrategy` instance.
Not specifying any arguments defaults to using all the available GPUs like we do
in this example.

```python
strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
```

Call train on the `Estimator` instance providing the `input_fn` and `steps`
arguments as input:

```python
keras_estimator.train(input_fn=input_fn, steps=10)
```
