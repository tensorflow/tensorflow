# Tensorflow Distribute Libraries

## Overview

tf.distribute.Strategy is a TensorFlow API to distribute training across
multiple GPUs, multiple machines or TPUs. Using this API, users can distribute
their existing models and training code with minimal code changes.

It can be used with TensorFlow's high level APIs, tf.keras and tf.estimator,
with just a couple of lines of code change. It does so by changing the
underlying components of TensorFlow to become strategy-aware.
This includes variables, layers, models, optimizers, metrics, summaries,
and checkpoints.

## Documentation

[Distributed Training Guide](https://www.tensorflow.org/guide/distributed_training)

[Distributed Training With Keras Tutorial](https://www.tensorflow.org/tutorials/distribute/keras)

[Distributed Training With Custom Training Loops Tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training)

[Multiworker Training With Keras Tutorial](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)

[Multiworker Training With Estimator Tutorial](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator)

[Save and Load with Distribution Strategy](https://www.tensorflow.org/tutorials/distribute/save_and_load)

## Simple Examples

### Using compile fit with GPUs.

```python
# Create the strategy instance. It will automatically detect all the GPUs.
mirrored_strategy = tf.distribute.MirroredStrategy()

# Create and compile the keras model under strategy.scope()
with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  model.compile(loss='mse', optimizer='sgd')

# Call model.fit and model.evaluate as before.
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)
```

### Custom training loop with TPUs.

```python
# Create the strategy instance.
tpu_strategy = tf.distribute.TPUStrategy(resolver)


# Create the keras model under strategy.scope()
with tpu_strategy.scope():
  model = keras.layers.Dense(1, name="dense")

# Create custom training loop body as tf.function.
@tf.function
def train_step(iterator):
  def step_fn(inputs):
    images, targets = inputs
    with tf.GradientTape() as tape:
      outputs = model(images)
      loss = tf.reduce_sum(outputs - targets)
    grads = tape.gradient(loss, model.variables)
    return grads

  return tpu_strategy.run(
      step_fn, args=(next(iterator),))

# Run the loop body once on at dataset.
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10
input_iterator = iter(tpu_strategy.experimental_distribute_dataset(dataset))
train_step(input_iterator)
```

## Testing

Tests here should cover all distribution strategies to ensure feature parity.
This can be done using the test decorators in `strategy_combinations.py`.

