# Distribution Strategy

> *NOTE*: This is a experimental feature. The API and performance
> characteristics are subject to change.

## Overview

[`DistributionStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/DistributionStrategy)
API is an easy way to distribute your training
across multiple devices/machines. Our goal is to allow users to use existing
models and training code with minimal changes to enable distributed training.
Moreover, we've design the API in such a way that it works with both eager and
graph execution.

Currently we support one type of strategy, called
[`MirroredStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/MirroredStrategy).
It does in-graph replication with synchronous training
on many GPUs on one machine. Essentially, we create copies of all variables in
the model's layers on each device. We then use all-reduce to combine gradients
across the devices before applying them to the variables to keep them in sync.
In the future, we intend to support other kinds of training configurations such
as multi-node, synchronous,
[asynchronous](https://www.tensorflow.org/deploy/distributed#putting_it_all_together_example_trainer_program),
parameter servers and model parallelism.

## Example

Let's demonstrate how to use this API with a simple example. We will use the
[`Estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
approach, and show you how to scale your model to run on multiple GPUs on one
machine using `MirroredStrategy`.

Let's consider a very simple model function which tries to learn a simple
function.

```python
def model_fn(features, labels, mode):
  layer = tf.layers.Dense(1)
  logits = layer(features)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {"logits": logits}
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  loss = tf.losses.mean_squared_error(
      labels=labels, predictions=tf.reshape(logits, []))

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss_fn())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

Let's also define a simple input function to feed data for training this model.
Note that we require using
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
with `DistributionStrategy`.


```python
def input_fn():
  features = tf.data.Dataset.from_tensors([[1.]]).repeat(100)
  labels = tf.data.Dataset.from_tensors(1.).repeat(100)
  return dataset_ops.Dataset.zip((features, labels))
```

Now that we have a model function and input function defined, we can define the
estimator. To use `MirroredStrategy`, all we need to do is:

* Create an instance of the `MirroredStrategy` class.
* Pass it to the
[`RunConfig`](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig)
parameter of `Estimator`.


```python
distribution = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=distribution)
classifier = tf.estimator.Estimator(model_fn=model_fn, config=config)
classifier.train(input_fn=input_fn)
```

That's it! This change will now configure estimator to run on all GPUs on your
machine, with the `MirroredStrategy` approach. It will take care of distributing
the input dataset, replicating layers and variables on each device, and
combining and applying gradients.

The model and input functions do not have to change because we have changed the
underlying components of TensorFlow (such as
optimizer, batch norm and summaries) to become distribution-aware.
That means those components know how to
combine their state across devices. Further, saving and checkpointing works
seamlessly, so you can save with one or no distribution strategy and resume with
another.

Above, we showed the easiest way to use [`MirroredStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/MirroredStrategy#__init__).
There are few things you can customize in practice:

* You can specify a list of specific GPUs (using param `devices`) or the number
of GPUs (using param `num_gpus`), in case you don't want auto detection.
* You can specify various parameters for all reduce with the `cross_tower_ops`
param, such as the all reduce algorithm to use, and gradient repacking.

## Performance Tips

We've tried to make it such that you get the best performance for your existing
model. We also recommend you follow the tips from
[Input Pipeline Performance Guide](https://www.tensorflow.org/performance/datasets_performance).
Specifically, we found using [`map_and_batch`](https://www.tensorflow.org/performance/datasets_performance#map_and_batch)
and [`dataset.prefetch`](https://www.tensorflow.org/performance/datasets_performance#pipelining)
in the input function gives a solid boost in performance. When using
`dataset.prefetch`, use `buffer_size=None` to let it detect optimal buffer size.

## Caveats
This feature is in early stages and there are a lot of improvements forthcoming:

* Metrics are not yet supported during distributed training. They are still
supported during the evaluation.
* Summaries are only computed in the first tower in `MirroredStrategy`.
* Evaluation is not yet distributed.
* Eager support is in the works; performance can be more challenging with eager
execution.
* As mentioned earlier, multi-node and other distributed strategies will be
introduced in the future.
* If you are [`batching`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch)
your input data, we will place one batch on each GPU in each step. So your
effective batch size will be `num_gpus * batch_size`. Therefore, consider
adjusting your learning rate or batch size according to the number of GPUs.
We are working on addressing this limitation by splitting each batch across GPUs
instead.
* PartitionedVariables are not supported yet.

## What's next?

Please give distribution strategies a try. This feature is in early stages and
is evolving, so we welcome your feedback via
[issues on GitHub](https://github.com/tensorflow/tensorflow/issues/new).


