# Distribution Strategy

> *NOTE*: This is an experimental feature. The API and performance
> characteristics are subject to change.

## Overview

[`DistributionStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/DistributionStrategy)
API is an easy way to distribute your training
across multiple devices/machines. Our goal is to allow users to use existing
models and training code with minimal changes to enable distributed training.
Moreover, we've designed the API in such a way that it works with both eager and
graph execution.

Currently we support several types of strategies:

* [`MirroredStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/MirroredStrategy):
This does in-graph replication with synchronous training
on many GPUs on one machine. Essentially, we create copies of all variables in
the model's layers on each device. We then use all-reduce to combine gradients
across the devices before applying them to the variables to keep them in sync.
* [`CollectiveAllReduceStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/CollectiveAllReduceStrategy):
This is a version of `MirroredStrategy` for multi-worker training. It uses
a collective op to do all-reduce. This supports between-graph communication and
synchronization, and delegates the specifics of the all-reduce implementation to
the runtime (as opposed to encoding it in the graph). This allows it to perform
optimizations like batching and switch between plugins that support different
hardware or algorithms. In the future, this strategy will implement
fault-tolerance to allow training to continue when there is worker failure.

* [`ParameterServerStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/ParameterServerStrategy):
This strategy supports using parameter servers either for multi-GPU local
training or asynchronous multi-machine training. When used to train locally,
variables are not mirrored, instead they are placed on the CPU and operations
are replicated across all local GPUs. In a multi-machine setting, some are
designated as workers and some as parameter servers. Each variable is placed on
one parameter server. Computation operations are replicated across all GPUs of
the workers.

## Multi-GPU Training

## Example with Keras API

Let's see how to scale to multiple GPUs on one machine using `MirroredStrategy` with [tf.keras] (https://www.tensorflow.org/guide/keras).

Let's define a simple input dataset for training this model. Note that currently we require using
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
with `DistributionStrategy`.

```python
import tensorflow as tf
from tensorflow import keras

features = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
labels = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
train_dataset = tf.data.Dataset.zip((features, labels))
```

To distribute this Keras model on multiple GPUs using `MirroredStrategy` we
first instantiate a `MirroredStrategy` object.

```python
distribution = tf.contrib.distribute.MirroredStrategy()
```

Take a very simple model consisting of a single layer. We need to create and compile
the model under the distribution strategy scope.

```python
with distribution.scope():
  inputs = tf.keras.layers.Input(shape=(1,))
  predictions = tf.keras.layers.Dense(1)(inputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

  model.compile(loss='mean_squared_error',
                optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.2))
```

To train the model we call Keras `fit` API using the input dataset that we
created earlier, same as how we would in a non-distributed case.

```python
model.fit(train_dataset, epochs=5, steps_per_epoch=10)
```

Similarly, we can also call `evaluate` and `predict` as before using appropriate
datasets.

```python
model.evaluate(eval_dataset, steps=1)
model.predict(predict_dataset, steps=1)
```

That's all you need to train your model with Keras on multiple GPUs with
`MirroredStrategy`. It will take care of splitting up
the input dataset, replicating layers and variables on each device, and
combining and applying gradients.

The model and input code does not have to change because we have changed the
underlying components of TensorFlow (such as
optimizer, batch norm and summaries) to become distribution-aware.
That means those components know how to
combine their state across devices. Further, saving and checkpointing works
seamlessly, so you can save with one or no distribution strategy and resume with
another.


## Example with Estimator API

You can also use Distribution Strategy API with [`Estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator). Let's see a simple example of it's usage with `MirroredStrategy`.


Consider a very simple model function which tries to learn a simple function.

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
    train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

Again, let's define a simple input function to feed data for training this model.


```python
def input_fn():
  features = tf.data.Dataset.from_tensors([[1.]]).repeat(100)
  labels = tf.data.Dataset.from_tensors(1.).repeat(100)
  return tf.data.Dataset.zip((features, labels))
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
classifier.evaluate(input_fn=input_fn)
```

That's it! This change will now configure estimator to run on all GPUs on your
machine.


## Customization and Performance Tips

Above, we showed the easiest way to use [`MirroredStrategy`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/distribute/MirroredStrategy#__init__).
There are few things you can customize in practice:

* You can specify a list of specific GPUs (using param `devices`) or the number
of GPUs (using param `num_gpus`), in case you don't want auto detection.
* You can specify various parameters for all reduce with the `cross_tower_ops`
param, such as the all reduce algorithm to use, and gradient repacking.

We've tried to make it such that you get the best performance for your existing
model. We also recommend you follow the tips from
[Input Pipeline Performance Guide](https://www.tensorflow.org/performance/datasets_performance).
Specifically, we found using [`map_and_batch`](https://www.tensorflow.org/performance/datasets_performance#map_and_batch)
and [`dataset.prefetch`](https://www.tensorflow.org/performance/datasets_performance#pipelining)
in the input function gives a solid boost in performance. When using
`dataset.prefetch`, use `buffer_size=None` to let it detect optimal buffer size.

## Multi-worker Training
### Overview

For multi-worker training, no code change is required to the `Estimator` code.
You can run the same model code for all tasks in your cluster including
parameter servers and the evaluator. But you need to use
`tf.estimator.train_and_evaluate`, explicitly specify `num_gpus_per_workers`
for your strategy object, and set "TF\_CONFIG" environment variables for each
binary running in your cluster. We'll provide a Kubernetes template in the
[tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) repo which sets
"TF\_CONFIG" for your training tasks.

### TF\_CONFIG environment variable

The "TF\_CONFIG" environment variables is a JSON string which specifies what
tasks constitute a cluster, their addresses and each task's role in the cluster.
One example of "TF\_CONFIG" is:

```python
TF_CONFIG='{
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
}'
```

This "TF\_CONFIG" specifies that there are three workers and two ps tasks in the
cluster along with their hosts and ports. The "task" part specifies that the
role of the current task in the cluster, worker 1. Valid roles in a cluster is
"chief", "worker", "ps" and "evaluator". There should be no "ps" job for
`CollectiveAllReduceStrategy` and `MirroredStrategy`. The "evaluator" job is
optional and can have at most one task. It does single machine evaluation and if
you don't want to do evaluation, you can pass in a dummy `input_fn` to the
`tf.estimator.EvalSpec` of `tf.estimator.train_and_evaluate`.

### Dataset

The `input_fn` you provide to estimator code is for one worker. So remember to
scale up your batch if you have multiple GPUs on each worker.

The same `input_fn` will be used for all workers if you use
`CollectiveAllReduceStrategy` and `ParameterServerStrategy`. Therefore it is
important to shuffle your dataset in your `input_fn`.

`MirroredStrategy` will insert a `tf.dataset.Dataset.shard` call in you
`input_fn` if `auto_shard_dataset` is set to `True`. As a result, each worker
gets a fraction of your input data.

### Performance Tips

We have been actively working on multi-worker performance. Currently, prefer
`CollectiveAllReduceStrategy` for synchronous multi-worker training.

### Example

Let's use the same example for multi-worker. We'll start a cluster with 3
workers doing synchronous all-reduce training. In the following code snippet, we
start multi-worker training using `tf.estimator.train_and_evaluate`:

```python
def model_main():
  distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
      num_gpus_per_worker=2)
  config = tf.estimator.RunConfig(train_distribute=distribution)
  estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
  train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

**Note**: You don't have to set "TF\_CONFIG" manually if you use our provided
Kubernetes template.

You'll then need 3 machines, find out their host addresses and one available
port on each machine. Then set  "TF\_CONFIG" in each binary and run the above
model code.

In your worker 0, run:

```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"]
    },
   "task": {"type": "worker", "index": 0}
})

# Call the model_main function defined above.
model_main()
```

In your worker 1, run:

```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"]
    },
   "task": {"type": "worker", "index": 1}
})

# Call the model_main function defined above.
model_main()
```

In your worker 2, run:

```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"]
    },
   "task": {"type": "worker", "index": 2}
})

# Call the model_main function defined above.
model_main()
```

Then you'll find your cluster has started training! You can inspect the logs of
workers or start a tensorboard.

### Standalone client mode

We have a new way to run distributed training. You can bring up standard
tensorflow servers in your cluster and run your model code anywhere such as on
your laptop.

In the above example, instead of calling `model_main`, you can call
`tf.contrib.distribute.run_standard_tensorflow_server().join()`. This will bring
up a cluster running standard tensorflow servers which wait for your request to
start training.

On your laptop, you can run

```python
distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
    num_gpus_per_worker=2)
config = tf.estimator.RunConfig(
    experimental_distribute=tf.contrib.distribute.DistributeConfig(
        train_distribute=distribution,
        remote_cluster={"worker": ["host1:port", "host2:port", "host3:port"]}))
estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

Then you will see the training logs on your laptop. You can terminate the
training by terminating your process on your laptop. You can also modify your
code and run a new model against the same cluster.

We've been optimizing the performance of standalone client mode. If you notice
high latency between your laptop and your cluster, you can reduce that latency
by running your model binary in the cluster.

## Caveats

This feature is in early stages and there are a lot of improvements forthcoming:

* Summaries are only computed in the first tower in `MirroredStrategy`.
* Eager support is in the works; performance can be more challenging with eager
execution.
* We currently support the following predefined Keras callbacks:
`ModelCheckpointCallback`, `TensorBoardCallback`. We will soon be adding support for
some of the other callbacks such as `EarlyStopping`, `ReduceLROnPlateau`, etc. If you
create your own callback, you will not have access to all model properties and
validation data.
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


