# Using the Estimator API with TPUs


This document describes how to train a TensorFlow model on TPUs using the
Estimator API. If you are interested in the hardware itself, check out the
[Cloud TPU documentation](https://cloud.google.com/tpu/docs).

The TPU Estimator simplifies running models on a Cloud TPU by automatically
handling numerous low-level hardware-specific details

[TOC]

## Introduction to Estimator

[TensorFlow
tutorials](https://www.tensorflow.org/extend/estimators) cover the Estimator
API. At a high-level, the Estimator API provides:

*   `Estimator.train()` - train a model on a given input for a fixed number of
    steps.
*   `Estimator.evaluate()` - evaluate the model on a test set.
*   `Estimator.predict()` - run inference using the trained model.
*   `Estimator.export_savedmodel()` - export your model for serving.

In addition, `Estimator` includes default behavior common to training jobs,
such as saving and restoring checkpoints, creating summaries for TensorBoard,
etc.

`Estimator` requires you to write a `model_fn` and an `input_fn`, which
correspond to the model and input portions of your TensorFlow graph.

The following code demonstrates using `TPUEstimator` with MNIST example to
handle training:

    def model_fn(features, labels, mode, params):
      """A simple CNN."""
      del params  # unused

      input_layer = tf.reshape(features, [-1, 28, 28, 1])
      conv1 = tf.layers.conv2d(
          inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same",
          activation=tf.nn.relu)
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
      conv2 = tf.layers.conv2d(
          inputs=pool1, filters=64, kernel_size=[5, 5],
          padding="same", activation=tf.nn.relu)
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
      pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
      dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
      logits = tf.layers.dense(inputs=dropout, units=10)
      onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)

      learning_rate = tf.train.exponential_decay(
          FLAGS.learning_rate, tf.train.get_global_step(), 100000, 0.96)

      optimizer = tpu_optimizer.CrossShardOptimizer(
          tf.train.GradientDescentOptimizer(learning_rate=learning_rate))

      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tpu_estimator.TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    def get_input_fn(filename):
      """Returns an `input_fn` for train and eval."""

      def input_fn(params):
        """An input_fn to parse 28x28 images from filename using tf.data."""
        batch_size = params["batch_size"]

        def parser(serialized_example):
          """Parses a single tf.Example into image and label tensors."""
          features = tf.parse_single_example(
              serialized_example,
              features={
                  "image_raw": tf.FixedLenFeature([], tf.string),
                  "label": tf.FixedLenFeature([], tf.int64),
              })
          image = tf.decode_raw(features["image_raw"], tf.uint8)
          image.set_shape([28 * 28])
          # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
          image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
          label = tf.cast(features["label"], tf.int32)
          return image, label

        dataset = tf.contrib.data.TFRecordDataset(
            filename, buffer_size=FLAGS.dataset_reader_buffer_size)
        dataset = dataset.map(parser).cache().repeat().batch(batch_size)
        images, labels = dataset.make_one_shot_iterator().get_next()
        # set_shape to give inputs statically known shapes.
        images.set_shape([batch_size, 28 * 28])
        labels.set_shape([batch_size])
        return images, labels
      return input_fn


    def main(unused_argv):

      tf.logging.set_verbosity(tf.logging.INFO)

      run_config = tpu_config.RunConfig(
          master=FLAGS.master,
          model_dir=FLAGS.model_dir,
          session_config=tf.ConfigProto(
              allow_soft_placement=True, log_device_placement=True),
          tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards),)

      estimator = tpu_estimator.TPUEstimator(
          model_fn=model_fn,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.batch_size,
          eval_batch_size=FLAGS.batch_size,
          config=run_config)

      estimator.train(input_fn=get_input_fn(FLAGS.train_file),
                      max_steps=FLAGS.train_steps)


Although this code is quite simple by appearance, there are some new
concepts to learn for using `TPU`s. The next section will cover the most
important details.

## New Concepts Related to TPU/TPUEstimator

TF programs run with `TPU Estimator` use an [in-graph
replication](https://www.tensorflow.org/deploy/distributed) approach.

In-graph replication (also known as single-session replication) differs from
the between-graph replication (also known as multi-session replication)
training typically used in distributed TensorFlow. The major
differences include:

1. The TensorFlow Session master is not local anymore. The user python program
   creates one single graph that is replicated across all the cores in the Cloud
   TPU. The typical configuration today sets the TensorFlow session master to be
   the first worker.

1. The input pipeline is placed on remote hosts (instead of local) to ensure the
   training examples can be fed as fast as possible to TPU system. All queue-based 
   input pipelines do not work effectively. Dataset (tf.data) is
   required.

1. Workers in the TPU system operate in synchronous fashion, and each perform
   the same step at the same time.

Regarding programming model, _"The programmer picks a (large) batch size B and
writes the program (and sets hyperparameters) based on that batch size. The
system distributes the computation across the available devices."

To align these, `TPUEstimator` wraps the computation (the `model_fn`) and
distributes it to all available TPU chips. 

To summarize:

- The `input_fn` models the input pipeline running on remote host CPU. Use
  `tf.data` to program the input Ops. `input_fn` is expected to be invoked
  multiple times when using TPU pods. Each handles one device's input of the
  global batch. The shard batch size should be retrieved from
  `params['batch_size']`. We plan to provide better abstraction about the
  sharding mechanism for `tf.data` to remove the `params['batch_size']`.

- The `model_fn` models the computation which will be replicated and distributed
  to all TPU chips. It should only contains ops that are supported by TPUs.

## Convert from Vanilla Estimator to TPUEstimator

It is always recommended to port a small, simple model first to make sure that
you are familiar with the basic concepts of `TPUEstimator` and test end-to-end
behavior. Once your simple model runs, gradually add more functionality.
In addition, there are several sample models, available at
[github.com/tensorflow/tpu-demos](https://github.com/tensorflow/tpu-demos).

To convert your code from the vanilla `Estimator` class to use TPUs, change the
following (note some of the details may change over time):

- Switch from `tf.estimator.RunConfig` to `tf.contrib.tpu.RunConfig`.
- Set the `TPUConfig` (part of the `tf.contrib.tpu.RunConfig`) to specify the
  `iterations_per_loop`, number of iterations to run on the TPU device for one
  `session.run` call (per training loop), and `num_shards`, the number of shards
  (typically the number of TPU cores you’re running on). TPUs run a number of
  iterations of the training loop before returning to host. Until all iterations
  on the TPU device are run, no checkpoints or summaries will be saved. In the
  future, we’ll choose a reasonable default.
- In `model_fn`, use `tf.contrib.tpu.CrossShardOptimizer` to wrap your
  optimizer. Example:

         optimizer = tpu_optimizer.CrossShardOptimizer(
              tf.train.GradientDescentOptimizer(learning_rate=learning_rate))

- Switch from `tf.estimator.Estimator` to `tf.contrib.tpu.TPUEstimator`.

The default `RunConfig` will save summaries for TensorBoard every 100 steps and
write checkpoints every 10 minutes.


## FAQ

### Why `tf.data` is Required for the Input Pipeline

There are two reasons:

1. The user code runs on the client, while the TPU computation is executed on
   the `worker`. Input pipeline ops must be placed on the remote worker for
   good performance. Only `tf.data` (Dataset) supports this.

1. In order to amortize the TPU launch cost, the model train step is wrapped in
   a `tf.while_loop`, such that one `Session.run` actually runs many iterations
   for one train loop.  To remove network back and forth, the input pipeline
   in the future will be wrapped in a `tf.while_loop` and be placed on the
   corresponding `worker`. Withou this, unnecessary network latency becomes
   the performance bottleneck for models with short training-step times, or in
   environments where network latency is higher. Only `tf.data` can be wrapped
   by a `tf.while_loop`.


### How to add other CPU Ops into Graph
As `model_fn` only allows TPU Ops for computation, the easier workaround to add
CPU Ops into Graph is:

1. Create a [SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook).
1. Modify the graph in the `def begin(self)`,
1. Pass the hook to `TPUEstimator.train`.

### Running On GCP Cloud TPUs
To run your models on GCP Cloud TPUs refer to the [Cloud Documentation](https://cloud.google.com/tpu/docs/tutorials/mnist).
Refer to this link for all [Cloud TPU documentation](https://cloud.google.com/tpu/docs).


### Profiling
You can profile the `worker` by using instructions as spcified in the [Cloud TPU Tools](https://cloud.google.com/tpu/docs/cloud-tpu-tools).


### Is `int64` supported?
`int64` is not supported by TPU. Cast to int32 if applicable. The only exception
is global step, which relies on `assign_add`. `int64` support for global step
is added to ensure checkpoint compatibility between `TPUEstimator` and non-TPU
`Estimator`.
