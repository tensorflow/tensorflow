# Using TPUs

This document walks through the principal TensorFlow APIs necessary to make
effective use of a [Cloud TPU](https://cloud.google.com/tpu/), and highlights
the differences between regular TensorFlow usage, and usage on a TPU.

This doc is aimed at users who:

* Are familiar with TensorFlow's `Estimator` and `Dataset` APIs
* Have maybe [tried out a Cloud TPU](https://cloud.google.com/tpu/docs/quickstart)
  using an existing model.
* Have, perhaps, skimmed the code of an example TPU model
  [[1]](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py)
  [[2]](https://github.com/tensorflow/tpu/tree/master/models).
* Are interested in porting an existing `Estimator` model to
  run on Cloud TPUs

## TPUEstimator

@{tf.estimator.Estimator$Estimators} are TensorFlow's model-level abstraction.
Standard `Estimators` can drive models on CPU and GPUs. You must use
@{tf.contrib.tpu.TPUEstimator} to drive a model on TPUs.

Refer to TensorFlow's Getting Started section for an introduction to the basics
of using a @{$premade_estimators$pre-made `Estimator`}, and
@{$custom_estimators$custom `Estimator`s}.

The `TPUEstimator` class differs somewhat from the `Estimator` class.

The simplest way to maintain a model that can be run both on CPU/GPU or on a
Cloud TPU is to define the model's inference phase (from inputs to predictions)
outside of the `model_fn`. Then maintain separate implementations of the
`Estimator` setup and `model_fn`, both wrapping this inference step. For an
example of this pattern compare the `mnist.py` and `mnist_tpu.py` implementation in
[tensorflow/models](https://github.com/tensorflow/models/tree/master/official/mnist).

### Running a `TPUEstimator` locally

To create a standard `Estimator` you call the constructor, and pass it a
`model_fn`, for example:

```
my_estimator = tf.estimator.Estimator(
  model_fn=my_model_fn)
```

The changes required to use a @{tf.contrib.tpu.TPUEstimator} on your local
machine are relatively minor. The constructor requires two additional arguments.
You should set the `use_tpu` argument to `False`, and pass a
@{tf.contrib.tpu.RunConfig} as the `config` argument, as shown below:

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config=tf.contrib.tpu.RunConfig()
    use_tpu=False)
```

Just this simple change will allow you to run a `TPUEstimator` locally.
The majority of example TPU models can be run in this local mode,
by setting the command line flags as follows:


```
$> python mnist_tpu.py --use_tpu=false --master=''
```

Note: This `use_tpu=False` argument is useful for trying out the `TPUEstimator`
API. It is not meant to be a complete TPU compatibility test. Successfully
running a model locally in a `TPUEstimator` does not guarantee that it will
work on a TPU.


### Building a `tpu.RunConfig`

While the default `RunConfig` is sufficient  for local training, these settings
cannot be ignored in real usage.

A more typical setup for a `RunConfig`, that can be switched to use a Cloud
TPU, might be as follows:

``` python
import tempfile
import subprocess

class FLAGS(object):
  use_tpu=False
  tpu_name=None
  # Use a local temporary path for the `model_dir`
  model_dir = tempfile.mkdtemp()
  # Number of training steps to run on the Cloud TPU before returning control.
  iterations = 50
  # A single Cloud TPU has 8 shards.
  num_shards = 8

if FLAGS.use_tpu:
    my_project_name = subprocess.check_output([
        'gcloud','config','get-value','project'])
    my_zone = subprocess.check_output([
        'gcloud','config','get-value','compute/zone'])
    cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=my_zone,
            project=my_project)
    master = tpu_cluster_resolver.get_master()
else:
    master = ''

my_tpu_run_config = tf.contrib.tpu.RunConfig(
    master=master,
    evaluation_master=master,
    model_dir=FLAGS.model_dir,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations,
                                        FLAGS.num_shards),
)
```

Then you must pass the @{tf.contrib.tpu.RunConfig} to the constructor:

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config = my_tpu_run_config,
    use_tpu=FLAGS.use_tpu)
```

Typically the `FLAGS` would be set by command line arguments. To switch from
training locally to training on a cloud TPU you would need to:

* Set `FLAGS.use_tpu` to `True`
* Set `FLAGS.tpu_name` so the `tf.contrib.cluster_resolver.TPUClusterResolver` can find it
* Set `FLAGS.model_dir` to a Google Cloud Storage bucket url (`gs://`).


## Optimizer

When training on a cloud TPU you **must** wrap the optimizer in a
@{tf.contrib.tpu.CrossShardOptimizer}, which uses an `allreduce` to aggregate
gradients and broadcast the result to each shard (each TPU core).

The `CrossShardOptimizer` is not compatible with local training. So, to have
the same code run both locally and on a Cloud TPU, add lines like the following:

``` python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
if FLAGS.use_tpu:
  optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
```

If you prefer to avoid a global `FLAGS` variable in your model code, one
approach is to set the optimizer as one of the `Estimator`'s params,
as follows:

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config = my_tpu_run_config,
    use_tpu=FLAGS.use_tpu,
    params={'optimizer':optimizer})
```

## Model Function

This section details the changes you must make to the model function
(`model_fn()`) to make it `TPUEstimator` compatible.

### Static shapes

During regular usage TensorFlow attempts to determine the shapes of each
`tf.Tensor` during graph construction. During execution any unknown shape
dimensions are determined dynamically,
see @{$programmers_guide/tensors#shape$Tensor Shapes} for more details.

To run on Cloud TPUs TensorFlow models are compiled using @{$xla$XLA}.
XLA uses a similar system for determining shapes at compile time. XLA requires
that all tensor dimensions be statically defined at compile time. All shapes
must evaluate to a constant, and not depend on external data, or stateful
operations like variables or a random number generator.


### Summaries

Remove any use of `tf.summary` from your model.

@{$summaries_and_tensorboard$TensorBoard summaries} are a great way see inside
your model. A minimal set of basic summaries are automatically recorded by the
`TPUEstimator`, to `event` files in the `model_dir`. Custom summaries, however,
are currently unsupported when training on a Cloud TPU. So while the
`TPUEstimator` will still run locally with summaries, it will fail if used on a
TPU.

### Metrics

Build your evaluation metrics dictionary in a stand-alone `metric_fn`.

<!-- TODO(markdaoust) link to programmers_guide/metrics when it exists -->

Evaluation metrics are an essential part of training a model. These are fully
supported on Cloud TPUs, but with a slightly different syntax.

A standard @{tf.metrics} returns two tensors. The first returns the running
average of the metric value, while the second updates the running average and
returns the value for this batch:

```
running_average, current_batch = tf.metrics.accuracy(labels, predictions)
```

In a standard `Estimator` you create a dictionary of these pairs, and return it
as part of the `EstimatorSpec`.

```python
my_metrics = {'accuracy': tf.metrics.accuracy(labels, predictions)}

return tf.estimator.EstimatorSpec(
  ...
  eval_metric_ops=my_metrics
)
```

In a `TPUEstimator` you instead pass a function (which returns a metrics
dictionary) and a list of argument tensors, as shown below:

```python
def my_metric_fn(labels, predictions):
   return {'accuracy': tf.metrics.accuracy(labels, predictions)}

return tf.contrib.tpu.TPUEstimatorSpec(
  ...
  eval_metrics=(my_metric_fn, [labels, predictions])
)
```

### Use `TPUEstimatorSpec`

`TPUEstimatorSpec` do not support hooks, and require function wrappers for
some fields.

An `Estimator`'s `model_fn` must return an `EstimatorSpec`. An `EstimatorSpec`
is a simple structure of named fields containing all the `tf.Tensors` of the
model that the `Estimator` may need to interact with.

`TPUEstimators` use a @{tf.contrib.tpu.TPUEstimatorSpec}. There are a few
differences between it and a standard @{tf.estimator.EstimatorSpec}:


*  The `eval_metric_ops` must be wrapped into a `metrics_fn`, this field is
   renamed `eval_metrics` ([see above](#metrics)).
*  The @{tf.train.SessionRunHook$hooks} are unsupported, so these fields are
   omitted.
*  The @{tf.train.Scaffold$`scaffold`}, if used, must also be wrapped in a
   function. This field is renamed to `scaffold_fn`.

`Scaffold` and `Hooks` are for advanced usage, and can typically be omitted.

## Input functions

Input functions work mainly unchanged as they run on the host computer, not the
Cloud TPU itself. This section explains the two necessary adjustments.

### Params argument

<!-- TODO(markdaoust) link to input_fn doc when it exists -->

The `input_fn` for a standard `Estimator` _can_ include a
`params` argument; the `input_fn` for a `TPUEstimator` *must* include a
`params` argument. This is necessary to allow the estimator to set the batch
size for each replica of the input stream. So the minimum signature for an
`input_fn` for a `TPUEstimator` is:

```
def my_input_fn(params):
  pass
```

Where `params['batch-size']` will contain the batch size.

### Static shapes and batch size

The input pipeline generated by your `input_fn` is run on CPU. So it is mostly
free from the strict static shape requirements imposed by the XLA/TPU environment.
The one requirement is that the batches of data fed from your input pipeline to
the TPU have a static shape, as determined by the standard TensorFlow shape
inference algorithm. Intermediate tensors are free to have a dynamic shapes.
If shape inference has failed, but the shape is known it is possible to
impose the correct shape using `tf.set_shape()`. 

In the example below the shape
inference algorithm fails, but it is correctly using `set_shape`:

```
>>> x = tf.zeros(tf.constant([1,2,3])+1)
>>> x.shape

TensorShape([Dimension(None), Dimension(None), Dimension(None)])

>>> x.set_shape([2,3,4])
```

In many cases the batch size is the only unknown dimension.

A typical input pipeline, using `tf.data`, will usually produce batches of a
fixed size. The last batch of a finite `Dataset`, however, is typically smaller,
containing just the remaining elements. Since a `Dataset` does not know its own
length or finiteness, the standard @{tf.data.Dataset.batch$`batch`} method
cannot determine if all batches will have a fixed size batch on its own:

```
>>> params = {'batch_size':32}
>>> ds = tf.data.Dataset.from_tensors([0, 1, 2])
>>> ds = ds.repeat().batch(params['batch-size'])
>>> ds

<BatchDataset shapes: (?, 3), types: tf.int32>
```

The most straightforward fix is to
@{tf.data.Dataset.apply$apply} @{tf.contrib.data.batch_and_drop_remainder}
as follows:

```
>>> params = {'batch_size':32}
>>> ds = tf.data.Dataset.from_tensors([0, 1, 2])
>>> ds = ds.repeat().apply(
...     tf.contrib.data.batch_and_drop_remainder(params['batch-size']))
>>> ds

 <_RestructuredDataset shapes: (32, 3), types: tf.int32>
```

The one downside to this approach is that, as the name implies, this batching
method throws out any fractional batch at the end of the dataset. This is fine
for an infinitely repeating dataset being used for training, but could be a
problem if you want to train for an exact number of epochs.

To do an exact 1-epoch of _evaluation_ you can work around this by manually
padding the length of the batches, and setting the padding entries to have zero
weight when creating your `tf.metrics`.

## Datasets

Efficient use of the `tf.data.Dataset` API is critical when using a Cloud
TPU, as it is impossible to use the Cloud TPU's unless you can feed it data
quickly enough. See @{$datasets_performance} for details on dataset performance.

For all but the simplest experimentation (using
@{tf.data.Dataset.from_tensor_slices} or other in-graph data) you will need to
store all data files read by the `TPUEstimator`'s `Dataset` in Google Cloud
Storage Buckets.

<!--TODO(markdaoust): link to the `TFRecord` doc when it exists.-->

For most use-cases, we recommend converting your data into `TFRecord`
format and using a @{tf.data.TFRecordDataset} to read it. This, however, is not
a hard requirement and you can use other dataset readers
(`FixedLengthRecordDataset` or `TextLineDataset`) if you prefer.

Small datasets can be loaded entirely into memory using
@{tf.data.Dataset.cache}.

Regardless of the data format used, it is strongly recommended that you
@{$performance_guide#use_large_files$use large files}, on the order of
100MB. This is especially important in this networked setting as the overhead
of opening a file is significantly higher.

It is also important, regardless of the type of reader used, to enable buffering
using the `buffer_size` argument to the constructor. This argument is specified
in bytes. A minimum of a few MB (`buffer_size=8*1024*1024`) is recommended so
that data is available when needed.

The TPU-demos repo includes
[a script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)
for downloading the imagenet dataset and converting it to an appropriate format.
This together with the imagenet
[models](https://github.com/tensorflow/tpu/tree/master/models)
included in the repo demonstrate all of these best-practices.


## What Next

For details on how to actually set up and run a Cloud TPU see:

 * [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs/)

This document is by no means exhaustive. The best source of more detail on how
to make a Cloud TPU compatible model are the example models published in:

 * The [TPU Demos Repository.](https://github.com/tensorflow/tpu)

For more information about tuning TensorFlow code for performance see:

 * The @{$performance$Performance Section.}

