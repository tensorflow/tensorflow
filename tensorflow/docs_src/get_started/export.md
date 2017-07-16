# Exporting a Trained Model for Serving

Once you have trained an `Estimator` model, you may want to create a service
from that model that takes requests and returns a result.  You can run such a
service locally on your machine or deploy it scalably in the cloud.

To prepare a trained Estimator for serving, you must export it in the standard
[`SavedModel`](https://www.tensorflow.org/code/tensorflow/python/saved_model/README.md)
format, which wraps the TensorFlow graph, the trained variable values, any
required assets, and metadata together in a hermetic package.

In this tutorial, we will discuss how to:

* Add graph nodes that accept and prepare inference requests
* Specify the output nodes and the corresponding [APIs](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)
  that can be served (Classify, Regress, or Predict)
* Export your model to the `SavedModel` format
* Deploy the model in Google Cloud ML Engine and request predictions
* Serve the model from a local server and request predictions


## The exported graph and its signatures

The export procedure assembles a new TensorFlow graph from two main components:
1) a Serving Input Receiver that defines the format of the inputs to be
   accepted, and
2) the trained model itself.

An exported `SavedModel` contains that combined graph packaged together with one
or more *signatures*.  Like a function signature in any programming language, a
graph signature specifies the required inputs (arguments) and the
expected outputs (return values) of performing the computation.  In the typical
case, a single signature is present, corresponding to the predictions that the
model has learned to make.

The *input* portion of the signature is determined by the Serving Input
Receiver.  To specify the inputs that your deployed model will accept, you must
provide a `serving_input_receiver_fn()` to `estimator.export_savedmodel()` (see
below).

The *output* portion of the signature is determined by the model.  For instance,
canned Estimators know the nature of the outputs they produce (e.g. whether the
output is a classification or a regression, and the type and shape of those
outputs).  Custom Estimators must provide this information via `export_outputs`
(see [below](#specifying_the_outputs_of_a_custom_model)).

> Note: A *multi-headed model* provides multiple signatures, each corresponding
> to a different "head", i.e. a set of predictions that can be made from the
> same inputs by executing a subgraph of the complete trained graph.  The
> *output* portions of these signatures are determined by the model.


![Overview of exporting a SavedModel from Estimator](../images/export_savedmodel_overview.png)

## Preparing serving inputs

During training, an @{$input_fn$`input_fn()`} ingests data and prepares it for
use by the model.  At serving time, similarly, a `serving_input_receiver_fn()`
accepts inference requests and prepares them for the model.  The purpose of this
function is to add placeholders to the graph which the serving system will feed
with inference requests, as well as to add any additional ops needed to convert
data from the input format into the feature `Tensor`s expected by the model.
The function returns a @{tf.estimator.export.ServingInputReceiver} object, which
packages the placeholders and the resulting feature `Tensor`s together.

A typical pattern is that inference requests arrive in the form of serialized
`tf.Example`s, so the `serving_input_receiver_fn()` creates a single string
placeholder to receive them.  The `serving_input_receiver_fn()` is then also
responsible for parsing the `tf.Example`s by adding a @{tf.parse_example} op to
the graph.

When writing such a `serving_input_receiver_fn()`, you must pass a parsing
specification to @{tf.parse_example} to tell the parser what feature names to
expect and how to map them to `Tensor`s. A parsing specification takes the
form of a dict from feature names to @{tf.FixedLenFeature}, @{tf.VarLenFeature},
and @{tf.SparseFeature}.  (Note this parsing specification should not include
any label or weight columns, since those will not be available at serving
time&mdash;in contrast to a parsing specification used in the `input_fn()` at
training time.)

In combination, then:

```py
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
```

The @{tf.estimator.export.build_parsing_serving_input_receiver_fn} utility
function provides that input receiver for the common case.

> Note: when training a model to be served using Google Cloud ML Engine (see
> below), the parsing step is not needed, because the model will receive raw
> feature data.  This is also true when using the Predict API with a local
> server.

Even if you require no parsing or other input processing&mdash;i.e., if the
serving system will feed feature `Tensor`s directly&mdash;you must still provide
a `serving_input_receiver_fn()` that creates placeholders for the feature
`Tensor`s and passes them through.  The
@{tf.estimator.export.build_raw_serving_input_receiver_fn} utility provides for
this.

If these utilities do not meet your needs, you are free to write your own
`serving_input_receiver_fn()`.  One case where this may be needed is if your
training `input_fn()` incorporates some preprocessing logic that must be
recapitulated at serving time.  To reduce the risk of training-serving skew, we
recommend encapsulating such processing in a function which is then called
from both `input_fn()` and `serving_input_receiver_fn()`.


## Performing the export

To export your trained Estimator, call
@{tf.estimator.Estimator.export_savedmodel} with the export base path, together
with the `serving_input_receiver_fn`.

```py
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn)
```

This method builds a new graph by first calling the
`serving_input_receiver_fn()` to obtain feature `Tensor`s, and then calling
this `Estimator`'s `model_fn()` to generate the model graph based on those
features. It starts a fresh `Session`, and, by default, restores the most recent
checkpoint into it.  (A different checkpoint may be passed, if needed.)
Finally it creates a timestamped export directory below the given
`export_dir_base` (i.e., `export_dir_base/<timestamp>`), and writes a
`SavedModel` into it containing a single `MetaGraphDef` saved from this
Session.

> Note: there is currently no built-in mechanism to garbage-collect old exports,
> so successive exports will accumulate under `export_dir_base` unless deleted
> by some external means.

## Specifying the outputs of a custom model

When writing a custom `model_fn`, you must populate the `export_outputs` element
of the @{tf.estimator.EstimatorSpec} return value. This is a dict of
`{name: output}` describing the output signatures to be exported and used during
serving.

In the usual case of making a single prediction, this dict contains
one element, and the `name` is immaterial.  In a multi-headed model, each head
is represented by an entry in this dict.  In this case the `name` is a string
of your choice that can be used to request a specific head at serving time.

Each `output` value must be an `ExportOutput` object  such as
@{tf.estimator.export.ClassificationOutput},
@{tf.estimator.export.RegressionOutput}, or
@{tf.estimator.export.PredictOutput}.

These output types map straightforwardly to the
[TensorFlow Serving APIs](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto),
and so determine which request types will be honored.

> Note: In the multi-headed case, a `SignatureDef` will be generated for each
> element of the `export_outputs` dict returned from the model_fn, named using
> the same keys.  These signatures differ only in their outputs, as provided by
> the corresponding `ExportOutput` entry.  The inputs are always those provided
> by the `serving_input_receiver_fn`.
> An inference request may specify the head by name.  One head must be named
> using [`signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`](https://www.tensorflow.org/code/saved_model/signature_constants.py)
> indicating which signature will be served when an inference request does not
> specify one.


## Serving the exported model on Google Cloud ML Engine

[Google Cloud ML Engine](https://cloud.google.com/ml-engine/) provides a fully
managed, scalable environment for serving your trained SavedModels to make
online or batch predictions.

Please see [Deploying Models](https://cloud.google.com/ml-engine/docs/how-tos/deploying-models)
to learn how to deploy your SavedModel on Cloud ML Engine.

> Note: Cloud ML Engine accepts inference requests in JSON, CSV, or TFRecords
> formats, depending on the circumstance.  Parsing these formats is not the
> responsibility of the graph.  Cloud ML Engine does the parsing for you, and
> feeds raw feature data directly into the graph.  Thus, when targeting Cloud ML
> Engine, you should use a `serving_input_receiver_fn()` of the passthrough form
> that simply creates placeholders for each feature.


## Requesting predictions from Google Cloud ML Engine

To learn how to request predictions from a model deployed in Cloud ML Engine,
please see:

* [Prediction Basics](https://cloud.google.com/ml-engine/docs/concepts/prediction-overview)
* [Getting Online Predictions](https://cloud.google.com/ml-engine/docs/how-tos/online-predict)
* [Getting Batch Predictions](https://cloud.google.com/ml-engine/docs/how-tos/batch-predict)


## Serving the exported model locally

For local deployment, you can serve your model using
@{$deploy/tfserve$Tensorflow Serving}, an open-source project that loads a
`SavedModel` and exposes it as a [gRPC](http://www.grpc.io/) service.

First, [install TensorFlow Serving](https://tensorflow.github.io/serving/setup#prerequisites).

Then build and run the local model server, substituting `$export_dir_base` with
the path to the `SavedModel` you exported above:

```sh
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=$export_dir_base
```

Now you have a server listening for inference requests via gRPC on port 9000!


## Requesting predictions from a local server

The server responds to gRPC requests according to the [PredictionService](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto#L15)
gRPC API service definition.  (The nested protocol buffers are defined in
various [neighboring files](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis)).

From the API service definition, the gRPC framework generates client libraries
in various languages providing remote access to the API.  In a project using the
Bazel build tool, these libraries are built automatically and provided via
dependencies like these (using Python for example):

```build
  deps = [
    "//tensorflow_serving/apis:classification_proto_py_pb2",
    "//tensorflow_serving/apis:regression_proto_py_pb2",
    "//tensorflow_serving/apis:predict_proto_py_pb2",
    "//tensorflow_serving/apis:prediction_service_proto_py_pb2"
  ]
```

Python client code can then import the libraries thus:

```py
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
```

> Note: `prediction_service_pb2` defines the service as a whole and so
> is always required.  However a typical client will need only one of
> `classification_pb2`, `regression_pb2`, and `predict_pb2`, depending on the
> type of requests being made.

Sending a gRPC request is then accomplished by assembling a protocol buffer
containing the request data and passing it to the service stub.  Note how the
request protocol buffer is created empty and then populated via the
[generated protocol buffer API](https://developers.google.com/protocol-buffers/docs/reference/python-generated).

```py
from grpc.beta import implementations

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))

result = stub.Classify(request, 10.0)  # 10 secs timeout
```

The returned result in this example is a `ClassificationResponse` protocol
buffer.

This is a skeletal example; please see the @{$deploy$Tensorflow Serving}
documentation and [examples](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example)
for more details.

> Note: `ClassificationRequest` and `RegressionRequest` contain a
> `tensorflow.serving.Input` protocol buffer, which in turn contains a list of
> `tensorflow.Example` protocol buffers.  `PredictRequest`, by contrast,
> contains a mapping from feature names to values encoded via `TensorProto`.
> Correspondingly: When using the `Classify` and `Regress` APIs, TensorFlow
> Serving feeds serialized `tf.Example`s to the graph, so your
> `serving_input_receiver_fn()` should include a `tf.parse_example()` Op.
> When using the generic `Predict` API, however, TensorFlow Serving feeds raw
> feature data to the graph, so a passthrough `serving_input_receiver_fn()`
> should be used.


<!-- TODO(soergel): give examples of making requests against this server, using
the different Tensorflow Serving APIs, selecting the signature by key, etc. -->

<!-- TODO(soergel): document ExportStrategy here once Experiment moves
from contrib to core. -->

