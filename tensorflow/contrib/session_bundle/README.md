# TensorFlow Inference Model Format

WARNING: SessionBundle has been deprecated. Please use
[SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) instead.

[TOC]

## Overview

This document describes the data formats and layouts for exporting
[TensorFlow](https://www.tensorflow.org/) models for inference.

These exports have the following properties:

*   Recoverable
    *   given an export the graph can easily be initialized and run
*   Hermetic
    *   an export directory is self-contained to facilitate distribution

The TensorFlow `Saver` writes **checkpoints** (graph variables) while training
so it can recover if it crashes. A TensorFlow Serving **export** contains a
checkpoint with the current state of the graph variables along with a MetaGraph
definition that's needed for serving.

## Directory Structure

~~~
# Directory overview
00000000/
         assets/
         export.meta
         export-?????-of-?????
~~~

*   `00000000` -- Export version
    *   Format `%08d`
*   `assets` -- Asset file directory
    *   Holds auxiliary files for the graph (e.g., vocabularies)
*   `export.meta` -- MetaGraph Definition
    *   Binary [`tensorflow::MetaGraphDef`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/protobuf/meta_graph.proto)
*   `export-?????-of-?????`
    *   A checkpoint of the Graph Variables
    *   Outputs from Python [`Saver`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training/saver.py)
        with `sharded=True`.

## Exporting (Python code)

The [`Exporter`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/session_bundle/exporter.py)
class can be used to export a model in the above format from a TensorFlow Python
binary.

### Exporting TF.learn models

TF.learn uses an
[Exporter wrapper](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/utils/export.py)
that can be used for building signatures. Use the `BaseEstimator.export`
function to export your Estimator with a signature.

## Importing (C++ code)

The [`LoadSessionBundleFromPath`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/session_bundle/session_bundle.h)
function can be used to create a `tensorflow::Session` and initialize it from an
export. This function takes session options and the path to the export as
arguments and returns a bundle of export data including a `tensorflow::Session`
which can be run.

## Signatures

Graphs used for inference tasks typically have set of inputs and outputs used at
inference time. We call this a 'Signature'.

### Standard Signatures (standard usage)

Graphs used for standard inference tasks have standard sets of inputs and
outputs. For example, a graph used for a regression task has an input tensor for
the data and an output tensor for the regression values. The signature mechanism
makes it easy to identify the relevant input and output tensors for common graph
applications.

The [`Manifest`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/session_bundle/manifest.proto)
contains a `Signature` message which contains the task specific inputs and
outputs.

~~~proto
// A Signature specifies the inputs and outputs of commonly used graphs.
message Signature {
  oneof type {
    RegressionSignature regression_signature = 1;
    ClassificationSignature classification_signature = 2;
    GenericSignature generic_signature = 3;
  }
};
~~~

A Standard Signature can be set at export time using the `Exporter` API.

~~~python
# Run an export.
signature = exporter.classification_signature(input_tensor=input,
                                              classes_tensor=output)
export = exporter.Exporter(saver)
export.init(sess.graph.as_graph_def(), default_graph_signature=signature)
export.export(export_path, global_step_tensor, sess)
~~~

#### TF.learn signatures

TF.learn models can use the `BaseEstimator.export` function directly to export.
To specify a Signature, use the [Exporter wrapper](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/utils/export.py)
helpers (e.g. `classification_signature_fn`).

~~~python
estimator = tf.contrib.learn.Estimator(...)
...
# Other possible parameters omitted for the sake of brevity.
estimator.export(
    export_path,
    signature_fn=tf.contrib.learn.utils.export.classification_signature_fn)
~~~

#### Recovering signatures

These can be recovered at serving time using utilities in [`signature.h`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/session_bundle/signature.h)

~~~c++
// Get the classification signature.
ClassificationSignature signature;
TF_CHECK_OK(GetClassificationSignature(bundle->meta_graph_def, &signature));

// Run the graph.
Tensor input_tensor = GetInputTensor();
Tensor classes_tensor;
Tensor scores_tensor;
TF_CHECK_OK(RunClassification(signature, input_tensor, session, &classes_tensor,
                              &scores_tensor));
~~~

### Generic Signatures (custom or advanced usage)

Generic Signatures enable fully custom usage of the `tensorflow::Session` API.
They are recommended for when the Standard Signatures do not satisfy a
particular use-case. A general example of when to use these is for a model
taking a single input and generating multiple outputs performing different
inferences.

~~~proto
// GenericSignature specifies a map from logical name to Tensor name.
// Typical application of GenericSignature is to use a single GenericSignature
// that includes all of the Tensor nodes and target names that may be useful at
// serving, analysis or debugging time. The recommended name for this signature
// is "generic_bindings".
message GenericSignature {
  map<string, TensorBinding> map = 1;
};
~~~

Generic Signatures can be used to compliment a Standard Signature, for example
to support debugging. Here is an example that includes the Standard regression
Signature and a Generic Signature.

~~~python
named_tensor_bindings = {"logical_input_A": v0,
                         "logical_input_B": v1}
signatures = {
    "regression": exporter.regression_signature(input_tensor=v0,
                                                output_tensor=v1),
    "generic": exporter.generic_signature(named_tensor_bindings)}
export = exporter.Exporter(saver)
export.init(sess.graph.as_graph_def(), named_graph_signatures=signatures)
export.export(export_path, global_step_tensor, sess)
~~~

Generic Signature does not differentiate between input and output tensors. It
provides full flexibility to specify the input and output tensors you need.
The benefit is preserving a mapping between names that you specify at export
time (we call these the logical names), and the actual graph node names that may
be less stable and/or auto-generated by TensorFlow.

In [`signature.h`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/session_bundle/signature.h)
note that the generic signature methods `BindGenericInputs` and
`BindGenericNames` are doing simple string to string mapping as a convenience.
These methods map from the names used at training time to actual names in the
graph.

The bound results from those methods can be used as inputs to
`tensorflow::Session->Run()`. Specifically, the bound result
`vector<pair<string, Tensor>>` from `BindGenericInputs` can be supplied as the
first parameter `inputs` to `tensorflow::Session->Run()`. Similarly, the bound
result `vector<string>` from `BindGenericNames`, can be mapped to
`output_tensor_names` in the `tensorflow::Session->Run()` arguments. The next
parameter, `target_node_names` is typically null at inference time. The last
parameter `outputs` is for the results, which share the same order as the
supplied `output_tensor_names`.

## Custom Initialization

Some graphs many require custom initialization after the variables have been
restored. Such initialization, done through an arbitrary Op, can be added using
the `Exporter` API. If set, `LoadSessionBundleFromPath` will automatically run
the Op when restoring a `Session` following the loading of variables.

## Assets

In many cases we have Ops which depend on external files for initialization
(such as vocabularies). These "assets" are not stored in the graph and are
needed for both training and inference.

In order to create hermetic exports these asset files need to be:

1. copied to each export directory, and
2. read when recovering a session from an export base directory.

Copying assets to the export directory is handled with a callback mechanism.
The callback function receives two parameters:

1. the dictionary of source files to desired basename, and
2. the export directory.
The default callback uses `gfile.Copy` to perform the copy.

The tensor that contains the filepath to be copied is specified by passing the
collection of asset filepath tensor, which is usually extracted from the graph
by `tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)`.

~~~python
# Run an export.
export = exporter.Exporter(save)
export.init(
    sess.graph.as_graph_def(),
    asset_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
export.export(export_path, global_step_tensor, sess)
~~~

Users can use their own callbacks as shown in the following example, with the
requirement to keep the basename of the original files:

~~~python
def my_custom_copy_callback(files_to_copy, export_dir_path):
  # Copy all source files (keys) in files_to_copy to export_dir_path using the
  # corresponding basename (value).
  ...

# Run an export.
export = exporter.Exporter(save)
export.init(
    sess.graph.as_graph_def(),
    asset_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
    asset_callback=my_custom_copy_callback)
export.export(export_path, global_step_tensor, sess)
~~~

`AssetFile` binds the name of a tensor in the graph to the name of a file
within the assets directory. `LoadSessionBundleFromPath` will handle the base
path and asset directory swap/concatenation such that the tensor is set with
the fully qualified filename upon return.

# Exporter Usage

The typical workflow of model exporting is:

1. Build model graph G.
2. Train variables or load trained variables from checkpoint in session S.
3. [Optional] Build inference graph I.
4. Export G.

The Exporter should be used as follows:

1. The Saver used in Exporter(saver) should be created within the context of G.
2. Exporter.init() should be called within the context of G.
3. Exporter.export() should be called using session S.
4. If I is provided for Exporter.init(), an exact same Saver should be created
   under I as the saver under G -- in the way that exact same Save/Restore ops
   are created in both G and S.
