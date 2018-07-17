# TensorFlow SavedModel

[TOC]

## Overview
This document describes SavedModel, the universal serialization format for
[TensorFlow](https://www.tensorflow.org/) models.

SavedModel provides a language-neutral format to save machine-learned models
that is recoverable and hermetic. It enables higher-level systems and tools to
produce, consume and transform TensorFlow models.

## Features

The following is a summary of the features in SavedModel:

* Multiple graphs sharing a single set of variables and assets can be added to a
  single SavedModel. Each graph is associated with a specific set of tags to
  allow identification during a load or restore operation.
* Support for `SignatureDefs`
    * Graphs that are used for inference tasks typically have a set of inputs
      and outputs. This is called a `Signature`.
    * SavedModel uses [SignatureDefs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)
      to allow generic support for signatures that may need to be saved with the graphs.
    * For commonly used SignatureDefs in the context of TensorFlow Serving,
      please see documentation [here](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md).
* Support for `Assets`.
    * For cases where ops depend on external files for initialization, such as
      vocabularies, SavedModel supports this via `assets`.
    * Assets are copied to the SavedModel location and can be read when loading
      a specific meta graph def.
* Support to clear devices before generating the SavedModel.

The following is a summary of features that are NOT supported in SavedModel.
Higher-level frameworks and tools that use SavedModel may provide these.

* Implicit versioning.
* Garbage collection.
* Atomic writes to the SavedModel location.

## Background
SavedModel manages and builds upon existing TensorFlow primitives such as
`TensorFlow Saver` and `MetaGraphDef`. Specifically, SavedModel wraps a [TensorFlow Saver](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training/saver.py).
The Saver is primarily used to generate the variable checkpoints. SavedModel
will replace the existing [TensorFlow Inference Model Format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md)
as the canonical way to export TensorFlow graphs for serving.

## Components
A SavedModel directory has the following structure:

```
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb
```

* SavedModel protocol buffer
    * `saved_model.pb` or `saved_model.pbtxt`
    * Includes the graph definitions as `MetaGraphDef` protocol buffers.
* Assets
    * Subfolder called `assets`.
    * Contains auxiliary files such as vocabularies, etc.
* Extra assets
    * Subfolder where higher-level libraries and users can add their own assets
      that co-exist with the model, but are not loaded by the graph.
    * This subfolder is not managed by the SavedModel libraries.
* Variables
    * Subfolder called `variables`.
    * Includes output from the [TensorFlow Saver](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training/saver.py).
        * `variables.data-?????-of-?????`
        * `variables.index`

## APIs
The APIs for building and loading a SavedModel are described in this section.

### Simple

The highest level API for exporting the model components described above
is using the [simple_save](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/simple_save.py)
module. For most use-cases this functionality is sufficient to export
a model which is compatible with TensorFlow serving. A corresponding [simple_load](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/simple_load.py)
module is implemented in python which enables SavedModels to be reloaded
within a python TensorFlow session.

#### Simple Save Usage
~~~python
import tensorflow as tf

# Define model graph
x = tf.placeholder(dtype=tf.int32)
y = x + 1

# Export the model to the given directory
export_dir = "example_model_directory"
with tf.Session() as sess:
    tf.saved_model.simple_save(sess, export_dir, {"x": x}, {"y": y})
~~~

After calling the `simple_save` function, the model components will be
exported to the given `export_dir`. These artifacts can be used with
TensorFlow serving or using the python `simple_load` method. Though the
final target may be to host the model in TensorFlow serving, it is often
useful to test the artifact in python.

For example, to load and execute the model exported in the example
above, the `simple_load` function would be used in the following way:

#### Simple Load Usage
~~~python
import tensorflow as tf

# Load exported model from the given directory
export_dir = "example_model_directory"
with tf.Session() as sess:
    inputs, outputs = tf.saved_model.simple_load(sess, export_dir)
    result = sess.run(outputs["y"], feed_dict={inputs["x"]: 1})

assert result == 2
~~~

### Builder

In specific use-cases the [simple_save](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/simple_save.py)
API may not provide the level of control to export a more complex model.

The SavedModel [builder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py)
is implemented in Python.

The `SavedModelBuilder` class provides functionality to save multiple meta graph
defs, associated variables and assets.

To build a SavedModel, the first meta graph must be saved with variables.
Subsequent meta graphs will simply be saved with their graph definitions. If
assets need to be saved and written or copied to disk, they can be provided
when the meta graph def is added. If multiple meta graph defs are associated
with an asset of the same name, only the first version is retained.

#### Tags
Each meta graph added to the SavedModel must be annotated with user specified
tags. The tags provide a means to identify the specific meta graph to load and
restore, along with the shared set of variables and assets. These tags
typically annotate a MetaGraph with its functionality (e.g. serving or
training), and possibly hardware specific aspects such as GPU.

#### Usage
The typical usage of `builder` is as follows:

~~~python
export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph(["bar-tag", "baz-tag"])
...
builder.save()
~~~

#### Stripping Default valued attributes
The SavedModelBuilder class allows users to control whether default-valued
attributes must be stripped from the NodeDefs while adding a meta graph to the
SavedModel bundle. Both `SavedModelBuilder.add_meta_graph_and_variables` and
`SavedModelBuilder.add_meta_graph` methods accept a Boolean flag
`strip_default_attrs` that controls this behavior.

If `strip_default_attrs` is `False`, the exported MetaGraphDef will have the
default valued attributes in all it's NodeDef instances. This can break forward
compatibility with a sequence of events such as the following:

* An existing Op (`Foo`) is updated to include a new attribute (`T`) with a
  default (`bool`) at version 101.
* A model producer (such as a Trainer) binary picks up this change
  (version 101) to the OpDef and re-exports an existing model that uses Op `Foo`.
* A model consumer (such as Tensorflow Serving) running an older binary
  (version 100) doesn't have attribute `T` for Op `Foo`, but tries to import
  this model. The model consumer doesn't recognize attribute `T` in a NodeDef
  that uses Op `Foo` and therefore fails to load the model.

By setting `strip_default_attrs` to `True`, the model producers can strip away
any default valued attributes in the NodeDefs. This helps ensure that newly
added attributes with defaults don't cause older model consumers to fail loading
models regenerated with newer training binaries.

TIP: If you care about forward compatibility, then set `strip_default_attrs`
to `True` while using `SavedModelBuilder.add_meta_graph_and_variables` and
`SavedModelBuilder.add_meta_graph`.

### Loader
The SavedModel loader is implemented in C++ and Python.

#### Python
The Python version of the SavedModel [loader](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/loader.py)
provides load and restore capability for a SavedModel. The `load` operation
requires the session in which to restore the graph definition and variables, the
tags used to identify the meta graph def to load and the location of the
SavedModel. Upon a load, the subset of variables and assets supplied as part of
the specific meta graph def, will be restored into the supplied session.

~~~python
export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...
~~~

#### C++
The C++ version of the SavedModel [loader](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/loader.h)
provides an API to load a SavedModel from a path, while allowing
`SessionOptions` and `RunOptions`. Similar to the Python version, the C++
version requires the tags associated with the graph to be loaded, to be
specified. The loaded version of SavedModel is referred to as `SavedModelBundle`
and contains the meta graph def and the session within which it is loaded.

~~~c++
const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &bundle);
~~~

### Constants
SavedModel offers the flexibility to build and load TensorFlow graphs for a
variety of use-cases. For the set of most common expected use-cases,
SavedModel's APIs provide a set of constants in Python and C++ that are easy to
reuse and share across tools consistently.

#### Tag constants
Sets of tags can be used to uniquely identify a `MetaGraphDef` saved in a
SavedModel. A subset of commonly used tags is specified in:

* [Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py)
* [C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h).

#### Signature constants
SignatureDefs are used to define the signature of a computation supported in a
TensorFlow graph. Commonly used input keys, output keys and method names are
defined in:

* [Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)
* [C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/signature_constants.h).

