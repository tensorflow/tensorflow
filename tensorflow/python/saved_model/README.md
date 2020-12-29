# TensorFlow SavedModel

[TOC]

## Overview

SavedModel is the universal serialization format for
[TensorFlow](https://www.tensorflow.org/) models.

SavedModel provides a language-neutral format to save machine-learning models
that is recoverable and hermetic. It enables higher-level systems and tools to
produce, consume and transform TensorFlow models.

## Guides
* [Using the SavedModel Format](https://www.tensorflow.org/guide/saved_model)
* [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize)
* [Save and load with checkpointing in Keras](https://www.tensorflow.org/tutorials/keras/save_and_load)
* [Training checkpoints](https://www.tensorflow.org/guide/checkpoint)
* [Save and load a model using a distribution strategy](https://www.tensorflow.org/tutorials/distribute/save_and_load)


## [Public API](https://www.tensorflow.org/api_docs/python/tf/saved_model)
* [`tf.saved_model.save`](https://www.tensorflow.org/api_docs/python/tf/saved_model/save)
* [`tf.saved_model.load`](https://www.tensorflow.org/api_docs/python/tf/saved_model/load)
* [`tf.saved_model.SaveOptions`](https://www.tensorflow.org/api_docs/python/tf/saved_model/SaveOptions)
* [`tf.saved_model.LoadOptions`](https://www.tensorflow.org/api_docs/python/tf/saved_model/LoadOptions)
* [`tf.saved_model.Asset`](https://www.tensorflow.org/api_docs/python/tf/saved_model/Asset)
* [`tf.saved_model.contains_saved_model`](https://www.tensorflow.org/api_docs/python/tf/saved_model/contains_saved_model)

### Related Modules and Functions
* [`tf.keras.models.save_model`](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
* [`tf.keras.models.load_model`](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)
* [`tf.train.Checkpoint`](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)


## The SavedModel Format
A SavedModel directory has the following structure:

```
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb
```

*   SavedModel protocol buffer
    *   [`saved_model.pb`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_model.proto)
        or `saved_model.pbtxt`
    *   Includes the graph definitions as `MetaGraphDef` protocol buffers.
*   Assets
    *   Subfolder called `assets`.
    *   Contains auxiliary files such as vocabularies, etc.
*   Extra assets
    *   Subfolder where higher-level libraries and users can add their own
        assets that co-exist with the model, but are not loaded by the graph.
    *   This subfolder is not managed by the SavedModel libraries.
*   Variables
    *   Subfolder called `variables`.
        *   `variables.data-?????-of-?????`
        *   `variables.index`

---

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
