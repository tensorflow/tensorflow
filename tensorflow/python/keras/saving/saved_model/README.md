# Keras SavedModel

For questions, feedback, and feature requests please file a bug/contact kathywu@

## TensorFlow Core SavedModel implementation

In TensorFlow 2.0, all saving and loading implementations revolve around the
object graph generated from a root trackable object, and all trackable objects
connected to it through attributes. Program building blocks such as variables,
assets, and tables, and high level objects like Optimizers and Layers all
subclass the trackable class. Other objects like TensorFlow functions and
concrete functions are also saved as nodes in the object graph. When loading a
SavedModel, the object graph is used to recreate the structure of the original
object.

Please see the links below for more details:

- [Saved Model Guide](https://www.tensorflow.org/guide/saved_model)
- [Checkpoint Guide](https://www.tensorflow.org/guide/checkpoint)

## Keras SavedModel implementation

### Overview

Keras object serialization is built on top of the core serialization.

All attributes that impact model execution or inspection are saved to the
SavedModel to allow the model to be recreated. These attributes are divided into
three categories:

1. python properties (e.g., layer name, layer config)
2. objects (e.g. data structures like list of variables or layers)
3. functions (e.g. call function, loss functions)

Trackable objects and TensorFlow functions are represented as nodes in the
trackable object graph, and each node in the graph stores information about
their python properties.

Since many attributes in Keras Layers/Models are not Trackable objects or
tf.functions, these attributes are wrapped as trackable objects/tf.functions at
serialization time. For example, `layer.variables` is implemented as a python
property that appends the lists of trainable/nontrainable variables. During
serialization, a new Trackable List object is created and saved to the
`variables` attribute. Another example is the call function. Most models do not
decorate their call function with `tf.function`, since Keras will take care of
the graph/function management. When the model is saved, the call function is
wrapped in a `tf.function` and added to the `__call__` attribute.


### `keras_api` attribute

Many attributes are only relevant for revivability. Instead of attaching these
directly to the exported object, they are saved to a new `keras_api` trackable
object that is then attached to the exported object. This avoids cluttering the
exported object with objects/functions that are only used by the Keras library.

For example, `__call__` and `call_and_return_conditional_losses` are functions
saved for all models. The `__call__` function is attached directly to the
exported object, while `call_and_return_conditional_losses` is attached to a
separate object. Say a user saves the model, then loads the SavedModel using the
core loader (tf.saved_model.load which does not rely on the Keras library to
revive the model).

The loaded object will have a structure that looks like:

```
  loaded object -- __call__
                -- keras_api -- __call__
                             -- call_and_return_conditional_losses
```

The two call functions may be accessed through:

  - `loaded.__call__` or `loaded.keras_api.__call__`
  - `loaded.keras_api.call_and_return_conditional_losses`.


### Saving details

Keras Layers use a helper abstract class and an attribute validator class to
define and standardize the serialization implementation:

- [`SerializationImpl`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/base_serialization.py):
Ensures that layer python properties are saved as a serialized JSON string in
the metadata field, and gathers all attributes to save with the Keras object.
Please see the docstrings in each of the abstract methods/properties to see what
is required.
- [`SerializedAttributes`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/serialized_attributes.py?):
Tracks all of the attributes that must be saved with a Keras object. Objects and
functions may be specified to be "keras_only", meaning that they will only
appear in the `keras_api` attribute.

The base `Layer` serialization is defined in
[`layer_serialization.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/layer_serialization.py).
See `LayerAttributes` and `LayerSerializationImpl`.

**Adding a new attribute to base Layer SavedModel**

1. Add a new attributes to `LayerAttributes`.
2. Modify `LayerSerializationImpl` internal methods:

   a. If adding a python property, add the key-value item to the dictionary
   returned by `_python_properties_internal`

   b.If adding a new object/function, modify the dictionary returned by
   `_get_serialized_attributes_internal`.


**Adding custom serialization for a Layer subclass.**

1. Create a new attribute validator by copying `LayerAttributes`, and add any
new attributes to serialize.
2. Subclass `LayerSerializationImpl`
3. Implement `_python_properties_internal` and/or
`_get_serialized_attributes_internal` to return the new attributes.

Unless you are modifying the loader (see section below on loading), please keep
the `object_identifier` the same.

These instructions also carry over for modifying
[Model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/model_serialization.py)
and
[Network](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/network_serialization.py)
serialization.


### Loading details

TODO(kathywu): Will write this section when the loading code is moved into
\*_serialization.py files.

