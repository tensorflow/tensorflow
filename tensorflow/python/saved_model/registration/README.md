# Registrations

To configure SaveModel or checkpointing beyond the basic saving and loading
steps [documentation TBD], registration is required.

Currently, only TensorFlow-internal
registrations are allowed, and must be added to the allowlist.

* `tensorflow.python.saved_model.registration.register_tf_serializable`
  * Allowlist: tf_serializable_allowlist.txt
*  `tensorflow.python.saved_model.registration.register_tf_checkpoint_saver`
  * Allowlist: tf_checkpoint_saver_allowlist.txt

[TOC]

## SavedModel serializable registration

Custom objects must be registered in order to get the correct deserialization
method when loading. The registered name of the class is saved to the proto.

Keras already has a similar mechanism for registering serializables:
[`tf.keras.utils.register_keras_serializable(package, name)`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/register_keras_serializable).
This has been imported to core TensorFlow:

```python
registration.register_serializable(package, name)
registration.register_tf_serializable(name)  # If TensorFlow-internal.
```

*   package: The package that this class belongs to.
*   name: The name of this class. The registered name that is saved in the proto
    is "{package}.{name}" (for TensorFlow internal registration, the package
    name is `tf`)

## Checkpoint saver registration

If `Trackables` share state or require complicated coordination between multiple
`Trackables` (e.g. `DTensor`), then users may register a save and restore
functions for these objects.

```
tf.saved_model.register_checkpoint_saver(
    predicate, save_fn=None, restore_fn=None):
```

*   `predicate`: A function that returns `True` if a `Trackable` object should
    be saved using the registered `save_fn` or `restore_fn`.
*   `save_fn`: A python function or `tf.function` or `None`. If `None`, run the
    default saving process which calls `Trackable._serialize_to_tensors`.
*   `restore_fn`: A `tf.function` or `None`. If `None`, run the default
    restoring process which calls `Trackable._restore_from_tensors`.

**`save_fn` details**

```
@tf.function  # optional decorator
def save_fn(trackables, file_prefix): -> List[shard filenames]
```

*   `trackables`: A dictionary of `{object_prefix: Trackable}`. The
    object_prefix can be used as the object names, and uniquely identify each
    `Trackable`. `trackables` is the filtered set of trackables that pass the
    predicate.
*   `file_prefix`: A string or string tensor of the checkpoint prefix.
*   `shard filenames`: A list of filenames written using `io_ops.save_v2`, which
    will be merged into the checkpoint data files. These should be prefixed by
    `file_prefix`.

This function can be a python function, in which case shard filenames can be an
empty list (if the values are written without the `SaveV2` op).

If this function is a `tf.function`, then the shards must be written using the
SaveV2 op. This guarantees the checkpoint format is compatible with existing
checkpoint readers and managers.

**`restore_fn` details**

```
@tf.function  # required decorator
def restore_fn(trackables, file_prefix): -> None
```

A `tf.function` with the spec:

*   `trackables`: A dictionary of `{object_prefix: Trackable}`. The
    `object_prefix` can be used as the object name, and uniquely identifies each
    Trackable. The Trackable objects are the filtered results of the registered
    predicate.
*   `file_prefix`: A string or string tensor of the checkpoint prefix.

**Why are restore functions required to be a `tf.function`?** The short answer
is, the SavedModel format must maintain the invariant that SavedModel packages
can be used for inference on any platform and language. SavedModel inference
needs to be able to restore checkpointed values, so the restore function must be
directly encoded into the SavedModel in the Graph. We also have security
measures over FunctionDef and GraphDef, so users can check that the SavedModel
will not run arbitrary code (a feature of `saved_model_cli`).

## Example

Below shows a `Stack` module that contains multiple `Parts` (a subclass of
`tf.Variable`). When a `Stack` is saved to a checkpoint, the `Parts` are stacked
together and a single entry in the checkpoint is created. The checkpoint value
is restored to all of the `Parts` in the `Stack`.

```
@registration.register_serializable()
class Part(resource_variable_ops.ResourceVariable):

  def __init__(self, value):
    self._init_from_args(value)

  @classmethod
  def _deserialize_from_proto(cls, **kwargs):
    return cls([0, 0])


@registration.register_serializable()
class Stack(tracking.AutoTrackable):

  def __init__(self, parts=None):
    self.parts = parts

  @def_function.function(input_signature=[])
  def value(self):
    return array_ops.stack(self.parts)


def get_tensor_slices(trackables):
  tensor_names = []
  shapes_and_slices = []
  tensors = []
  restored_trackables = []
  for obj_prefix, obj in trackables.items():
    if isinstance(obj, Part):
      continue  # only save stacks
    tensor_names.append(obj_prefix + "/value")
    shapes_and_slices.append("")
    x = obj.value()
    with ops.device("/device:CPU:0"):
      tensors.append(array_ops.identity(x))
    restored_trackables.append(obj)

  return tensor_names, shapes_and_slices, tensors, restored_trackables


def save_stacks_and_parts(trackables, file_prefix):
  """Save stack and part objects to a checkpoint shard."""
  tensor_names, shapes_and_slices, tensors, _ = get_tensor_slices(trackables)
  io_ops.save_v2(file_prefix, tensor_names, shapes_and_slices, tensors)
  return file_prefix


def restore_stacks_and_parts(trackables, merged_prefix):
  tensor_names, shapes_and_slices, tensors, restored_trackables = (
      get_tensor_slices(trackables))
  dtypes = [t.dtype for t in tensors]
  restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                       shapes_and_slices, dtypes)
  for trackable, restored_tensor in zip(restored_trackables, restored_tensors):
    expected_shape = trackable.value().get_shape()
    restored_tensor = array_ops.reshape(restored_tensor, expected_shape)
    parts = array_ops.unstack(restored_tensor)
    for part, restored_part in zip(trackable.parts, parts):
      part.assign(restored_part)


registration.register_checkpoint_saver(
    name="stacks",
    predicate=lambda x: isinstance(x, (Stack, Part)),
    save_fn=save_stacks_and_parts,
    restore_fn=restore_stacks_and_parts)
```
