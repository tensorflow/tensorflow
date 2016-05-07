Saves and restores variables.

See [Variables](../../how_tos/variables/index.md)
for an overview of variables, saving and restoring.

The `Saver` class adds ops to save and restore variables to and from
*checkpoints*.  It also provides convenience methods to run these ops.

Checkpoints are binary files in a proprietary format which map variable names
to tensor values.  The best way to examine the contents of a checkpoint is to
load it using a `Saver`.

Savers can automatically number checkpoint filenames with a provided counter.
This lets you keep multiple checkpoints at different steps while training a
model.  For example you can number the checkpoint filenames with the training
step number.  To avoid filling up disks, savers manage checkpoint files
automatically. For example, they can keep only the N most recent files, or
one checkpoint for every N hours of training.

You number checkpoint filenames by passing a value to the optional
`global_step` argument to `save()`:

```python
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
```

Additionally, optional arguments to the `Saver()` constructor let you control
the proliferation of checkpoint files on disk:

* `max_to_keep` indicates the maximum number of recent checkpoint files to
  keep.  As new files are created, older files are deleted.  If None or 0,
  all checkpoint files are kept.  Defaults to 5 (that is, the 5 most recent
  checkpoint files are kept.)

* `keep_checkpoint_every_n_hours`: In addition to keeping the most recent
  `max_to_keep` checkpoint files, you might want to keep one checkpoint file
  for every N hours of training.  This can be useful if you want to later
  analyze how a model progressed during a long training session.  For
  example, passing `keep_checkpoint_every_n_hours=2` ensures that you keep
  one checkpoint file for every 2 hours of training.  The default value of
  10,000 hours effectively disables the feature.

Note that you still have to call the `save()` method to save the model.
Passing these arguments to the constructor will not save variables
automatically for you.

A training program that saves regularly looks like:

```python
...
# Create a saver.
saver = tf.train.Saver(...variables...)
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.Session()
for step in xrange(1000000):
    sess.run(..training_op..)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)
```

In addition to checkpoint files, savers keep a protocol buffer on disk with
the list of recent checkpoints. This is used to manage numbered checkpoint
files and by `latest_checkpoint()`, which makes it easy to discover the path
to the most recent checkpoint. That protocol buffer is stored in a file named
'checkpoint' next to the checkpoint files.

If you create several savers, you can specify a different filename for the
protocol buffer file in the call to `save()`.

- - -

#### `tf.train.Saver.__init__(var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, saver_def=None, builder=None)` {#Saver.__init__}

Creates a `Saver`.

The constructor adds ops to save and restore variables.

`var_list` specifies the variables that will be saved and restored. It can
be passed as a `dict` or a list:

* A `dict` of names to variables: The keys are the names that will be
  used to save or restore the variables in the checkpoint files.
* A list of variables: The variables will be keyed with their op name in
  the checkpoint files.

For example:

```python
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
```

The optional `reshape` argument, if `True`, allows restoring a variable from
a save file where the variable had a different shape, but the same number
of elements and type.  This is useful if you have reshaped a variable and
want to reload it from an older checkpoint.

The optional `sharded` argument, if `True`, instructs the saver to shard
checkpoints per device.

##### Args:


*  <b>`var_list`</b>: A list of `Variable` objects or a dictionary mapping names to
    variables.  If `None`, defaults to the list of all variables.
*  <b>`reshape`</b>: If `True`, allows restoring parameters from a checkpoint
    where the variables have a different shape.
*  <b>`sharded`</b>: If `True`, shard the checkpoints, one per device.
*  <b>`max_to_keep`</b>: Maximum number of recent checkpoints to keep.
    Defaults to 5.
*  <b>`keep_checkpoint_every_n_hours`</b>: How often to keep checkpoints.
    Defaults to 10,000 hours.
*  <b>`name`</b>: String.  Optional name to use as a prefix when adding operations.
*  <b>`restore_sequentially`</b>: A `Bool`, which if true, causes restore of different
    variables to happen sequentially within each device.  This can lower
    memory usage when restoring very large models.
*  <b>`saver_def`</b>: Optional `SaverDef` proto to use instead of running the
    builder. This is only useful for specialty code that wants to recreate
    a `Saver` object for a previously built `Graph` that had a `Saver`.
    The `saver_def` proto should be the one returned by the
    `as_saver_def()` call of the `Saver` that was created for that `Graph`.
*  <b>`builder`</b>: Optional `SaverBuilder` to use if a `saver_def` was not provided.
    Defaults to `BaseSaverBuilder()`.

##### Raises:


*  <b>`TypeError`</b>: If `var_list` is invalid.
*  <b>`ValueError`</b>: If any of the keys or values in `var_list` are not unique.


- - -

#### `tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True)` {#Saver.save}

Saves variables.

This method runs the ops added by the constructor for saving variables.
It requires a session in which the graph was launched.  The variables to
save must also have been initialized.

The method returns the path of the newly created checkpoint file.  This
path can be passed directly to a call to `restore()`.

##### Args:


*  <b>`sess`</b>: A Session to use to save the variables.
*  <b>`save_path`</b>: String.  Path to the checkpoint filename.  If the saver is
    `sharded`, this is the prefix of the sharded checkpoint filename.
*  <b>`global_step`</b>: If provided the global step number is appended to
    `save_path` to create the checkpoint filename. The optional argument
    can be a `Tensor`, a `Tensor` name or an integer.
*  <b>`latest_filename`</b>: Optional name for the protocol buffer file that will
    contains the list of most recent checkpoint filenames.  That file,
    kept in the same directory as the checkpoint files, is automatically
    managed by the saver to keep track of recent checkpoints.  Defaults to
    'checkpoint'.
*  <b>`meta_graph_suffix`</b>: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
*  <b>`write_meta_graph`</b>: `Boolean` indicating whether or not to write the meta
    graph file.

##### Returns:

  A string: path at which the variables were saved.  If the saver is
    sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'
    is the number of shards created.

##### Raises:


*  <b>`TypeError`</b>: If `sess` is not a `Session`.
*  <b>`ValueError`</b>: If `latest_filename` contains path components.


- - -

#### `tf.train.Saver.restore(sess, save_path)` {#Saver.restore}

Restores previously saved variables.

This method runs the ops added by the constructor for restoring variables.
It requires a session in which the graph was launched.  The variables to
restore do not have to have been initialized, as restoring is itself a way
to initialize variables.

The `save_path` argument is typically a value previously returned from a
`save()` call, or a call to `latest_checkpoint()`.

##### Args:


*  <b>`sess`</b>: A `Session` to use to restore the parameters.
*  <b>`save_path`</b>: Path where parameters were previously saved.

##### Raises:


*  <b>`ValueError`</b>: If the given `save_path` does not point to a file.



Other utility methods.

- - -

#### `tf.train.Saver.last_checkpoints` {#Saver.last_checkpoints}

List of not-yet-deleted checkpoint filenames.

You can pass any of the returned values to `restore()`.

##### Returns:

  A list of checkpoint filenames, sorted from oldest to newest.


- - -

#### `tf.train.Saver.set_last_checkpoints(last_checkpoints)` {#Saver.set_last_checkpoints}

DEPRECATED: Use set_last_checkpoints_with_time.

Sets the list of old checkpoint filenames.

##### Args:


*  <b>`last_checkpoints`</b>: A list of checkpoint filenames.

##### Raises:


*  <b>`AssertionError`</b>: If last_checkpoints is not a list.


- - -

#### `tf.train.Saver.as_saver_def()` {#Saver.as_saver_def}

Generates a `SaverDef` representation of this saver.

##### Returns:

  A `SaverDef` proto.



#### Other Methods
- - -

#### `tf.train.Saver.export_meta_graph(filename=None, collection_list=None, as_text=False)` {#Saver.export_meta_graph}

Writes `MetaGraphDef` to save_path/filename.

##### Args:


*  <b>`filename`</b>: Optional meta_graph filename including the path.
*  <b>`collection_list`</b>: List of string keys to collect.
*  <b>`as_text`</b>: If `True`, writes the meta_graph as an ASCII proto.

##### Returns:

  A `MetaGraphDef` proto.


- - -

#### `tf.train.Saver.from_proto(saver_def)` {#Saver.from_proto}

Returns a `Saver` object created from `saver_def`.


- - -

#### `tf.train.Saver.set_last_checkpoints_with_time(last_checkpoints_with_time)` {#Saver.set_last_checkpoints_with_time}

Sets the list of old checkpoint filenames and timestamps.

##### Args:


*  <b>`last_checkpoints_with_time`</b>: A list of tuples of checkpoint filenames and
    timestamps.

##### Raises:


*  <b>`AssertionError`</b>: If last_checkpoints_with_time is not a list.


- - -

#### `tf.train.Saver.to_proto()` {#Saver.to_proto}

Converts this `Saver` to a `SaverDef` protocol buffer.

##### Returns:

  A `SaverDef` protocol buffer.


