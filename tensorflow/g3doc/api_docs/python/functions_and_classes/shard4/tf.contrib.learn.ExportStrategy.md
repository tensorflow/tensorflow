A class representing a type of model export.

Typically constructed by a utility function specific to the exporter, such as
`saved_model_export_utils.make_export_strategy()`.

The fields are:
  name: The directory name under the export base directory where exports of
    this type will be written.
  export_fn: A function that writes an export, given an estimator, a
    destination path, and optionally a checkpoint path and an evaluation
    result for that checkpoint.  This export_fn() may be run repeatedly during
    continuous training, or just once at the end of fixed-length training.
    Note the export_fn() may choose whether or not to export based on the eval
    result or based on an internal timer or any other criterion, if exports
    are not desired for every checkpoint.

    The signature of this function must be one of:
      * (estimator, export_path) -> export_path`
      * (estimator, export_path, checkpoint_path) -> export_path`
      * (estimator, export_path, checkpoint_path, eval_result) -> export_path`
- - -

#### `tf.contrib.learn.ExportStrategy.__getnewargs__()` {#ExportStrategy.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.contrib.learn.ExportStrategy.__getstate__()` {#ExportStrategy.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.contrib.learn.ExportStrategy.__new__(_cls, name, export_fn)` {#ExportStrategy.__new__}

Create new instance of ExportStrategy(name, export_fn)


- - -

#### `tf.contrib.learn.ExportStrategy.__repr__()` {#ExportStrategy.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.contrib.learn.ExportStrategy.export(estimator, export_path, checkpoint_path=None, eval_result=None)` {#ExportStrategy.export}

Exports the given Estimator to a specific format.

##### Args:


*  <b>`estimator`</b>: the Estimator to export.
*  <b>`export_path`</b>: A string containing a directory where to write the export.
*  <b>`checkpoint_path`</b>: The checkpoint path to export.  If None (the default),
    the strategy may locate a checkpoint (e.g. the most recent) by itself.
*  <b>`eval_result`</b>: The output of Estimator.evaluate on this checkpoint.  This
    should be set only if checkpoint_path is provided (otherwise it is
    unclear which checkpoint this eval refers to).

##### Returns:

  The string path to the exported directory.

##### Raises:


*  <b>`ValueError`</b>: if the export_fn does not have the required signature


- - -

#### `tf.contrib.learn.ExportStrategy.export_fn` {#ExportStrategy.export_fn}

Alias for field number 1


- - -

#### `tf.contrib.learn.ExportStrategy.name` {#ExportStrategy.name}

Alias for field number 0


