A class representing a type of model export.

Typically constructed by a utility function specific to the exporter, such as
`saved_model_export_utils.make_export_strategy()`.

The fields are:
  name: The directory name under the export base directory where exports of
    this type will be written.
  export_fn: A function that writes an export, given an estimator and a
    destination path.  This may be run repeatedly during continuous training,
    or just once at the end of fixed-length training.
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

#### `tf.contrib.learn.ExportStrategy.export(estimator, export_path)` {#ExportStrategy.export}




- - -

#### `tf.contrib.learn.ExportStrategy.export_fn` {#ExportStrategy.export_fn}

Alias for field number 1


- - -

#### `tf.contrib.learn.ExportStrategy.name` {#ExportStrategy.name}

Alias for field number 0


