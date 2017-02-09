### `tf.contrib.learn.make_export_strategy(serving_input_fn, default_output_alternative_key=None, assets_extra=None, as_text=False, exports_to_keep=5)` {#make_export_strategy}

Create an ExportStrategy for use with Experiment.

##### Args:


*  <b>`serving_input_fn`</b>: A function that takes no arguments and returns an
    `InputFnOps`.
*  <b>`default_output_alternative_key`</b>: the name of the head to serve when an
    incoming serving request does not explicitly request a specific head.
    Not needed for single-headed models.
*  <b>`assets_extra`</b>: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
*  <b>`as_text`</b>: whether to write the SavedModel proto in text format.
*  <b>`exports_to_keep`</b>: Number of exports to keep.  Older exports will be
    garbage-collected.  Defaults to 5.  Set to None to disable garbage
    collection.

##### Returns:

  An ExportStrategy that can be passed to the Experiment constructor.

