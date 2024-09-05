# SavedModel Image Format

Everything related to the SavedModel Image format belongs in this directory.
If you are a TensorFlow Python user, you can try this format by setting the
`experimental_image_format` option:

```
tf.savedmodel.save(
    model, path,
    options=tf.saved_model.SaveOptions(experimental_image_format=True)
)
```

When this option is enabled, exported SavedModels with proto size > 2GB will
automatically save with the new format (`.cpb` instead of `.pb`).

<!-- **Compatibility** -->

The official TF APIs (TF1/TF2 python or C++ loading) have already been
integrated to handle the new format, but some downstream converters may not
have been updated.
