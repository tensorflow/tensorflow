# TensorFlow Lite Android Wrapper Code Generator

For TensorFlow Lite model enhanced with [metadata](https://www.tensorflow.org/lite/convert/metadata.md),
developers can use the TensorFlow Lite Android wrapper code generator to create
platform specific wrapper code. The wrapper code removes the need to interact
directly with `ByteBuffer`. Instead, developers can interact with the TensorFlow
Lite model with typed objects such as `Bitmap` and `Rect`.

The usefulness of the code generator depend on the completeness of the
TensorFlow Lite model's metadata entry. Refer to the `<Codegen usage>` section
under relevant fields in
[metadata_schema.fbs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs),
to see how the codegen tool parses each field.
