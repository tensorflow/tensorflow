# TensorFlow Lite Android Support Library

Mobile application developers typically interact with typed objects such as
bitmaps or primitives such as integers. However, the TensorFlow Lite Interpreter
that runs the on-device machine learning model uses tensors in the form of
ByteBuffer, which can be difficult to debug and manipulate. The TensorFlow Lite
Android Support Library is designed to help process the input and output of
TensorFlow Lite models, and make the TensorFlow Lite interpreter easier to use.

We welcome feedback from the community as we develop this support library,
especially around:

*   Use-cases we should support including data types and operations
*   Ease of use - does the APIs make sense to the community

See the [documentation](https://www.tensorflow.org/lite/guide/lite_support) for
instruction and examples.
