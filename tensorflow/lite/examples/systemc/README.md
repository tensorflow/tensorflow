# ATTENTION

Bazel bug is preventing this to compile with multiple jobs.

See `https://github.com/bazelbuild/bazel/issues/10384`

Please compile and run with:

```bash
bazel build --jobs 1 //tensorflow/lite/examples/systemc:hello_systemc
bazel run //tensorflow/lite/examples/systemc:hello_systemc
```

# Example on how to use SystemC with tflite libraries

This is a simple example on how to use SystemC from within tflite build

```bash
# Hello World example
bazel run //tensorflow/lite/examples/systemc:hello_systemc

# GTest example
bazel test //tensorflow/lite/examples/systemc:sc_example_test

# Multi-file example
bazel run //tensorflow/lite/examples/systemc:hello_channel
```
