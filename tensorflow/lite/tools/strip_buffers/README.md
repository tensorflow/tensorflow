# TFLite Buffer-Stripping Tool/Library

**NOTE: This is an advanced tool used to reduce bandwidth usage in Neural
Architecture Search applications. Use with caution.**

The tools in this directory make it easier to distribute TFLite models to
multiple devices over networks with the sole aim of benchmarking *latency*
performance. The intended workflow is as follows:

*   The stripping tool empties eligible constants from a TFLite flatbuffer to
    reduce its size.
*   This lean model can be easily transported to devices over a network.
*   The reconstitution tool on the device takes in a flatbuffer in memory, and
    fills in the appropriate buffers with random data.

As an example, see the before/after sizes for MobileNetV1:

*   Float: 16.9MB -> 12KB
*   Quantized: 4.3MB -> 17.6 KB

**NOTE: This tool only supports single subgraphs for now.**

There are two tools in this directory:

## 1. Stripping buffers out of TFLite flatbuffers

This tool takes in an input `flatbuffer`, and strips out (or 'empties') the
buffers (constant data) for tensors that follow the following guidelines:

*   Are either of: Float32, Int32, UInt8, Int8
*   If Int32, the tensor should have a min of 10 elements

The second rule above protects us from invalidating constant data that cannot be
randomised (for example, Reshape 'shape' input).

To run the associated script:

```
bazel run -c opt tensorflow/lite/tools/strip_buffers:strip_buffers_from_fb -- --input_flatbuffer=/input/path.tflite --output_flatbuffer=/output/path.tflite
```

## 2. Stripping buffers out of TFLite flatbuffers

The idea here is to reconstitute the lean flatbuffer `Model` generared in the
above step, by filling in random data whereever necessary.

The prototype script can be called as:

```
bazel run -c opt tensorflow/lite/tools/strip_buffers:reconstitute_buffers_into_fb -- --input_flatbuffer=/input/path.tflite --output_flatbuffer=/output/path.tflite
```

## C++ Library

Both the above tools are present as `stripping_lib` in this directory, which
mutate the flatbuffer(s) in-memory. This ensures we can do the above two steps
without touching the filesystem again.
