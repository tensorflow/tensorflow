# TFLite Quantize Weights Tool

## Recommended usage

The Quantize Weights transformation is integrated with
[tflite_convert](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/cmdline_reference.md#transformation-flags).

The recommended way of invoking this tool is by simply adding the
`--post_training_quantize` flag to your original tflite_convert invocation. For
example,

```
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --saved_model_dir=/tmp/saved_model \
  --post_training_quantize
```

## Overview

The Quantize Weights tool provides a simple way to quantize the weights for a
float TFLite model.

TODO(raghuramank): Add link to weight quantization tutorial.

### Size reduction

float32 weights will be converted to 8 bit integers. This results in a model
that is around 1/4th the size of the original model.

### Latency reduction

TFLite also has "hybrid" kernels implemented for many operations. These "hybrid"
kernels take 8 bit integer weights and float inputs, dynamically quantize the
inputs tensor (based on the input tensor's min and max elements), and does
computations using the 8 bit integer values. This results in a 2-4x reduction in
latency for "hybrid" kernels. In this mode the inference type is still FLOAT
since the inputs and output to each operation is still float.

For operations that do not yet have "hybrid" kernels implemented, we introduce a
Dequantize operation after 8 bit integer weights. These convert weights back to
float32 during inference to allow original float32 kernels to run. Since we
cache dequantized results, the result of each of this dequantized path will be
on-par with the original float model.

TODO(yunluli): Fill in latency results from latency experiments.

### Accuracy

Since this technique quantizes weights after the model has already been trained,
there can be accuracy drops depending on the model. For common CNN networks, the
observed accuracy drops are small and can be seen below.

TODO(yunluli): Fill in accuracy results from accuracy experiments.

## Direct usage

One can also invoke the Quantize Weights directly via C++ if they have a float
`::tflite::Model` that they want to convert. They must provide a
`flatbuffers::FlatBufferBuilder` which owns the underlying buffer of the created
model. Here is an example invocation:

```
::tflite::Model* input_model = ...;
flatbuffers::FlatBufferBuilder builder;
TfLiteStatus status = ::tflite::optimize::QuantizeWeights(&builder, input_model);
CHECK(status, kTfLiteStatusOk);
const uint8_t* buffer = builder->GetBufferPointer();
tflite::Model* output_model = ::tflite::GetModel(buffer);
```
