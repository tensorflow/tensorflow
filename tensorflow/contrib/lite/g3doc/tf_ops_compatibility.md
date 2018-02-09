# TensorFlow Compatibility Guide

TensorFlow Lite supports a number of TensorFlow operations used in common
inference models. As they are processed by the TensorFlow Lite Optimizing
Converter, those operations may be elided or fused, before the supported
operations are mapped to their TensorFlow Lite counterparts.

Since the set of TensorFlow Lite operations is smaller than TensorFlow's, not
every model is convertible. Even for supported operations, very specific usage
patterns are sometimes expected, for performance reasons. We expect to expand
the set of supported operations in future TensorFlow Lite releases.

The best way to understand how to build a TensorFlow model that can be used with
TensorFlow Lite is to carefully consider how operations are converted and
optimized, along with the limitations imposed by this process.

## Supported Types

Most TensorFlow Lite operations target both floating-point (float32) and
quantized (uint8) inference, but usually there is little or no support for other
types like tf.float16 and strings.

Apart from using different version of the operations, the other difference
between floating-point and quantized models lies in the way they are converted.
Quantized conversion expect the models to be annotated with "fake quantization"
nodes that record the dynamic range of the tensors. Without that information TF
Lite is not able to accurately quantize a model, which means that proper
quantized training is necessary before conversion.

## Data Format and Broadcasting

At the moment TensorFlow Lite supports only TensorFlow's "NHWC" format, and
broadcasting in operations like tf.add and tf.mul is generally not supported.

## Compatible Operations

The following TensorFlow operations are usually mapped to their TensorFlow Lite
counterparts:

*   [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul) - *as long
    as the second argument is constant and transposition is not used*
*   [tf.nn.avg_pool](https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool)
*   [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d) -
    *as long as the filter is constant*
*   [tf.nn.depthwise_conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d) -
    *as long as the filter is constant and rate is [1,1]*
*   [tf.nn.l2_normalize](https://www.tensorflow.org/api_docs/python/tf/nn/l2_normalize) -
    *as long as normalization is done along the last dimension*
*   [tf.nn.local_response_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)
*   [tf.nn.max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)
*   [tf.nn.softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) -
    *as long as tensors are 2D and axis is the last dimension*
*   [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape)
*   [tf.sigmoid](https://www.tensorflow.org/api_docs/python/tf/sigmoid)
*   [tf.space_to_depth](https://www.tensorflow.org/api_docs/python/tf/space_to_depth)

## Straightforward Conversions, Constant-Folding and Fusing

A number of TensorFlow operations can be processed by TensorFlow Lite even
though they have no direct equivalent. This is the case for operations that can
be simply removed from the graph (tf.identity), replaced by tensors
(tf.placeholder), or fused into more complex operations (tf.nn.bias_add). Even
some supported operations may sometimes be removed through one of these
processes.

Here is a list of TensorFlow operations that are usually removed from the graph:

*   [tf.add](https://www.tensorflow.org/api_docs/python/tf/add)
*   [tf.check_numerics](https://www.tensorflow.org/api_docs/python/tf/check_numerics)
*   [tf.constant](https://www.tensorflow.org/api_docs/python/tf/constant)
*   [tf.div](https://www.tensorflow.org/api_docs/python/tf/div)
*   [tf.divide](https://www.tensorflow.org/api_docs/python/tf/divide)
*   [tf.fake_quant_with_min_max_args](https://www.tensorflow.org/api_docs/python/tf/fake_quant_with_min_max_args)
*   [tf.fake_quant_with_min_max_vars](https://www.tensorflow.org/api_docs/python/tf/fake_quant_with_min_max_vars)
*   [tf.greater](https://www.tensorflow.org/api_docs/python/tf/greater)
*   [tf.greater_equal](https://www.tensorflow.org/api_docs/python/tf/greater_equal)
*   [tf.identity](https://www.tensorflow.org/api_docs/python/tf/identity)
*   [tf.less](https://www.tensorflow.org/api_docs/python/tf/less)
*   [tf.less_equal](https://www.tensorflow.org/api_docs/python/tf/less_equal)
*   [tf.maximum](https://www.tensorflow.org/api_docs/python/tf/maximum)
*   [tf.minimum](https://www.tensorflow.org/api_docs/python/tf/minimum)
*   [tf.multiply](https://www.tensorflow.org/api_docs/python/tf/multiply)
*   [tf.no_op](https://www.tensorflow.org/api_docs/python/tf/no_op)
*   [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
*   [tf.placeholder_with_default](https://www.tensorflow.org/api_docs/python/tf/placeholder_with_default)
*   [tf.realdiv](https://www.tensorflow.org/api_docs/python/tf/realdiv)
*   [tf.reduce_max](https://www.tensorflow.org/api_docs/python/tf/reduce_max)
*   [tf.reduce_min](https://www.tensorflow.org/api_docs/python/tf/reduce_min)
*   [tf.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/reduce_sum)
*   [tf.rsqrt](https://www.tensorflow.org/api_docs/python/tf/rsqrt)
*   [tf.shape](https://www.tensorflow.org/api_docs/python/tf/shape)
*   [tf.sqrt](https://www.tensorflow.org/api_docs/python/tf/sqrt)
*   [tf.square](https://www.tensorflow.org/api_docs/python/tf/square)
*   [tf.squeeze](https://www.tensorflow.org/api_docs/python/tf/squeeze)
*   [tf.subtract](https://www.tensorflow.org/api_docs/python/tf/subtract)
*   [tf.tile](https://www.tensorflow.org/api_docs/python/tf/tile)
*   [tf.nn.batch_norm_with_global_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_norm_with_global_normalization)
*   [tf.nn.bias_add](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add)
*   [tf.nn.fused_batch_norm](https://www.tensorflow.org/api_docs/python/tf/nn/fused_batch_norm)
*   [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)
*   [tf.nn.relu6](https://www.tensorflow.org/api_docs/python/tf/nn/relu6)

Note that many of those operations don't have TensorFlow Lite equivalents and
the corresponding model will not be convertible if they can't be elided or
fused.

## Unsupported Operations

TensorFlow operation not listed above are likely unsupported. Notably, the
following common ops are not supported at the moment:

*   [tf.batch_to_space_nd](https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd)
*   [tf.depth_to_space](https://www.tensorflow.org/api_docs/python/tf/depth_to_space)
*   [tf.floor](https://www.tensorflow.org/api_docs/python/tf/floor)
*   [tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather)
*   [tf.image.resize_bilinear](https://www.tensorflow.org/api_docs/python/tf/image/resize_bilinear)
*   [tf.pad](https://www.tensorflow.org/api_docs/python/tf/pad)
*   [tf.reduce_mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)
*   [tf.slice](https://www.tensorflow.org/api_docs/python/tf/slice)
*   [tf.space_to_batch_nd](https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd)
*   [tf.split](https://www.tensorflow.org/api_docs/python/tf/split)
*   [tf.strided_slice](https://www.tensorflow.org/api_docs/python/tf/strided_slice)
*   [tf.tanh](https://www.tensorflow.org/api_docs/python/tf/tanh)

## TensorFlow Lite Operations

The following TensorFlow Lite operations are fully supported and used in place
of the TensorFlow operations listed above:

**ADD**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: elementwise sum of the input tensors
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
}
```

**AVERAGE_POOL_2D**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor where each entry is the mean of the input values in the
     corresponding window.
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
  padding: SAME|VALID
  stride_w,stride_h: stride of the sliding window
  filter_width,filter_height: size of the sliding window
}
```

**CONCATENATION**

```
Inputs {
  0-N: any number of tensors
}
Outputs {
  0: concatenation of the input tensors along the given axis.
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
  axis: dimension along which the concatenation is performed
}
```

**CONV_2D**

```
Inputs {
  0: 4D tensor
  1: filter
  2: bias (optional)
}
Outputs {
  0: result of 2D convolution of the input tensor
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
  padding: SAME|VALID
  stride_w,stride_h: stride of the filter window
}
```

**DEPTHWISE_CONV_2D**

```
Inputs {
  0: 4D tensor
  1: filter
  2: bias (optional)
}
Outputs {
  0: result of a depthwise-2D convolution of the input tensor
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
  padding: SAME|VALID
  stride_w,stride_h: stride of the filter window
  depth_multiplier: relation between the last dimension of the input and output
    tensors
}
```

**FULLY_CONNECTED**

```
Inputs {
  0: 4D tensor
  1: filter
  2: bias (optional)
}
Outputs {
  0: output of a fully (densely) connected layer, which connects all
     elements in the input tensor with each element in this tensor.
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
}
```

**L2_NORMALIZATION**

```
Inputs {
  0: input tensor
}
Outputs {
  0: normalized tensor (along the last dimension)
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
}
```

**L2_POOL_2D**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to tf.sqrt(tf.nn.ave_pool(tf.square(input))
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
  padding: SAME|VALID
  stride_w,stride_h: stride of the sliding window
  filter_width,filter_height: size of the sliding window
}
```

**LOCAL_RESPONSE_NORMALIZATION**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to tf.nn.local_response_normalization
}
Options {
  radius
  bias
  alpha
  beta
}
```

**LOGISTIC**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to 1 / (1 + exp(-input))
}
```

**MAX_POOL_2D**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor where each entry is the maximum of the input values in the
     corresponding window.
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
  padding: SAME|VALID
  stride_w,stride_h: stride of the sliding window
  filter_width,filter_height: size of the sliding window
}
```

**MUL**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: elementwise multiplication of the input tensors
}
Options {
  fused_activation_function:  NONE|RELU|RELU6
}
```

**RELU**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to max(0, input)
}
```

**RELU_N1_TO_1**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to max(-1, min(input, 1)
}
```

**RELU6**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to max(0, min(input, 6)
}
```

**RESHAPE**

```
Inputs {
  0: a tensor
  1: ignored
}
Outputs {
  0: a tensor with the same elements as the input but with the new shape
}
Options {
  new_shape
}
```

**SOFTMAX**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to exp(input) / tf.reduce_sum(exp(input * beta), dim),
     where dim is always the last dimension of the input tensor.
}
Options {
  beta
}
```

**SPACE_TO_DEPTH**

```
Inputs {
  0: a 4D tensor
}
Outputs {
  0: a tensor rearranged using block_size. See tf.space_to_depth for details.
}
Options {
  block_size
}
```

And these are TensorFlow Lite operations that are present but not ready for
custom models yet:

*   CALL
*   CONCAT_EMBEDDINGS
*   CUSTOM
*   EMBEDDING_LOOKUP
*   EMBEDDING_LOOKUP_SPARSE
*   HASHTABLE_LOOKUP
*   LSH_PROJECTION
*   LSTM
*   RESIZE_BILINEAR
*   RNN
*   SKIP_GRAM
*   SVDF
*   TANH
