
# TensorFlow Lite & TensorFlow Compatibility Guide

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
broadcasting is only support in a limited number of ops (tf.add, tf.mul, tf.sub,
and tf.div).

## Compatible Operations

The following TensorFlow operations are usually mapped to their TensorFlow Lite
counterparts:

*   [tf.batch_to_space_nd](https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd) -
    *as long as the input tensor is 4D (1 batch + 2 spatial + 1 other) and the
    crops attribute is not used*
*   [tf.exp](https://www.tensorflow.org/api_docs/python/tf/exp)
*   [tf.fake_quant*](https://www.tensorflow.org/api_docs/python/tf/fake_quant_with_min_max_args)
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
*   [tf.nn.log_softmax](https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax) -
    *as long as axis is not provided*
*   [tf.nn.max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)
*   [tf.nn.softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) -
    *as long as tensors are 2D and axis is the last dimension*
*   [tf.nn.top_k](https://www.tensorflow.org/api_docs/python/tf/nn/top_k)
*   [tf.one_hot](https://www.tensorflow.org/api_docs/python/tf/one_hot)
*   [tf.pad](https://www.tensorflow.org/api_docs/python/tf/pad) - *as long as
    mode and constant_values are not used*
*   [tf.reduce_mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean) -
    *as long as the reduction_indices attribute is not used*
*   [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape)
*   [tf.sigmoid](https://www.tensorflow.org/api_docs/python/tf/sigmoid)
*   [tf.space_to_batch_nd](https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd) -
    *as long as the input tensor is 4D (1 batch + 2 spatial + 1 other)*
*   [tf.space_to_depth](https://www.tensorflow.org/api_docs/python/tf/space_to_depth)
*   [tf.split](https://www.tensorflow.org/api_docs/python/tf/split) - *as long
    as num is not provided and num_or_size_split contains number of splits as a
    0D tensor*
*   [tf.squeeze](https://www.tensorflow.org/api_docs/python/tf/squeeze) - *as
    long as axis is not provided*
*   [tf.strided_slice](https://www.tensorflow.org/api_docs/python/tf/strided_slice) -
    *as long as ellipsis_mask and new_axis_mask are not used*
*   [tf.transpose](https://www.tensorflow.org/versions/master/api_docs/python/tf/transpose) -
    *as long as conjugate is not used*

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
*   [tf.identity](https://www.tensorflow.org/api_docs/python/tf/identity)
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

*   [tf.depth_to_space](https://www.tensorflow.org/api_docs/python/tf/depth_to_space)
*   [tf.image.resize_bilinear](https://www.tensorflow.org/api_docs/python/tf/image/resize_bilinear)
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

**BATCH_TO_SPACE_ND**

```
Inputs {
  0: 4D tensor
  1: 1D tensor
  2: 2D tensor
}
Outputs {
  0: tensor rearranged using block_shape. See tf.batch_to_space_nd for
     details.
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

**CONV_2D_TRANSPOSE**

```
Inputs {
  0: output_shape
  1: filter
  2: 4D tensor
}
Outputs {
  0: the transpose (gradient) of conv2d
}
Options {
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

**EQUAL**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: a tensor of type bool, true whenever an element of the first tensor is
  equal to the corresponding element of the second tensor.
}
```

**EXP**

```
Inputs {
  0: tensor
}
Outputs {
  0: result of computing element-wise exponential of the input tensor
}
```

**FLOOR**

```
inputs {
  0: tensor
}
outputs: {
  0: result of computing element-wise floor of the input tensor
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

**GATHER**

```
Inputs {
  0: params tensor
  1: indices tensor
  2: axis tensor (optional)
}
Outputs {
  0: a tensor with same type as the params tensor.
}
```

**GREATER**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: a tensor of type bool, true whenever an element of the first tensor is
  greater than the corresponding element of the second tensor.
}
```

**GREATER_EQUAL**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: a tensor of type bool, true whenever an element of the first tensor is
  greater than or equal to the corresponding element of the second tensor.
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

**LESS**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: a tensor of type bool, true whenever an element of the first tensor is less
  than the corresponding element of the second tensor.
}
```

**LESS_EQUAL**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: a tensor of type bool, true whenever an element of the first tensor is less
  than or equal to the corresponding element of the second tensor.
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

**LOG**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a tensor equivalent to log(input)
}
```

**LOG_SOFTMAX**

```
Inputs {
  0: tensor
}
Outputs {
  0: tensor equivalent to logits - log(reduce_sum(exp(logits), -1))
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

**NEG**

```
Inputs {
  0: a tensor
}
Outputs {
  0: elementwise negation of the input tensor
}
```

**PAD**

```
Inputs {
  0: tensor
  1: tensor
}
Outputs {
  0: tensor where additional values are added before and after the contents of
     each dimension
}
```

**MEAN (tf.reduce_mean)**

```
Inputs {
  0: tensor
  1: tensor
}
Outputs {
  0: tensor containing the mean of the elements
}
Options {
  keep_dims: whether to retain reduced dimensions
}
```

**NOT_EQUAL**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: a tensor of type bool, true whenever an element of the first tensor is not
  equal to the corresponding element of the second tensor.
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

**RSQRT**

```
Inputs {
  0: a tensor
}
Outputs {
  0: result of computing element-wise reciprocal square root of the input tensor
}
```

**SHAPE**

```
Inputs {
  0: a tensor
}
Outputs {
  0: a 1D tensor representing the shape of the input tensor
}
Options {
  out_type: the output type of the op (int32 or int64). Defaults to int32.
}
```

**SLICE**

```
Inputs {
  0: tensor
  1: 1D tensor
  2: 1D tensor
}
Outputs {
  0: slice of the input tensor of the given size from the given begin index.
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

**SPACE_TO_BATCH_ND**

```
Inputs {
  0: 4D tensor
  1: 1D tensor
  2: 2D tensor
}
Outputs {
  0: a tensor rearranged using block_shape. See tf.space_to_batch_nd for
     details.
}
```

**SPARSE_TO_DENSE**

```
Inputs {
  0: 0D or 1D or 2D tensor
  1: 1D tensor
  2: 0D or 1D tensor
  3: 0D tensor
  4: a boolean value
}
Outputs {
  0: Dense Tensor of shape output_shape. Has the same type as sparse_values.
}
```

**SPLIT**

```
Inputs {
  0: 0D tensor (axis)
  1: tensor (input)
}
Outputs {
  0-N: subtensors built from the input tensors
}
Options {
  num_splits: Specifies number of outputs
}
```

**SQRT**

```
Inputs {
  0: a tensor
}
Outputs {
  0: result of computing element-wise square root of the input tensor
}
```

**SQUEEZE**

```
Inputs {
  0: tensor
}
Outputs {
  0: tensor without any dimensions of size 1
}
Options {
  squeeze_dims
}
```

**STRIDED_SLICE**

```
Inputs {
  0: tensor
  1: 1D tensor
  2: 1D tensor
  3: 1D tensor
}
Outputs {
  0: slice of the input tensor of the given size
}
Options {
  begin_mask: mask for begin indices
  end_mask: mask for end indices
  shrink_axis_mask: mask that indicates which dimensions to remove
}
```

**TOP_K**

```
Inputs {
  0: tensor
  1: OD tensor
}
Outputs {
  0: k largest element along each last dimensional slice
  1: indices of values within the last dimension of the input ensor
}
```

**TRANSPOSE**

```
Inputs {
  0: tensor
  1: tensor
}
Outputs {
  0: tensor permuted according to perm
}
```

**SELECT**

```
Inputs {
  0: tensor
  1: tensor
  2: tensor
}
Outputs {
  0: tensor that contains the elementwise values of 'tensor 1' if the
  corresponding value of 'tensor 0' is true or the value of 'tensor 2' if false.
}
```

**POW**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: elementwise pow of the input tensors
}
```

**ARG_MAX**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: A tensor of indices of maximum values.
}
```

**ARG_MIN**

```
Inputs {
  0: a tensor
  1: a tensor
}
Outputs {
  0: A tensor of indices of minium values.
}
```

**PACK**

```
Inputs {
  0: a list of tensors.
  1: an integer.
}
Outputs {
  0: A tensor of stacked tensors.
}
```

**LOGICAL_OR**

```
Inputs {
  0: a list of tensors.
  1: a list of tensors.
}
Outputs {
  0: A tensor of logical_or output tensors.
}
```

**UNPACK**

```
Inputs {
  0: a tensor.
  1: an integer.
  2: an integer.
}
Outputs {
  0-N: tensors of unpacked tensor.
}
```

**FLOOR_DIV**

```
Inputs {
  0: a list of tensors.
  1: a list of tensors.
}
Outputs {
  0: A tensor of floor_div output tensors.
}
```

**ZEROS_LIKE**

```
Inputs {
  0: a tensor
}
Outputs {
  0: A tensor of the same shape and type as x but filled with zeros
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
