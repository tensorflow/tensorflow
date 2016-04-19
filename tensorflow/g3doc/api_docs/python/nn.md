<!-- This file is machine generated: DO NOT EDIT! -->

# Neural Network

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Activation Functions

The activation ops provide different types of nonlinearities for use in neural
networks.  These include smooth nonlinearities (`sigmoid`, `tanh`, `elu`,
`softplus`, and `softsign`), continuous but not everywhere differentiable
functions (`relu`, `relu6`, and `relu_x`), and random regularization
(`dropout`).

All activation ops apply componentwise, and produce a tensor of the same
shape as the input tensor.

- - -

### `tf.nn.relu(features, name=None)` {#relu}

Computes rectified linear: `max(features, 0)`.

##### Args:


*  <b>`features`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `features`.


- - -

### `tf.nn.relu6(features, name=None)` {#relu6}

Computes Rectified Linear 6: `min(max(features, 0), 6)`.

##### Args:


*  <b>`features`</b>: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
    `int16`, or `int8`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with the same type as `features`.


- - -

### `tf.nn.elu(features, name=None)` {#elu}

Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)

##### Args:


*  <b>`features`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `features`.


- - -

### `tf.nn.softplus(features, name=None)` {#softplus}

Computes softplus: `log(exp(features) + 1)`.

##### Args:


*  <b>`features`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `features`.


- - -

### `tf.nn.softsign(features, name=None)` {#softsign}

Computes softsign: `features / (abs(features) + 1)`.

##### Args:


*  <b>`features`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `features`.


- - -

### `tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)` {#dropout}

Computes dropout.

With probability `keep_prob`, outputs the input element scaled up by
`1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
sum is unchanged.

By default, each element is kept or dropped independently.  If `noise_shape`
is specified, it must be
[broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
kept independently and each row and column will be kept or not kept together.

##### Args:


*  <b>`x`</b>: A tensor.
*  <b>`keep_prob`</b>: A scalar `Tensor` with the same type as x. The probability
    that each element is kept.
*  <b>`noise_shape`</b>: A 1-D `Tensor` of type `int32`, representing the
    shape for randomly generated keep/drop flags.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A Tensor of the same shape of `x`.

##### Raises:


*  <b>`ValueError`</b>: If `keep_prob` is not in `(0, 1]`.


- - -

### `tf.nn.bias_add(value, bias, data_format=None, name=None)` {#bias_add}

Adds `bias` to `value`.

This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.
Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
case where both types are quantized.

##### Args:


*  <b>`value`</b>: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
    `int16`, `int8`, or `complex64`.
*  <b>`bias`</b>: A 1-D `Tensor` with size matching the last dimension of `value`.
    Must be the same type as `value` unless `value` is a quantized type,
    in which case a different quantized type may be used.
*  <b>`data_format`</b>: A string. 'NHWC' and 'NCHW' are supported.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with the same type as `value`.


- - -

### `tf.sigmoid(x, name=None)` {#sigmoid}

Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.

##### Args:


*  <b>`x`</b>: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
    or `qint32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A Tensor with the same type as `x` if `x.dtype != qint32`
    otherwise the return type is `quint8`.


- - -

### `tf.tanh(x, name=None)` {#tanh}

Computes hyperbolic tangent of `x` element-wise.

##### Args:


*  <b>`x`</b>: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
    or `qint32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
    the return type is `quint8`.



## Convolution

The convolution ops sweep a 2-D filter over a batch of images, applying the
filter to each window of each image of the appropriate size.  The different
ops trade off between generic vs. specific filters:

* `conv2d`: Arbitrary filters that can mix channels together.
* `depthwise_conv2d`: Filters that operate on each channel independently.
* `separable_conv2d`: A depthwise spatial filter followed by a pointwise filter.

Note that although these ops are called "convolution", they are strictly
speaking "cross-correlation" since the filter is combined with an input window
without reversing the filter.  For details, see [the properties of
cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation#Properties).

The filter is applied to image patches of the same size as the filter and
strided according to the `strides` argument.  `strides = [1, 1, 1, 1]` applies
the filter to a patch at every offset, `strides = [1, 2, 2, 1]` applies the
filter to every other image patch in each dimension, etc.

Ignoring channels for the moment, and assume that the 4-D `input` has shape
`[batch, in_height, in_width, ...]` and the 4-D `filter` has shape
`[filter_height, filter_width, ...]`, then the spatial semantics of the
convolution ops are as follows: first, according to the padding scheme chosen
as `'SAME'` or `'VALID'`, the output size and the padding pixels are computed.
For the `'SAME'` padding, the output height and width are computed as:

    out_height = ceil(float(in_height) / float(strides[1]))
    out_width  = ceil(float(in_width) / float(strides[2]))

and the padding on the top and left are computed as:

    pad_along_height = ((out_height - 1) * strides[1] +
                        filter_height - in_height)
    pad_along_width = ((out_width - 1) * strides[2] +
                       filter_width - in_width)
    pad_top = pad_along_height / 2
    pad_left = pad_along_width / 2

Note that the division by 2 means that there might be cases when the padding on
both sides (top vs bottom, right vs left) are off by one. In this case, the
bottom and right sides always get the one additional padded pixel. For example,
when `pad_along_height` is 5, we pad 2 pixels at the top and 3 pixels at the
bottom. Note that this is different from existing libraries such as cuDNN and
Caffe, which explicitly specify the number of padded pixels and always pad the
same number of pixels on both sides.

For the `'VALID`' padding, the output height and width are computed as:

    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

and the padding values are always zero. The output is then computed as

    output[b, i, j, :] =
        sum_{di, dj} input[b, strides[1] * i + di - pad_top,
                           strides[2] * j + dj - pad_left, ...] *
                     filter[di, dj, ...]

where any value outside the original input image region are considered zero (
i.e. we pad zero values around the border of the image).

Since `input` is 4-D, each `input[b, i, j, :]` is a vector.  For `conv2d`, these
vectors are multiplied by the `filter[di, dj, :, :]` matrices to produce new
vectors.  For `depthwise_conv_2d`, each scalar component `input[b, i, j, k]`
is multiplied by a vector `filter[di, dj, k]`, and all the vectors are
concatenated.

- - -

### `tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)` {#conv2d}

Computes a 2-D convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:

1. Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
   vector.

In detail, with the default NHWC format,

    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`filter`</b>: A `Tensor`. Must have the same type as `input`.
*  <b>`strides`</b>: A list of `ints`.
    1-D of length 4.  The stride of the sliding window for each dimension
    of `input`. Must be in the same order as the dimension specified with format.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`use_cudnn_on_gpu`</b>: An optional `bool`. Defaults to `True`.
*  <b>`data_format`</b>: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
    Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)` {#depthwise_conv2d}

Depthwise 2-D convolution.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`
containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
applies a different filter to each input channel (expanding from 1 channel
to `channel_multiplier` channels for each), then concatenates the results
together.  The output has `in_channels * channel_multiplier` channels.

In detail,

    output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                     filter[di, dj, k, q]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the
same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

##### Args:


*  <b>`input`</b>: 4-D with shape `[batch, in_height, in_width, in_channels]`.
*  <b>`filter`</b>: 4-D with shape
    `[filter_height, filter_width, in_channels, channel_multiplier]`.
*  <b>`strides`</b>: 1-D of size 4.  The stride of the sliding window for each
    dimension of `input`.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A 4-D `Tensor` of shape
  `[batch, out_height, out_width, in_channels * channel_multiplier].`


- - -

### `tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None)` {#separable_conv2d}

2-D convolution with separable filters.

Performs a depthwise convolution that acts separately on channels followed by
a pointwise convolution that mixes channels.  Note that this is separability
between dimensions `[1, 2]` and `3`, not spatial separability between
dimensions `1` and `2`.

In detail,

    output[b, i, j, k] = sum_{di, dj, q, r]
        input[b, strides[1] * i + di, strides[2] * j + dj, q] *
        depthwise_filter[di, dj, q, r] *
        pointwise_filter[0, 0, q * channel_multiplier + r, k]

`strides` controls the strides for the depthwise convolution only, since
the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
`strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

##### Args:


*  <b>`input`</b>: 4-D `Tensor` with shape `[batch, in_height, in_width, in_channels]`.
*  <b>`depthwise_filter`</b>: 4-D `Tensor` with shape
    `[filter_height, filter_width, in_channels, channel_multiplier]`.
    Contains `in_channels` convolutional filters of depth 1.
*  <b>`pointwise_filter`</b>: 4-D `Tensor` with shape
    `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
    filter to mix channels after `depthwise_filter` has convolved spatially.
*  <b>`strides`</b>: 1-D of size 4.  The strides for the depthwise convolution for
    each dimension of `input`.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A 4-D `Tensor` of shape `[batch, out_height, out_width, out_channels]`.


- - -

### `tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)` {#conv2d_transpose}

The transpose of `conv2d`.

This operation is sometimes called "deconvolution" after (Deconvolutional
Networks)[http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf], but is
actually the transpose (gradient) of `conv2d` rather than an actual
deconvolution.

##### Args:


*  <b>`value`</b>: A 4-D `Tensor` of type `float` and shape
    `[batch, height, width, in_channels]`.
*  <b>`filter`</b>: A 4-D `Tensor` with the same type as `value` and shape
    `[height, width, output_channels, in_channels]`.  `filter`'s
    `in_channels` dimension must match that of `value`.
*  <b>`output_shape`</b>: A 1-D `Tensor` representing the output shape of the
    deconvolution op.
*  <b>`strides`</b>: A list of ints. The stride of the sliding window for each
    dimension of the input tensor.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
*  <b>`name`</b>: Optional name for the returned tensor.

##### Returns:

  A `Tensor` with the same type as `value`.

##### Raises:


*  <b>`ValueError`</b>: If input/output depth does not match `filter`'s shape, or if
    padding is other than `'VALID'` or `'SAME'`.



## Pooling

The pooling ops sweep a rectangular window over the input tensor, computing a
reduction operation for each window (average, max, or max with argmax).  Each
pooling op uses rectangular windows of size `ksize` separated by offset
`strides`.  For example, if `strides` is all ones every window is used, if
`strides` is all twos every other window is used in each dimension, etc.

In detail, the output is

    output[i] = reduce(value[strides * i:strides * i + ksize])

where the indices also take into consideration the padding values. Please refer
to the `Convolution` section for details about the padding calculation.

- - -

### `tf.nn.avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None)` {#avg_pool}

Performs the average pooling on the input.

Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.

##### Args:


*  <b>`value`</b>: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
    `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
*  <b>`ksize`</b>: A list of ints that has length >= 4.
    The size of the window for each dimension of the input tensor.
*  <b>`strides`</b>: A list of ints that has length >= 4.
    The stride of the sliding window for each dimension of the
    input tensor.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
*  <b>`data_format`</b>: A string. 'NHWC' and 'NCHW' are supported.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  A `Tensor` with the same type as `value`.  The average pooled output tensor.


- - -

### `tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)` {#max_pool}

Performs the max pooling on the input.

##### Args:


*  <b>`value`</b>: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
    type `tf.float32`.
*  <b>`ksize`</b>: A list of ints that has length >= 4.  The size of the window for
    each dimension of the input tensor.
*  <b>`strides`</b>: A list of ints that has length >= 4.  The stride of the sliding
    window for each dimension of the input tensor.
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
*  <b>`data_format`</b>: A string. 'NHWC' and 'NCHW' are supported.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  A `Tensor` with type `tf.float32`.  The max pooled output tensor.


- - -

### `tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)` {#max_pool_with_argmax}

Performs max pooling on the input and outputs both max values and indices.

The indices in `argmax` are flattened, so that a maximum value at position
`[b, y, x, c]` becomes flattened index
`((b * height + y) * width + x) * channels + c`.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `float32`.
    4-D with shape `[batch, height, width, channels]`.  Input to pool over.
*  <b>`ksize`</b>: A list of `ints` that has length `>= 4`.
    The size of the window for each dimension of the input tensor.
*  <b>`strides`</b>: A list of `ints` that has length `>= 4`.
    The stride of the sliding window for each dimension of the
    input tensor.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`Targmax`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (output, argmax).

*  <b>`output`</b>: A `Tensor` of type `float32`. The max pooled output tensor.
*  <b>`argmax`</b>: A `Tensor` of type `Targmax`. 4-D.  The flattened indices of the max values chosen for each output.



## Normalization

Normalization is useful to prevent neurons from saturating when inputs may
have varying scale, and to aid generalization.

- - -

### `tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)` {#l2_normalize}

Normalizes along dimension `dim` using an L2 norm.

For a 1-D tensor with `dim = 0`, computes

    output = x / sqrt(max(sum(x**2), epsilon))

For `x` with more dimensions, independently normalizes each 1-D slice along
dimension `dim`.

##### Args:


*  <b>`x`</b>: A `Tensor`.
*  <b>`dim`</b>: Dimension along which to normalize.
*  <b>`epsilon`</b>: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
    divisor if `norm < sqrt(epsilon)`.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A `Tensor` with the same shape as `x`.


- - -

### `tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)` {#local_response_normalization}

Local Response Normalization.

The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
dimension), and each vector is normalized independently.  Within a given vector,
each component is divided by the weighted, squared sum of inputs within
`depth_radius`.  In detail,

    sqr_sum[a, b, c, d] =
        sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    output = input / (bias + alpha * sqr_sum) ** beta

For details, see [Krizhevsky et al., ImageNet classification with deep
convolutional neural networks (NIPS 2012)]
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

##### Args:


*  <b>`input`</b>: A `Tensor` of type `float32`. 4-D.
*  <b>`depth_radius`</b>: An optional `int`. Defaults to `5`.
    0-D.  Half-width of the 1-D normalization window.
*  <b>`bias`</b>: An optional `float`. Defaults to `1`.
    An offset (usually positive to avoid dividing by 0).
*  <b>`alpha`</b>: An optional `float`. Defaults to `1`.
    A scale factor, usually positive.
*  <b>`beta`</b>: An optional `float`. Defaults to `0.5`. An exponent.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.


- - -

### `tf.nn.sufficient_statistics(x, axes, shift=True, keep_dims=False, name=None)` {#sufficient_statistics}

Calculate the sufficient statistics for the mean and variance of `x`.

These sufficient statistics are computed using the one pass algorithm on
an input that's optionally shifted using the value of the 1st element in `x`.
See:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

##### Args:


*  <b>`x`</b>: A `Tensor`.
*  <b>`axes`</b>: Array of ints. Axes along which to compute mean and variance.
*  <b>`shift`</b>: If true, shift the data to provide more numerically stable results.
*  <b>`keep_dims`</b>: produce statistics with the same dimensionality as the input.
*  <b>`name`</b>: Name used to scope the operations that compute the sufficient stats.

##### Returns:

  Four `Tensor` objects of the same type as `x`:
  * the count (number of elements to average over).
  * the (possibly shifted) sum of the elements in the array.
  * the (possibly shifted) sum of squares of the elements in the array.
  * the shift by which the mean must be corrected or None if `shift` is False.


- - -

### `tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift, name=None)` {#normalize_moments}

Calculate the mean and variance of based on the sufficient statistics.

##### Args:


*  <b>`counts`</b>: A `Tensor` containing a the total count of the data (one value).
*  <b>`mean_ss`</b>: A `Tensor` containing the mean sufficient statistics: the (possibly
    shifted) sum of the elements to average over.
*  <b>`variance_ss`</b>: A `Tensor` containing the variance sufficient statistics: the
    (possibly shifted) squared sum of the data to compute the variance over.
*  <b>`shift`</b>: A `Tensor` containing the value by which the data is shifted for
    numerical stability, or `None` if no shift was performed.
*  <b>`name`</b>: Name used to scope the operations that compute the moments.

##### Returns:

  Two `Tensor` objects: `mean` and `variance`.


- - -

### `tf.nn.moments(x, axes, name=None, keep_dims=False)` {#moments}

Calculate the mean and variance of `x`.

The mean and variance are calculated by aggregating the contents of `x`
across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
and variance of a vector.

When using these moments for batch normalization (see
`tf.nn.batch_normalization`):
  * for so-called "global normalization", used with convolutional filters with
    shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
  * for simple batch normalization pass `axes=[0]` (batch only).

##### Args:


*  <b>`x`</b>: A `Tensor`.
*  <b>`axes`</b>: array of ints.  Axes along which to compute mean and
    variance.
*  <b>`keep_dims`</b>: produce moments with the same dimensionality as the input.
*  <b>`name`</b>: Name used to scope the operations that compute the moments.

##### Returns:

  Two `Tensor` objects: `mean` and `variance`.



## Losses

The loss ops measure error between two tensors, or between a tensor and zero.
These can be used for measuring accuracy of a network in a regression task
or for regularization purposes (weight decay).

- - -

### `tf.nn.l2_loss(t, name=None)` {#l2_loss}

L2 Loss.

Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2

##### Args:


*  <b>`t`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    Typically 2-D, but may have any dimensions.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `t`. 0-D.



## Classification

TensorFlow provides several operations that help you perform classification.

- - -

### `tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)` {#sigmoid_cross_entropy_with_logits}

Computes sigmoid cross entropy given `logits`.

Measures the probability error in discrete classification tasks in which each
class is independent and not mutually exclusive.  For instance, one could
perform multilabel classification where a picture can contain both an elephant
and a dog at the same time.

For brevity, let `x = logits`, `z = targets`.  The logistic loss is

      z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
    = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
    = (1 - z) * x + log(1 + exp(-x))
    = x - x * z + log(1 + exp(-x))

To ensure stability and avoid overflow, the implementation uses

    max(x, 0) - x * z + log(1 + exp(-abs(x)))

`logits` and `targets` must have the same type and shape.

##### Args:


*  <b>`logits`</b>: A `Tensor` of type `float32` or `float64`.
*  <b>`targets`</b>: A `Tensor` of the same type and shape as `logits`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of the same shape as `logits` with the componentwise
  logistic losses.

##### Raises:


*  <b>`ValueError`</b>: If `logits` and `targets` do not have the same shape.


- - -

### `tf.nn.softmax(logits, name=None)` {#softmax}

Computes softmax activations.

For each batch `i` and class `j` we have

    softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

##### Args:


*  <b>`logits`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    2-D with shape `[batch_size, num_classes]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `logits`. Same shape as `logits`.


- - -

### `tf.nn.log_softmax(logits, name=None)` {#log_softmax}

Computes log softmax activations.

For each batch `i` and class `j` we have

    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

##### Args:


*  <b>`logits`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    2-D with shape `[batch_size, num_classes]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `logits`. Same shape as `logits`.


- - -

### `tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)` {#softmax_cross_entropy_with_logits}

Computes softmax cross entropy between `logits` and `labels`.

Measures the probability error in discrete classification tasks in which the
classes are mutually exclusive (each entry is in exactly one class).  For
example, each CIFAR-10 image is labeled with one and only one label: an image
can be a dog or a truck, but not both.

**NOTE:**  While the classes are mutually exclusive, their probabilities
need not be. If using exclusive `labels` (wherein one and only one class is
true at a time), see `sparse_softmax_cross_entropy_with_logits`.

**WARNING:** This op expects unscaled logits, since it performs a `softmax`
on `logits` internally for efficiency.  Do not call this op with the
output of `softmax`, as it will produce incorrect results.

`logits` and `labels` must have the same shape `[batch_size, num_classes]`
and the same dtype (either `float32` or `float64`).

##### Args:


*  <b>`logits`</b>: Unscaled log probabilities.
*  <b>`labels`</b>: Each row `labels[i]` must be a valid probability distribution or
      all zeros. If all zeros, the corresponding loss will be `0`, regardless
      of the contents of `logits[i]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
  softmax cross entropy loss.


- - -

### `tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)` {#sparse_softmax_cross_entropy_with_logits}

Computes sparse softmax cross entropy between `logits` and `labels`.

Measures the probability error in discrete classification tasks in which the
classes are mutually exclusive (each entry is in exactly one class).  For
example, each CIFAR-10 image is labeled with one and only one label: an image
can be a dog or a truck, but not both.

**NOTE:**  For this operation, the probability of a given label is considered
exclusive.  That is, soft classes are not allowed, and the `labels` vector
must provide a single specific index for the true class for each row of
`logits` (each minibatch entry).  For soft softmax classification with
a probability distribution for each entry, see
`softmax_cross_entropy_with_logits`.

**WARNING:** This op expects unscaled logits, since it performs a softmax
on `logits` internally for efficiency.  Do not call this op with the
output of `softmax`, as it will produce incorrect results.

`logits` must have the shape `[batch_size, num_classes]`
and dtype `float32` or `float64`.

`labels` must have the shape `[batch_size]` and dtype `int32` or `int64`.

##### Args:


*  <b>`logits`</b>: Unscaled log probabilities.
*  <b>`labels`</b>: Each entry `labels[i]` must be an index in `[0, num_classes)` or
      `-1`. If `-1`, the corresponding loss will be `0`, regardless
      of the contents of `logits[i]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
  softmax cross entropy loss.


- - -

### `tf.nn.weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None)` {#weighted_cross_entropy_with_logits}

Computes a weighted cross entropy.

This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
allows one to trade off recall and precision by up- or down-weighting the
cost of a positive error relative to a negative error.

The usual cross-entropy cost is defined as:

  targets * -log(sigmoid(logits)) + (1 - targets) * -log(1 - sigmoid(logits))

The argument `pos_weight` is used as a multiplier for the positive targets:

  targets * -log(sigmoid(logits)) * pos_weight +
      (1 - targets) * -log(1 - sigmoid(logits))

For brevity, let `x = logits`, `z = targets`, `q = pos_weight`.
The loss is:

      qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
    = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
    = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
    = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
the implementation uses

    (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

`logits` and `targets` must have the same type and shape.

##### Args:


*  <b>`logits`</b>: A `Tensor` of type `float32` or `float64`.
*  <b>`targets`</b>: A `Tensor` of the same type and shape as `logits`.
*  <b>`pos_weight`</b>: A coefficient to use on the positive examples.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of the same shape as `logits` with the componentwise
  weightedlogistic losses.

##### Raises:


*  <b>`ValueError`</b>: If `logits` and `targets` do not have the same shape.



## Embeddings

TensorFlow provides library support for looking up values in embedding
tensors.

- - -

### `tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True)` {#embedding_lookup}

Looks up `ids` in a list of embedding tensors.

This function is used to perform parallel lookups on the list of
tensors in `params`.  It is a generalization of
[`tf.gather()`](../../api_docs/python/array_ops.md#gather), where `params` is
interpreted as a partition of a larger embedding tensor.

If `len(params) > 1`, each element `id` of `ids` is partitioned between
the elements of `params` according to the `partition_strategy`.
In all strategies, if the id space does not evenly divide the number of
partitions, each of the first `(max_id + 1) % len(params)` partitions will
be assigned one more id.

If `partition_strategy` is `"mod"`, we assign each id to partition
`p = id % len(params)`. For instance,
13 ids are split across 5 partitions as:
`[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`

If `partition_strategy` is `"div"`, we assign ids to partitions in a
contiguous manner. In this case, 13 ids are split across 5 partitions as:
`[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

The results of the lookup are concatenated into a dense
tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

##### Args:


*  <b>`params`</b>: A list of tensors with the same type and which can be concatenated
    along dimension 0. Each `Tensor` must be appropriately sized for the given
    `partition_strategy`.
*  <b>`ids`</b>: A `Tensor` with type `int32` or `int64` containing the ids to be looked
    up in `params`.
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy, relevant
    if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
    is `"mod"`.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`validate_indices`</b>: Whether or not to validate gather indices.

##### Returns:

  A `Tensor` with the same type as the tensors in `params`.

##### Raises:


*  <b>`ValueError`</b>: If `params` is empty.


- - -

### `tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy='mod', name=None, combiner='mean')` {#embedding_lookup_sparse}

Computes embeddings for the given ids and weights.

This op assumes that there is at least one id for each row in the dense tensor
represented by sp_ids (i.e. there are no rows with empty features), and that
all the indices of sp_ids are in canonical row-major order.

It also assumes that all id values lie in the range [0, p0), where p0
is the sum of the size of params along dimension 0.

##### Args:


*  <b>`params`</b>: A single tensor representing the complete embedding tensor,
    or a list of P tensors all of same shape except for the first dimension,
    representing sharded embedding tensors.
*  <b>`sp_ids`</b>: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
    where N is typically batch size and M is arbitrary.
*  <b>`sp_weights`</b>: either a SparseTensor of float / double weights, or None to
    indicate all weights should be taken to be 1. If specified, sp_weights
    must have exactly the same shape and indices as sp_ids.
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy, relevant
    if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
    is `"mod"`. See `tf.nn.embedding_lookup` for more details.
*  <b>`name`</b>: Optional name for the op.
*  <b>`combiner`</b>: A string specifying the reduction op. Currently "mean", "sqrtn"
    and "sum" are supported.
    "sum" computes the weighted sum of the embedding results for each row.
    "mean" is the weighted sum divided by the total weight.
    "sqrtn" is the weighted sum divided by the square root of the sum of the
    squares of the weights.

##### Returns:

  A dense tensor representing the combined embeddings for the
  sparse ids. For each row in the dense tensor represented by sp_ids, the op
  looks up the embeddings for all ids in that row, multiplies them by the
  corresponding weight, and combines these embeddings as specified.

  In other words, if
    shape(combined params) = [p0, p1, ..., pm]
  and
    shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]
  then
    shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].

  For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

    [0, 0]: id 1, weight 2.0
    [0, 1]: id 3, weight 0.5
    [1, 0]: id 0, weight 1.0
    [2, 3]: id 1, weight 3.0

  with combiner="mean", then the output will be a 3x20 matrix where
    output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
    output[1, :] = params[0, :] * 1.0
    output[2, :] = params[1, :] * 3.0

##### Raises:


*  <b>`TypeError`</b>: If sp_ids is not a SparseTensor, or if sp_weights is neither
    None nor SparseTensor.
*  <b>`ValueError`</b>: If combiner is not one of {"mean", "sqrtn", "sum"}.



## Evaluation

The evaluation ops are useful for measuring the performance of a network.
Since they are nondifferentiable, they are typically used at evaluation time.

- - -

### `tf.nn.top_k(input, k=1, sorted=True, name=None)` {#top_k}

Finds values and indices of the `k` largest entries for the last dimension.

If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

    values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.

##### Args:


*  <b>`input`</b>: 1-D or higher `Tensor` with last dimension at least `k`.
*  <b>`k`</b>: 0-D `int32` `Tensor`.  Number of top elements to look for along the last
    dimension (along each row for matrices).
*  <b>`sorted`</b>: If true the resulting `k` elements will be sorted by the values in
    descending order.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:


*  <b>`values`</b>: The `k` largest elements along each last dimensional slice.
*  <b>`indices`</b>: The indices of `values` within the last dimension of `input`.


- - -

### `tf.nn.in_top_k(predictions, targets, k, name=None)` {#in_top_k}

Says whether the targets are in the top `K` predictions.

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is among the top `k` predictions among
all predictions for example `i`. Note that the behavior of `InTopK` differs
from the `TopK` op in its handling of ties; if multiple classes have the
same prediction value and straddle the top-`k` boundary, all of those
classes are considered to be in the top `k`.

More formally, let

  \\(predictions_i\\) be the predictions for all classes for example `i`,
  \\(targets_i\\) be the target class for example `i`,
  \\(out_i\\) be the output for example `i`,

$$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

##### Args:


*  <b>`predictions`</b>: A `Tensor` of type `float32`.
    A `batch_size` x `classes` tensor.
*  <b>`targets`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A `batch_size` vector of class ids.
*  <b>`k`</b>: An `int`. Number of top elements to look at for computing precision.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `bool`. Computed Precision at `k` as a `bool Tensor`.



## Candidate Sampling

Do you want to train a multiclass or multilabel model with thousands
or millions of output classes (for example, a language model with a
large vocabulary)?  Training with a full Softmax is slow in this case,
since all of the classes are evaluated for every training example.
Candidate Sampling training algorithms can speed up your step times by
only considering a small randomly-chosen subset of contrastive classes
(called candidates) for each batch of training examples.

See our [Candidate Sampling Algorithms Reference]
(../../extras/candidate_sampling.pdf)

### Sampled Loss Functions

TensorFlow provides the following sampled loss functions for faster training.

- - -

### `tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, partition_strategy='mod', name='nce_loss')` {#nce_loss}

Computes and returns the noise-contrastive estimation training loss.

See [Noise-contrastive estimation: A new estimation principle for
unnormalized statistical models]
(http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
Also see our [Candidate Sampling Algorithms Reference]
(../../extras/candidate_sampling.pdf)

Note: In the case where `num_true` > 1, we assign to each target class
the target probability 1 / `num_true` so that the target probabilities
sum to 1 per-example.

Note: It would be useful to allow a variable number of target classes per
example.  We hope to provide this functionality in a future release.
For now, if you have a variable number of target classes, you can pad them
out to a constant number by either repeating them or by padding
with an otherwise unused class.

##### Args:


*  <b>`weights`</b>: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
      objects whose concatenation along dimension 0 has shape
      [num_classes, dim].  The (possibly-partitioned) class embeddings.
*  <b>`biases`</b>: A `Tensor` of shape `[num_classes]`.  The class biases.
*  <b>`inputs`</b>: A `Tensor` of shape `[batch_size, dim]`.  The forward
      activations of the input network.
*  <b>`labels`</b>: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
*  <b>`num_sampled`</b>: An `int`.  The number of classes to randomly sample per batch.
*  <b>`num_classes`</b>: An `int`. The number of possible classes.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`sampled_values`</b>: a tuple of (`sampled_candidates`, `true_expected_count`,
      `sampled_expected_count`) returned by a `*_candidate_sampler` function.
      (if None, we default to `log_uniform_candidate_sampler`)
*  <b>`remove_accidental_hits`</b>: A `bool`.  Whether to remove "accidental hits"
      where a sampled class equals one of the target classes.  If set to
      `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
      learning to generate log-odds instead of log probabilities.  See
      our [Candidate Sampling Algorithms Reference]
      (../../extras/candidate_sampling.pdf).
      Default is False.
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy, relevant
      if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
      Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `batch_size` 1-D tensor of per-example NCE losses.


- - -

### `tf.nn.sampled_softmax_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, partition_strategy='mod', name='sampled_softmax_loss')` {#sampled_softmax_loss}

Computes and returns the sampled softmax training loss.

This is a faster way to train a softmax classifier over a huge number of
classes.

This operation is for training only.  It is generally an underestimate of
the full softmax loss.

At inference time, you can compute full softmax probabilities with the
expression `tf.nn.softmax(tf.matmul(inputs, weights) + biases)`.

See our [Candidate Sampling Algorithms Reference]
(../../extras/candidate_sampling.pdf)

Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

##### Args:


*  <b>`weights`</b>: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
      objects whose concatenation along dimension 0 has shape
      [num_classes, dim].  The (possibly-sharded) class embeddings.
*  <b>`biases`</b>: A `Tensor` of shape `[num_classes]`.  The class biases.
*  <b>`inputs`</b>: A `Tensor` of shape `[batch_size, dim]`.  The forward
      activations of the input network.
*  <b>`labels`</b>: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.  Note that this format differs from
      the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
*  <b>`num_sampled`</b>: An `int`.  The number of classes to randomly sample per batch.
*  <b>`num_classes`</b>: An `int`. The number of possible classes.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`sampled_values`</b>: a tuple of (`sampled_candidates`, `true_expected_count`,
      `sampled_expected_count`) returned by a `*_candidate_sampler` function.
      (if None, we default to `log_uniform_candidate_sampler`)
*  <b>`remove_accidental_hits`</b>: A `bool`.  whether to remove "accidental hits"
      where a sampled class equals one of the target classes.  Default is
      True.
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy, relevant
      if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
      Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `batch_size` 1-D tensor of per-example sampled softmax losses.



### Candidate Samplers

TensorFlow provides the following samplers for randomly sampling candidate
classes when using one of the sampled loss functions above.

- - -

### `tf.nn.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)` {#uniform_candidate_sampler}

Samples a set of classes using a uniform base distribution.

This operation randomly samples a tensor of sampled classes
(`sampled_candidates`) from the range of integers `[0, range_max]`.

The elements of `sampled_candidates` are drawn without replacement
(if `unique=True`) or with replacement (if `unique=False`) from
the base distribution.

The base distribution for this operation is the uniform distribution
over the range of integers `[0, range_max]`.

In addition, this operation returns tensors `true_expected_count`
and `sampled_expected_count` representing the number of times each
of the target classes (`true_classes`) and the sampled
classes (`sampled_candidates`) is expected to occur in an average
tensor of sampled classes.  These values correspond to `Q(y|x)`
defined in [this
document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
If `unique=True`, then these are post-rejection probabilities and we
compute them approximately.

##### Args:


*  <b>`true_classes`</b>: A `Tensor` of type `int64` and shape `[batch_size,
    num_true]`. The target classes.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`num_sampled`</b>: An `int`.  The number of classes to randomly sample per batch.
*  <b>`unique`</b>: A `bool`. Determines whether all sampled classes in a batch are
    unique.
*  <b>`range_max`</b>: An `int`. The number of possible classes.
*  <b>`seed`</b>: An `int`. An operation-specific seed. Default is 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`sampled_candidates`</b>: A tensor of type `int64` and shape `[num_sampled]`.
    The sampled classes.
*  <b>`true_expected_count`</b>: A tensor of type `float`.  Same shape as
    `true_classes`. The expected counts under the sampling distribution
    of each of `true_classes`.
*  <b>`sampled_expected_count`</b>: A tensor of type `float`. Same shape as
    `sampled_candidates`. The expected counts under the sampling distribution
    of each of `sampled_candidates`.


- - -

### `tf.nn.log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)` {#log_uniform_candidate_sampler}

Samples a set of classes using a log-uniform (Zipfian) base distribution.

This operation randomly samples a tensor of sampled classes
(`sampled_candidates`) from the range of integers `[0, range_max]`.

The elements of `sampled_candidates` are drawn without replacement
(if `unique=True`) or with replacement (if `unique=False`) from
the base distribution.

The base distribution for this operation is an approximately log-uniform
or Zipfian distribution:

`P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

This sampler is useful when the target classes approximately follow such
a distribution - for example, if the classes represent words in a lexicon
sorted in decreasing order of frequency. If your classes are not ordered by
decreasing frequency, do not use this op.

In addition, this operation returns tensors `true_expected_count`
and `sampled_expected_count` representing the number of times each
of the target classes (`true_classes`) and the sampled
classes (`sampled_candidates`) is expected to occur in an average
tensor of sampled classes.  These values correspond to `Q(y|x)`
defined in [this
document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
If `unique=True`, then these are post-rejection probabilities and we
compute them approximately.

##### Args:


*  <b>`true_classes`</b>: A `Tensor` of type `int64` and shape `[batch_size,
    num_true]`. The target classes.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`num_sampled`</b>: An `int`.  The number of classes to randomly sample per batch.
*  <b>`unique`</b>: A `bool`. Determines whether all sampled classes in a batch are
    unique.
*  <b>`range_max`</b>: An `int`. The number of possible classes.
*  <b>`seed`</b>: An `int`. An operation-specific seed. Default is 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`sampled_candidates`</b>: A tensor of type `int64` and shape `[num_sampled]`.
    The sampled classes.
*  <b>`true_expected_count`</b>: A tensor of type `float`.  Same shape as
    `true_classes`. The expected counts under the sampling distribution
    of each of `true_classes`.
*  <b>`sampled_expected_count`</b>: A tensor of type `float`. Same shape as
    `sampled_candidates`. The expected counts under the sampling distribution
    of each of `sampled_candidates`.


- - -

### `tf.nn.learned_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)` {#learned_unigram_candidate_sampler}

Samples a set of classes from a distribution learned during training.

This operation randomly samples a tensor of sampled classes
(`sampled_candidates`) from the range of integers `[0, range_max]`.

The elements of `sampled_candidates` are drawn without replacement
(if `unique=True`) or with replacement (if `unique=False`) from
the base distribution.

The base distribution for this operation is constructed on the fly
during training.  It is a unigram distribution over the target
classes seen so far during training.  Every integer in `[0, range_max]`
begins with a weight of 1, and is incremented by 1 each time it is
seen as a target class.  The base distribution is not saved to checkpoints,
so it is reset when the model is reloaded.

In addition, this operation returns tensors `true_expected_count`
and `sampled_expected_count` representing the number of times each
of the target classes (`true_classes`) and the sampled
classes (`sampled_candidates`) is expected to occur in an average
tensor of sampled classes.  These values correspond to `Q(y|x)`
defined in [this
document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
If `unique=True`, then these are post-rejection probabilities and we
compute them approximately.

##### Args:


*  <b>`true_classes`</b>: A `Tensor` of type `int64` and shape `[batch_size,
    num_true]`. The target classes.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`num_sampled`</b>: An `int`.  The number of classes to randomly sample per batch.
*  <b>`unique`</b>: A `bool`. Determines whether all sampled classes in a batch are
    unique.
*  <b>`range_max`</b>: An `int`. The number of possible classes.
*  <b>`seed`</b>: An `int`. An operation-specific seed. Default is 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`sampled_candidates`</b>: A tensor of type `int64` and shape `[num_sampled]`.
    The sampled classes.
*  <b>`true_expected_count`</b>: A tensor of type `float`.  Same shape as
    `true_classes`. The expected counts under the sampling distribution
    of each of `true_classes`.
*  <b>`sampled_expected_count`</b>: A tensor of type `float`. Same shape as
    `sampled_candidates`. The expected counts under the sampling distribution
    of each of `sampled_candidates`.


- - -

### `tf.nn.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file='', distortion=1.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=(), seed=None, name=None)` {#fixed_unigram_candidate_sampler}

Samples a set of classes using the provided (fixed) base distribution.

This operation randomly samples a tensor of sampled classes
(`sampled_candidates`) from the range of integers `[0, range_max]`.

The elements of `sampled_candidates` are drawn without replacement
(if `unique=True`) or with replacement (if `unique=False`) from
the base distribution.

The base distribution is read from a file or passed in as an
in-memory array. There is also an option to skew the distribution by
applying a distortion power to the weights.

In addition, this operation returns tensors `true_expected_count`
and `sampled_expected_count` representing the number of times each
of the target classes (`true_classes`) and the sampled
classes (`sampled_candidates`) is expected to occur in an average
tensor of sampled classes.  These values correspond to `Q(y|x)`
defined in [this
document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
If `unique=True`, then these are post-rejection probabilities and we
compute them approximately.

##### Args:


*  <b>`true_classes`</b>: A `Tensor` of type `int64` and shape `[batch_size,
    num_true]`. The target classes.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`num_sampled`</b>: An `int`.  The number of classes to randomly sample per batch.
*  <b>`unique`</b>: A `bool`. Determines whether all sampled classes in a batch are
    unique.
*  <b>`range_max`</b>: An `int`. The number of possible classes.
*  <b>`vocab_file`</b>: Each valid line in this file (which should have a CSV-like
    format) corresponds to a valid word ID. IDs are in sequential order,
    starting from num_reserved_ids. The last entry in each line is expected
    to be a value corresponding to the count or relative probability. Exactly
    one of `vocab_file` and `unigrams` needs to be passed to this operation.
*  <b>`distortion`</b>: The distortion is used to skew the unigram probability
    distribution.  Each weight is first raised to the distortion's power
    before adding to the internal unigram distribution. As a result,
    `distortion = 1.0` gives regular unigram sampling (as defined by the vocab
    file), and `distortion = 0.0` gives a uniform distribution.
*  <b>`num_reserved_ids`</b>: Optionally some reserved IDs can be added in the range
    `[0, num_reserved_ids]` by the users. One use case is that a special
    unknown word token is used as ID 0. These IDs will have a sampling
    probability of 0.
*  <b>`num_shards`</b>: A sampler can be used to sample from a subset of the original
    range in order to speed up the whole computation through parallelism. This
    parameter (together with `shard`) indicates the number of partitions that
    are being used in the overall computation.
*  <b>`shard`</b>: A sampler can be used to sample from a subset of the original range
    in order to speed up the whole computation through parallelism. This
    parameter (together with `num_shards`) indicates the particular partition
    number of the operation, when partitioning is being used.
*  <b>`unigrams`</b>: A list of unigram counts or probabilities, one per ID in
    sequential order. Exactly one of `vocab_file` and `unigrams` should be
    passed to this operation.
*  <b>`seed`</b>: An `int`. An operation-specific seed. Default is 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`sampled_candidates`</b>: A tensor of type `int64` and shape `[num_sampled]`.
    The sampled classes.
*  <b>`true_expected_count`</b>: A tensor of type `float`.  Same shape as
    `true_classes`. The expected counts under the sampling distribution
    of each of `true_classes`.
*  <b>`sampled_expected_count`</b>: A tensor of type `float`. Same shape as
    `sampled_candidates`. The expected counts under the sampling distribution
    of each of `sampled_candidates`.



### Miscellaneous candidate sampling utilities

- - -

### `tf.nn.compute_accidental_hits(true_classes, sampled_candidates, num_true, seed=None, name=None)` {#compute_accidental_hits}

Compute the position ids in `sampled_candidates` matching `true_classes`.

In Candidate Sampling, this operation facilitates virtually removing
sampled classes which happen to match target classes.  This is done
in Sampled Softmax and Sampled Logistic.

See our [Candidate Sampling Algorithms
Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf).

We presuppose that the `sampled_candidates` are unique.

We call it an 'accidental hit' when one of the target classes
matches one of the sampled classes.  This operation reports
accidental hits as triples `(index, id, weight)`, where `index`
represents the row number in `true_classes`, `id` represents the
position in `sampled_candidates`, and weight is `-FLOAT_MAX`.

The result of this op should be passed through a `sparse_to_dense`
operation, then added to the logits of the sampled classes. This
removes the contradictory effect of accidentally sampling the true
target classes as noise classes for the same example.

##### Args:


*  <b>`true_classes`</b>: A `Tensor` of type `int64` and shape `[batch_size,
    num_true]`. The target classes.
*  <b>`sampled_candidates`</b>: A tensor of type `int64` and shape `[num_sampled]`.
    The sampled_candidates output of CandidateSampler.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`seed`</b>: An `int`. An operation-specific seed. Default is 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`indices`</b>: A `Tensor` of type `int32` and shape `[num_accidental_hits]`.
    Values indicate rows in `true_classes`.
*  <b>`ids`</b>: A `Tensor` of type `int64` and shape `[num_accidental_hits]`.
    Values indicate positions in `sampled_candidates`.
*  <b>`weights`</b>: A `Tensor` of type `float` and shape `[num_accidental_hits]`.
    Each value is `-FLOAT_MAX`.



## Other Functions and Classes
- - -

### `tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)` {#batch_normalization}

Batch normalization.

As described in http://arxiv.org/abs/1502.03167.
Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
`scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):

\\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)

`mean`, `variance`, `offset` and `scale` are all expected to be of one of two
shapes:
  * In all generality, they can have the same number of dimensions as the
    input `x`, with identical sizes as `x` for the dimensions that are not
    normalized over (the 'depth' dimension(s)), and dimension 1 for the
    others which are being normalized over.
    `mean` and `variance` in this case would typically be the outputs of
    `tf.nn.moments(..., keep_dims=True)` during training, or running averages
    thereof during inference.
  * In the common case where the 'depth' dimension is the last dimension in
    the input tensor `x`, they may be one dimensional tensors of the same
    size as the 'depth' dimension.
    This is the case for example for the common `[batch, depth]` layout of
    fully-connected layers, and `[batch, height, width, depth]` for
    convolutions.
    `mean` and `variance` in this case would typically be the outputs of
    `tf.nn.moments(..., keep_dims=False)` during training, or running averages
    thereof during inference.

##### Args:


*  <b>`x`</b>: Input `Tensor` of arbitrary dimensionality.
*  <b>`mean`</b>: A mean `Tensor`.
*  <b>`variance`</b>: A variance `Tensor`.
*  <b>`offset`</b>: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
    None. If present, will be added to the normalized tensor.
*  <b>`scale`</b>: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
    `None`. If present, the scale is applied to the normalized tensor.
*  <b>`variance_epsilon`</b>: A small float number to avoid dividing by 0.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  the normalized, scaled, offset tensor.


- - -

### `tf.nn.depthwise_conv2d_native(input, filter, strides, padding, name=None)` {#depthwise_conv2d_native}

Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
a different filter to each input channel (expanding from 1 channel to
`channel_multiplier` channels for each), then concatenates the results
together. Thus, the output has `in_channels * channel_multiplier` channels.

for k in 0..in_channels-1
  for q in 0..channel_multiplier-1
    output[b, i, j, k * channel_multiplier + q] =
      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                        filter[di, dj, k, q]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
*  <b>`filter`</b>: A `Tensor`. Must have the same type as `input`.
*  <b>`strides`</b>: A list of `ints`.
    1-D of length 4.  The stride of the sliding window for each dimension
    of `input`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


