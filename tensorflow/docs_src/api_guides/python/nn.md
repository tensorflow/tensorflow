# Neural Network

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Activation Functions

The activation ops provide different types of nonlinearities for use in neural
networks. These include smooth nonlinearities (`sigmoid`, `tanh`, `elu`, `selu`,
`softplus`, and `softsign`), continuous but not everywhere differentiable
functions (`relu`, `relu6`, `crelu` and `relu_x`), and random regularization
(`dropout`).

All activation ops apply componentwise, and produce a tensor of the same
shape as the input tensor.

*   @{tf.nn.relu}
*   @{tf.nn.relu6}
*   @{tf.nn.crelu}
*   @{tf.nn.elu}
*   @{tf.nn.selu}
*   @{tf.nn.softplus}
*   @{tf.nn.softsign}
*   @{tf.nn.dropout}
*   @{tf.nn.bias_add}
*   @{tf.sigmoid}
*   @{tf.tanh}

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

Ignoring channels for the moment, assume that the 4-D `input` has shape
`[batch, in_height, in_width, ...]` and the 4-D `filter` has shape
`[filter_height, filter_width, ...]`. The spatial semantics of the
convolution ops depend on the padding scheme chosen: `'SAME'` or `'VALID'`.
Note that the padding values are always zero.

First, consider the `'SAME'` padding scheme. A detailed explanation of the
reasoning behind it is given in
[these notes](#Notes_on_SAME_Convolution_Padding). Here, we summarize the
mechanics of this padding scheme. When using `'SAME'`, the output height and
width are computed as:

    out_height = ceil(float(in_height) / float(strides[1]))
    out_width  = ceil(float(in_width) / float(strides[2]))

The total padding applied along the height and width is computed as:

    if (in_height % strides[1] == 0):
      pad_along_height = max(filter_height - strides[1], 0)
    else:
      pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if (in_width % strides[2] == 0):
      pad_along_width = max(filter_width - strides[2], 0)
    else:
      pad_along_width = max(filter_width - (in_width % strides[2]), 0)

Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

Note that the division by 2 means that there might be cases when the padding on
both sides (top vs bottom, right vs left) are off by one. In this case, the
bottom and right sides always get the one additional padded pixel. For example,
when `pad_along_height` is 5, we pad 2 pixels at the top and 3 pixels at the
bottom. Note that this is different from existing libraries such as cuDNN and
Caffe, which explicitly specify the number of padded pixels and always pad the
same number of pixels on both sides.

For the `'VALID'` scheme, the output height and width are computed as:

    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

and no padding is used.

Given the output size and the padding, the output can be computed as

$$    output[b, i, j, :] =
        sum_{d_i, d_j} input[b, strides[1] * i + d_i - pad_{top},\
                           strides[2] * j + d_j - pad_{left}, ...] *
                     filter[d_i, d_j,\ ...]$$

where any value outside the original input image region are considered zero (
i.e. we pad zero values around the border of the image).

Since `input` is 4-D, each `input[b, i, j, :]` is a vector.  For `conv2d`, these
vectors are multiplied by the `filter[di, dj, :, :]` matrices to produce new
vectors.  For `depthwise_conv_2d`, each scalar component `input[b, i, j, k]`
is multiplied by a vector `filter[di, dj, k]`, and all the vectors are
concatenated.

*   @{tf.nn.convolution}
*   @{tf.nn.conv2d}
*   @{tf.nn.depthwise_conv2d}
*   @{tf.nn.depthwise_conv2d_native}
*   @{tf.nn.separable_conv2d}
*   @{tf.nn.atrous_conv2d}
*   @{tf.nn.atrous_conv2d_transpose}
*   @{tf.nn.conv2d_transpose}
*   @{tf.nn.conv1d}
*   @{tf.nn.conv3d}
*   @{tf.nn.conv3d_transpose}
*   @{tf.nn.conv2d_backprop_filter}
*   @{tf.nn.conv2d_backprop_input}
*   @{tf.nn.conv3d_backprop_filter_v2}
*   @{tf.nn.depthwise_conv2d_native_backprop_filter}
*   @{tf.nn.depthwise_conv2d_native_backprop_input}

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

*   @{tf.nn.avg_pool}
*   @{tf.nn.max_pool}
*   @{tf.nn.max_pool_with_argmax}
*   @{tf.nn.avg_pool3d}
*   @{tf.nn.max_pool3d}
*   @{tf.nn.fractional_avg_pool}
*   @{tf.nn.fractional_max_pool}
*   @{tf.nn.pool}

## Morphological filtering

Morphological operators are non-linear filters used in image processing.

[Greyscale morphological dilation
](https://en.wikipedia.org/wiki/Dilation_(morphology))
is the max-sum counterpart of standard sum-product convolution:

$$    output[b, y, x, c] =
        max_{dy, dx} input[b,
                           strides[1] * y + rates[1] * dy,
                           strides[2] * x + rates[2] * dx,
                           c] +
                     filter[dy, dx, c]$$

The `filter` is usually called structuring function. Max-pooling is a special
case of greyscale morphological dilation when the filter assumes all-zero
values (a.k.a. flat structuring function).

[Greyscale morphological erosion
](https://en.wikipedia.org/wiki/Erosion_(morphology))
is the min-sum counterpart of standard sum-product convolution:

$$    output[b, y, x, c] =
        min_{dy, dx} input[b,
                           strides[1] * y - rates[1] * dy,
                           strides[2] * x - rates[2] * dx,
                           c] -
                     filter[dy, dx, c]$$

Dilation and erosion are dual to each other. The dilation of the input signal
`f` by the structuring signal `g` is equal to the negation of the erosion of
`-f` by the reflected `g`, and vice versa.

Striding and padding is carried out in exactly the same way as in standard
convolution. Please refer to the `Convolution` section for details.

*   @{tf.nn.dilation2d}
*   @{tf.nn.erosion2d}
*   @{tf.nn.with_space_to_batch}

## Normalization

Normalization is useful to prevent neurons from saturating when inputs may
have varying scale, and to aid generalization.

*   @{tf.nn.l2_normalize}
*   @{tf.nn.local_response_normalization}
*   @{tf.nn.sufficient_statistics}
*   @{tf.nn.normalize_moments}
*   @{tf.nn.moments}
*   @{tf.nn.weighted_moments}
*   @{tf.nn.fused_batch_norm}
*   @{tf.nn.batch_normalization}
*   @{tf.nn.batch_norm_with_global_normalization}

## Losses

The loss ops measure error between two tensors, or between a tensor and zero.
These can be used for measuring accuracy of a network in a regression task
or for regularization purposes (weight decay).

*   @{tf.nn.l2_loss}
*   @{tf.nn.log_poisson_loss}

## Classification

TensorFlow provides several operations that help you perform classification.

*   @{tf.nn.sigmoid_cross_entropy_with_logits}
*   @{tf.nn.softmax}
*   @{tf.nn.log_softmax}
*   @{tf.nn.softmax_cross_entropy_with_logits}
*   @{tf.nn.softmax_cross_entropy_with_logits_v2} - identical to the base
    version, except it allows gradient propagation into the labels.
*   @{tf.nn.sparse_softmax_cross_entropy_with_logits}
*   @{tf.nn.weighted_cross_entropy_with_logits}

## Embeddings

TensorFlow provides library support for looking up values in embedding
tensors.

*   @{tf.nn.embedding_lookup}
*   @{tf.nn.embedding_lookup_sparse}

## Recurrent Neural Networks

TensorFlow provides a number of methods for constructing Recurrent
Neural Networks.  Most accept an `RNNCell`-subclassed object
(see the documentation for `tf.contrib.rnn`).

*   @{tf.nn.dynamic_rnn}
*   @{tf.nn.bidirectional_dynamic_rnn}
*   @{tf.nn.raw_rnn}

## Connectionist Temporal Classification (CTC)

*   @{tf.nn.ctc_loss}
*   @{tf.nn.ctc_greedy_decoder}
*   @{tf.nn.ctc_beam_search_decoder}

## Evaluation

The evaluation ops are useful for measuring the performance of a network.
They are typically used at evaluation time.

*   @{tf.nn.top_k}
*   @{tf.nn.in_top_k}

## Candidate Sampling

Do you want to train a multiclass or multilabel model with thousands
or millions of output classes (for example, a language model with a
large vocabulary)?  Training with a full Softmax is slow in this case,
since all of the classes are evaluated for every training example.
Candidate Sampling training algorithms can speed up your step times by
only considering a small randomly-chosen subset of contrastive classes
(called candidates) for each batch of training examples.

See our
[Candidate Sampling Algorithms
Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)

### Sampled Loss Functions

TensorFlow provides the following sampled loss functions for faster training.

*   @{tf.nn.nce_loss}
*   @{tf.nn.sampled_softmax_loss}

### Candidate Samplers

TensorFlow provides the following samplers for randomly sampling candidate
classes when using one of the sampled loss functions above.

*   @{tf.nn.uniform_candidate_sampler}
*   @{tf.nn.log_uniform_candidate_sampler}
*   @{tf.nn.learned_unigram_candidate_sampler}
*   @{tf.nn.fixed_unigram_candidate_sampler}

### Miscellaneous candidate sampling utilities

*   @{tf.nn.compute_accidental_hits}

### Quantization ops

*   @{tf.nn.quantized_conv2d}
*   @{tf.nn.quantized_relu_x}
*   @{tf.nn.quantized_max_pool}
*   @{tf.nn.quantized_avg_pool}

## Notes on SAME Convolution Padding

In these notes, we provide more background on the use of the `'SAME'` padding
scheme for convolution operations.

Tensorflow uses the smallest possible padding to achieve the desired output
size. To understand what is done, consider the \\(1\\)-dimensional case. Denote
\\(n_i\\) and \\(n_o\\) the input and output sizes, respectively, and denote the
kernel size \\(k\\) and stride \\(s\\). As discussed in the
[Convolution section](#Convolution), for `'SAME'`,
\\(n_o = \left \lceil{\frac{n_i}{s}}\right \rceil\\).

To achieve a desired output size \\(n_o\\), we need to pad the input such that the
output size after a `'VALID'` convolution is \\(n_o\\). In other words, we need to
have padding \\(p_i\\) such that:

\begin{equation}
\left \lceil{\frac{n_i + p_i - k + 1}{s}}\right \rceil = n_o
\label{eq:tf_pad_1}
\end{equation}

What is the smallest \\(p_i\\) that we could possibly use? In general, \\(\left
\lceil{\frac{x}{a}}\right \rceil = b\\) (with \\(a > 0\\)) means that \\(b-1 <
\frac{x}{a} \leq b\\), and the smallest integer \\(x\\) we can choose to satisfy
this is \\(x = a\cdot (b-1) + 1\\). The same applies to our problem; we need
\\(p_i\\) such that:

\begin{equation}
n_i + p_i - k + 1 = s\cdot (n_o - 1) + 1
\label{eq:tf_pad_2}
\end{equation}

which leads to:

\begin{equation}
p_i = s\cdot (n_o - 1) + k - n_i
\label{eq:tf_pad_3}
\end{equation}

Note that this might lead to negative \\(p_i\\), since in some cases we might
already have more input samples than we actually need. Thus,

\begin{equation}
p_i = max(s\cdot (n_o - 1) + k - n_i, 0)
\label{eq:tf_pad_4}
\end{equation}

Remember that, for `'SAME'` padding,
\\(n_o = \left \lceil{\frac{n_i}{s}}\right \rceil\\), as mentioned above.
We need to analyze in detail two cases:

- \\(n_i \text{ mod } s = 0\\)

In this simple case, \\(n_o = \frac{n_i}{s}\\), and the expression for \\(p_i\\)
becomes:

\begin{equation}
p_i = max(k - s, 0)
\label{eq:tf_pad_5}
\end{equation}

- \\(n_i \text{ mod } s \neq 0\\)

This case is more involved to parse. First, we write:

\begin{equation}
n_i = s\cdot\left \lceil{\frac{n_i}{s}}\right \rceil
- s \left(\left \lceil{\frac{n_i}{s}}\right \rceil -
          \left \lfloor{\frac{n_i}{s}}\right \rfloor\right)
+ (n_i \text{ mod } s)
\label{eq:tf_pad_6}
\end{equation}

For the case where \\((n_i \text{ mod } s) \neq 0\\), we have \\(\left
\lceil{\frac{n_i}{s}}\right \rceil -\left \lfloor{\frac{n_i}{s}}\right \rfloor =
1\\), leading to:

\begin{equation}
n_i = s\cdot\left \lceil{\frac{n_i}{s}}\right \rceil
- s
+ (n_i \text{ mod } s)
\label{eq:tf_pad_7}
\end{equation}

We can use this expression to substitute \\(n_o = \left
\lceil{\frac{n_i}{s}}\right \rceil\\) and get:

$$\begin{align}
p_i &= max\left(s\cdot \left(\frac{n_i + s - (n_i \text{ mod } s)}{s}
  - 1\right) + k - n_i, 0\right) \nonumber\\
&= max(n_i + s - (n_i \text{ mod } s) - s + k - n_i,0) \nonumber \\
&= max(k - (n_i \text{ mod } s),0)
\label{eq:tf_pad_8}
\end{align}$$

### Final expression

Putting all together, the total padding used by tensorflow's convolution with
`'SAME'` mode is:

$$\begin{align}
p_i =
 \begin{cases}
 max(k - s, 0),  & \text{if $(n_i \text{ mod } s) = 0$} \\
 max(k - (n_i \text{ mod } s),0), & \text{if $(n_i \text{ mod } s) \neq 0$}
 \end{cases}
 \label{eq:tf_pad_9}
\end{align}$$

This expression is exactly equal to the ones presented for `pad_along_height`
and `pad_along_width` in the [Convolution section](#Convolution).
