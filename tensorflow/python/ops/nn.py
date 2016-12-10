# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""## Activation Functions.

The activation ops provide different types of nonlinearities for use in neural
networks.  These include smooth nonlinearities (`sigmoid`, `tanh`, `elu`,
`softplus`, and `softsign`), continuous but not everywhere differentiable
functions (`relu`, `relu6`, `crelu` and `relu_x`), and random regularization
(`dropout`).

All activation ops apply componentwise, and produce a tensor of the same
shape as the input tensor.

@@relu
@@relu6
@@crelu
@@elu
@@softplus
@@softsign
@@dropout
@@bias_add
@@sigmoid
@@tanh

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

@@convolution
@@conv2d
@@depthwise_conv2d
@@depthwise_conv2d_native
@@separable_conv2d
@@atrous_conv2d
@@atrous_conv2d_transpose
@@conv2d_transpose
@@conv1d
@@conv3d
@@conv3d_transpose
@@conv2d_backprop_filter
@@conv2d_backprop_input
@@conv3d_backprop_filter_v2
@@depthwise_conv2d_native_backprop_filter
@@depthwise_conv2d_native_backprop_input

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

@@avg_pool
@@max_pool
@@max_pool_with_argmax
@@avg_pool3d
@@max_pool3d
@@fractional_avg_pool
@@fractional_max_pool
@@pool

## Morphological filtering

Morphological operators are non-linear filters used in image processing.

[Greyscale morphological dilation
](https://en.wikipedia.org/wiki/Dilation_(morphology))
is the max-sum counterpart of standard sum-product convolution:

    output[b, y, x, c] =
        max_{dy, dx} input[b,
                           strides[1] * y + rates[1] * dy,
                           strides[2] * x + rates[2] * dx,
                           c] +
                     filter[dy, dx, c]

The `filter` is usually called structuring function. Max-pooling is a special
case of greyscale morphological dilation when the filter assumes all-zero
values (a.k.a. flat structuring function).

[Greyscale morphological erosion
](https://en.wikipedia.org/wiki/Erosion_(morphology))
is the min-sum counterpart of standard sum-product convolution:

    output[b, y, x, c] =
        min_{dy, dx} input[b,
                           strides[1] * y - rates[1] * dy,
                           strides[2] * x - rates[2] * dx,
                           c] -
                     filter[dy, dx, c]

Dilation and erosion are dual to each other. The dilation of the input signal
`f` by the structuring signal `g` is equal to the negation of the erosion of
`-f` by the reflected `g`, and vice versa.

Striding and padding is carried out in exactly the same way as in standard
convolution. Please refer to the `Convolution` section for details.

@@dilation2d
@@erosion2d
@@with_space_to_batch

## Normalization

Normalization is useful to prevent neurons from saturating when inputs may
have varying scale, and to aid generalization.

@@l2_normalize
@@local_response_normalization
@@sufficient_statistics
@@normalize_moments
@@moments
@@weighted_moments
@@fused_batch_norm
@@batch_normalization
@@batch_norm_with_global_normalization

## Losses

The loss ops measure error between two tensors, or between a tensor and zero.
These can be used for measuring accuracy of a network in a regression task
or for regularization purposes (weight decay).

@@l2_loss
@@log_poisson_loss

## Classification

TensorFlow provides several operations that help you perform classification.

@@sigmoid_cross_entropy_with_logits
@@softmax
@@log_softmax
@@softmax_cross_entropy_with_logits
@@sparse_softmax_cross_entropy_with_logits
@@weighted_cross_entropy_with_logits

## Embeddings

TensorFlow provides library support for looking up values in embedding
tensors.

@@embedding_lookup
@@embedding_lookup_sparse

## Recurrent Neural Networks

TensorFlow provides a number of methods for constructing Recurrent
Neural Networks.  Most accept an `RNNCell`-subclassed object
(see the documentation for `tf.contrib.rnn`).

@@dynamic_rnn
@@rnn
@@state_saving_rnn
@@bidirectional_dynamic_rnn
@@bidirectional_rnn
@@raw_rnn

## Connectionist Temporal Classification (CTC)

@@ctc_loss
@@ctc_greedy_decoder
@@ctc_beam_search_decoder

## Evaluation

The evaluation ops are useful for measuring the performance of a network.
They are typically used at evaluation time.

@@top_k
@@in_top_k

## Candidate Sampling

Do you want to train a multiclass or multilabel model with thousands
or millions of output classes (for example, a language model with a
large vocabulary)?  Training with a full Softmax is slow in this case,
since all of the classes are evaluated for every training example.
Candidate Sampling training algorithms can speed up your step times by
only considering a small randomly-chosen subset of contrastive classes
(called candidates) for each batch of training examples.

See our
[Candidate Sampling Algorithms Reference](../../extras/candidate_sampling.pdf)

### Sampled Loss Functions

TensorFlow provides the following sampled loss functions for faster training.

@@nce_loss
@@sampled_softmax_loss

### Candidate Samplers

TensorFlow provides the following samplers for randomly sampling candidate
classes when using one of the sampled loss functions above.

@@uniform_candidate_sampler
@@log_uniform_candidate_sampler
@@learned_unigram_candidate_sampler
@@fixed_unigram_candidate_sampler

### Miscellaneous candidate sampling utilities

@@compute_accidental_hits

### Quantization ops

@@quantized_conv2d
@@quantized_relu_x
@@quantized_max_pool
@@quantized_avg_pool

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys as _sys

# pylint: disable=unused-import
from tensorflow.python.ops import ctc_ops as _ctc_ops
from tensorflow.python.ops import embedding_ops as _embedding_ops
from tensorflow.python.ops import nn_grad as _nn_grad
from tensorflow.python.ops import nn_ops as _nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
# pylint: enable=unused-import
from tensorflow.python.util.all_util import remove_undocumented

# Bring more nn-associated functionality into this package.
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.ctc_ops import *
from tensorflow.python.ops.nn_impl import *
from tensorflow.python.ops.nn_ops import *
from tensorflow.python.ops.candidate_sampling_ops import *
from tensorflow.python.ops.embedding_ops import *
from tensorflow.python.ops.rnn import *
# pylint: enable=wildcard-import


# TODO(cwhipkey): sigmoid and tanh should not be exposed from tf.nn.
_allowed_symbols = [
    "zero_fraction",  # documented in training.py
    # Modules whitelisted for reference through tf.nn.
    # TODO(cwhipkey): migrate callers to use the submodule directly.
    "rnn_cell",
    "seq2seq",
    # Symbols whitelisted for export without documentation.
    # TODO(cwhipkey): review these and move to contrib or expose through
    # documentation.
    "all_candidate_sampler",  # Excluded in gen_docs_combined.
    "lrn",  # Excluded in gen_docs_combined.
    "relu_layer",  # Excluded in gen_docs_combined.
    "xw_plus_b",  # Excluded in gen_docs_combined.
]

remove_undocumented(__name__, _allowed_symbols,
                    [_sys.modules[__name__], _ctc_ops, _nn_ops, _nn_grad,
                     rnn_cell, seq2seq])
