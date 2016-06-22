/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_OPS_UTIL_H_
#define TENSORFLOW_KERNELS_OPS_UTIL_H_

// This file contains utilities for various operations.

#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// GetWindowedOutputSize(): Given an input tensor, kernel, stride and padding
// type, the function computes the output and padding dimensions.
//
// For example, ignoring batches or multiple features, a 1D convolution
// takes as input a 1D tensor of shape (H), and convolves it with a filter of
// shape (K).
//
// It also takes in a few additional parameters:
//
// Stride (S): the stride with which we apply the filters. This is the offset
// between locations where we apply the filters. A larger stride
// means that the output will be spatially smaller.
//
// Padding (P): the padding we apply to the input tensor along each
// dimension. This is usually used to make sure that the spatial dimensions
// do not shrink when we progress with convolutions. Two types of padding are
// often used:
//   SAME: the pad value is computed so that the output will have size H/S.
//   VALID: no padding is carried out.
// The padded area is zero-filled.
//
// The output dimensions for convolution and many other operations, when given
// all the parameters above, are as follows:
// - When Padding = SAME: the output size is (H'), where
//     H' = ceil(float(H) / float(S))
//   where ceil is the ceiling function. The number of padded cells
//   is computed as:
//     Pc = ((H' - 1) * S + K - H) / 2
//   When the stride is 1, the expression simplifies to
//     H' = H, Pc = (K-1)/2.
//   This is where SAME comes from - the output has the same size as the input
//   has.
//
// - When Padding = VALID: the output size is computed as
//     H' = ceil(float(H - K + 1) / float(S))
//   and the number of padded cells is always zero.
//   When the stride is 1, the expression simplifies to
//     H' = H-K+1.
//
// For convolution, mathematically, the output value at location (r')
// is the inner product of two vectors: the chunk of input at
//    ((r'*S-Pr) : (r'*S-Pr+K)),
// and the filter.
//
// For 2D and 3D convolutions, the spatial dimensions are orthogonal, so the
// size and padding of each spatial dimension can be computed by calling
// GetWindowedOutputSize separately for each dimension.
//
Status GetWindowedOutputSize(int64 input_size, int64 filter_size, int64 stride,
                             Padding padding_type, int64* output_size,
                             int64* padding_size);

// Returns the same output dimensions as in GetWindowedOutputSize, but returns
// verbose padding dimensions (before/after). Any excess padding
// (caused by an odd padding size value) is added to the 'padding_after'
// dimension.
Status GetWindowedOutputSizeVerbose(int64 input_size, int64 filter_size,
                                    int64 stride, Padding padding_type,
                                    int64* output_size, int64* padding_before,
                                    int64* padding_after);

// Given an input tensor, kernel, stride and padding type, populates the 3D size
// of the output tensor and padding to be applied to the input tensor at the
// lower end of every dimension. Use for 3D convolutions, where the input data
// is padded with zeros, as well as for 3D avg/max pooling, where the input data
// is padded with invalid values that are not considered for pooling.
Status Get3dOutputSize(const std::array<int64, 3>& input,
                       const std::array<int64, 3>& window,
                       const std::array<int64, 3>& strides,
                       Padding padding_type, std::array<int64, 3>* output,
                       std::array<int64, 3>* padding);

// Calculates broadcast starting index and size.  For SAME padding, addition
// padding could be applied to right, left, top and bottom.  Depending on the
// current index, input size, kernel size, stride, padding size, the starting
// index and size for broadcast for that dimension are different from the
// current index and kernel size.
// This is mainly used by gradient algorithms for pooling operations.
Status GetBroadcastSize(const int index, const int in_size, const int ksize,
                        const int stride, const int pad_size, int* bindex,
                        int* bsize);

// Converts Brain's Padding to Eigen's PaddingType.
Eigen::PaddingType BrainPadding2EigenPadding(Padding padding);

// Given a shape 's' of a tensor of type T. Returns true iff the
// number of bytes occupied by each dim 0 (i.e., &tensor(i + 1, ...) -
// &tensor(i, ...)) is multiple of EIGEN_MAX_ALIGN_BYTES.
template <typename T>
bool IsInnerDimsSizeAligned(const TensorShape& s) {
  if (s.dims() == 0) return false;
  const int64 dim0_size = s.dim_size(0);
  if (dim0_size == 0) return false;
  const int64 bytes_per_dim0 = (s.num_elements() / dim0_size) * sizeof(T);
  return bytes_per_dim0 % EIGEN_MAX_ALIGN_BYTES == 0;
}

// Returns <suffix> sanitized to have only [a-zA-Z0-9-_].
string SanitizeThreadSuffix(string suffix);

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_OPS_UTIL_H_
