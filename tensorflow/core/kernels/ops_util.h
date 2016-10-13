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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

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
