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

#ifndef TENSORFLOW_CORE_KERNELS_SPACETOBATCH_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_SPACETOBATCH_FUNCTOR_H_

#include <type_traits>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Maximum number of non-collapsible blocked dimensions supported by the
// {SpaceToBatch,BatchToSpace}ND operation.  To change the limit, modify this
// constant and the TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS macro definition
// below.
constexpr int kMaxSpaceToBatchBlockDims = 4;

// Expands to:
//   MACRO(1, ## __VA_ARGS__)
//   ...
//   MACRO(kMaxSpaceToBatchBlockDims, ## __VA_ARGS__)
//
// Note: The space between the number and the comma is necessary for proper GCC
// comma handling: https://gcc.gnu.org/onlinedocs/cpp/Variadic-Macros.html
#define TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(MACRO, ...) \
  MACRO(1 /**/, ##__VA_ARGS__)                              \
  MACRO(2 /**/, ##__VA_ARGS__)                              \
  MACRO(3 /**/, ##__VA_ARGS__)                              \
  MACRO(4 /**/, ##__VA_ARGS__)                              \
  /**/

namespace internal {
namespace spacetobatch {

template <typename InputType, typename OutputType>
void SubtleMustCopyFlatHelper(const Tensor& t, OutputType* output) {
  const int64_t num_elements = t.shape().num_elements();
  output->resize(num_elements);
  auto eigen_vec = t.flat<InputType>();
  for (int64_t i = 0; i < num_elements; ++i) {
    (*output)[i] = SubtleMustCopy(eigen_vec(i));
  }
}

// Copies flat contents of `t` to std::vector-like `*output`, which is resized
// as needed.  `OutputType` may be either `std::vector<int64_t>` or
// `gtl::InlinedVector<int64_t>`.
//
// Precondition: t.dtype() must be either DT_INT32 or DT_INT64.
template <typename OutputType>
void SubtleMustCopyFlat(const Tensor& t, OutputType* output) {
  if (t.dtype() == DT_INT32) {
    SubtleMustCopyFlatHelper<int32, OutputType>(t, output);
  } else {
    SubtleMustCopyFlatHelper<int64_t, OutputType>(t, output);
  }
}

}  // namespace spacetobatch
}  // namespace internal

namespace functor {

// Functor used by {SpaceToBatch,BatchToSpace}{ND,}Op to do the conversion.
//
// If B2S is false, then this performs the space-to-batch conversion.  If B2S is
// true, then this performs the inverse batch-to-space conversion.
template <typename Device, typename T, int NUM_BLOCK_DIMS, bool B2S = false>
struct SpaceToBatchFunctor {
  using InputT = typename std::conditional<B2S, T, const T>::type;
  using OutputT = typename std::conditional<B2S, const T, T>::type;
  // Implements the space to batch conversion.
  //
  // space_tensor: input tensor of space-to-batch operation.  If B2S = false,
  //     then this is the input to the conversion.  If B2S = true, then this
  //     is the output of the conversion.
  // block_size: array of shape [NUM_BLOCK_DIMS] specifying the block sizes for
  //     dimensions 1 through NUM_BLOCK_DIMS.
  // paddings: row-major array of shape [NUM_BLOCK_DIMS, 2] specifying the
  //     start and end padding for dimensions 1 through NUM_BLOCK_DIMS.
  // batch_tensor: output tensor of the space-to-batch operation.  If
  //     B2S = false, then this is the output of the conversion.  If B2S = true,
  //     then this is the input to the conversion.
  //
  // The caller must ensure that the dimensions of the tensors are correct.
  Status operator()(
      const Device& d,
      typename TTypes<InputT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64_t block_shape[NUM_BLOCK_DIMS],
      const int64_t paddings[NUM_BLOCK_DIMS * 2],
      typename TTypes<OutputT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPACETOBATCH_FUNCTOR_H_
