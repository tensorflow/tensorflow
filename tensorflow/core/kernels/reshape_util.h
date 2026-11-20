
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_RESHAPE_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_RESHAPE_UTIL_H_

#include <cstdint>
#include <functional>

#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class OpKernelContext;
class Tensor;

// Reshapes the input indices and input shape to the target shape.
// Note: This template is explicitly instantiated for CPU devices only. Use
// ReshapeSparseTensorGpuAsync for GPU devices.
template <typename Device>
void ReshapeSparseTensor(OpKernelContext *context,
                         const Tensor &input_indices_in,
                         const Tensor &input_shape_in,
                         const Tensor &target_shape_in, int output_indices_idx,
                         int output_shape_idx);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// GPU version of ReshapeSparseTensor, for use in an AsyncOpKernel. The input
// indices are in device memory and are validated on the device, so any error
// is reported asynchronously. `done` is called once the op has completed.
void ReshapeSparseTensorGpuAsync(OpKernelContext* context,
                                 const Tensor& input_indices_in,
                                 const Tensor& input_shape_in,
                                 const Tensor& target_shape_in,
                                 int output_indices_idx, int output_shape_idx,
                                 std::function<void()> done);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

// Computes `output_indices` from `input_indices`. Returns an error if any input
// index is out of bounds for `input_shape`.
template <typename Device>
struct ReshapeSparseTensorFunctor {
  absl::Status operator()(
      OpKernelContext *context, const TensorShape &input_shape,
      const TensorShape &output_shape,
      typename TTypes<int64_t>::ConstMatrix input_indices,
      typename TTypes<int64_t>::Matrix output_indices) const;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// A GPU kernel cannot return an error, so the GPU functor takes an extra
// device scalar `first_invalid_row`. It is set to the smallest row of
// `input_indices` that is out of bounds for `input_shape`, or to the number of
// rows if all of them are valid. Output rows for invalid input rows are not
// written. The functor allocates device copies of the shapes in
// `*input_shape_dev` and `*output_shape_dev`; the caller must keep them alive
// until the kernel has completed.
template <>
struct ReshapeSparseTensorFunctor<Eigen::GpuDevice> {
  absl::Status operator()(OpKernelContext* context,
                          const TensorShape& input_shape,
                          const TensorShape& output_shape,
                          typename TTypes<int64_t>::ConstMatrix input_indices,
                          typename TTypes<int64_t>::Matrix output_indices,
                          typename TTypes<int64_t>::Scalar first_invalid_row,
                          Tensor* input_shape_dev,
                          Tensor* output_shape_dev) const;
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RESHAPE_UTIL_H_
