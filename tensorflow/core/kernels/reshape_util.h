
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

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class OpKernelContext;
class Tensor;

// Reshapes the input indices and input shape to the target shape.
// Note: This template is explicitly instantiated for CPU and GPU devices.
template <typename Device>
void ReshapeSparseTensor(OpKernelContext *context,
                         const Tensor &input_indices_in,
                         const Tensor &input_shape_in,
                         const Tensor &target_shape_in, int output_indices_idx,
                         int output_shape_idx);

namespace functor {

template <typename Device>
struct ReshapeSparseTensorFunctor {
  Status operator()(OpKernelContext *context, const TensorShape &input_shape,
                    const TensorShape &output_shape,
                    typename TTypes<int64>::ConstMatrix input_indices,
                    typename TTypes<int64>::Matrix output_indices) const;
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RESHAPE_UTIL_H_
