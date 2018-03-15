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

#ifndef TENSORFLOW_KERNELS_MATRIX_SET_DIAG_OP_H_
#define TENSORFLOW_KERNELS_MATRIX_SET_DIAG_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct MatrixSetDiag {
  static void Compute(OpKernelContext* context, const Device& d,
                      typename TTypes<T, 3>::ConstTensor input,
                      typename TTypes<T, 2>::ConstTensor diag,
                      typename TTypes<T, 3>::Tensor output);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MATRIX_SET_DIAG_OP_H_
