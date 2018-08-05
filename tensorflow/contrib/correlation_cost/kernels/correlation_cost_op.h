/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORRELATION_COST_OP_H_
#define TENSORFLOW_CORRELATION_COST_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct CorrelationCostFunctor {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, Tensor* output_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format);
};

template <typename Device, typename T>
struct CorrelationCostGradFunctor {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, const Tensor& topdiff_t,
                    Tensor* output_a_gradient_t, Tensor* output_b_gradient_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORRELATION_COST_OP_H_
