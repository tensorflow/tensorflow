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

#ifndef TENSORFLOW_CORE_KERNELS_HISTOGRAM_OP_H_
#define TENSORFLOW_CORE_KERNELS_HISTOGRAM_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, typename Tout>
struct HistogramFixedWidthFunctor {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        const typename TTypes<T, 1>::ConstTensor& value_range,
                        int32 nbins, typename TTypes<Tout, 1>::Tensor& out);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HISTOGRAM_OP_H_
