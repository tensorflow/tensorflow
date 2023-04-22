/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_INPLACE_OPS_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_INPLACE_OPS_FUNCTOR_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace functor {

template <typename Device>
Status DoParallelConcat(const Device& device, const Tensor& value, int32 loc,
                        Tensor* output);

// Inplace update/add/sub values in 'y'. It computes
//   y[i, :] = v if op is I_UPDATE
//   y[i, :] += v if op is I_ADD
//   y[i, :] -= v if op is I_SUB
// Returns an error if the operation fails.
enum InplaceOpType {
  I_UPDATE,  // x = y
  I_ADD,     // x += y
  I_SUB,     // x -= y
};
template <typename Device>
Status DoInplace(const Device& device, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y);
// Copies x into y.
template <typename Device>
Status DoCopy(const Device& device, const Tensor& x, Tensor* y);

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_INPLACE_OPS_FUNCTOR_H_
