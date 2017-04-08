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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_PARTIAL_REDUCTION_OPS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_PARTIAL_REDUCTION_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {

namespace reduce_functions {

template <typename T> T sum(T a,T b);
template <typename T> T prod(T a,T b);
template <typename T> T max(T a,T b);
template <typename T> T min(T a,T b);
template <typename T> T zero();
template <typename T> T one();
template <typename T> T infinity();
template <typename T> T negative_infinity();

} // namespace reduce_functions

// BaseFunctor for definition of PartialReductionOp
template <typename Device, typename T, typename Index, T beginning(), T reduce(T,T)>
struct PartialReductionFunctor{
  virtual ~PartialReductionFunctor(){}
  virtual void operator()(OpKernelContext* ctx, const Device& d,
                  typename TTypes<Index>::ConstMatrix indices,
                  typename TTypes<T>::ConstMatrix data,
                  typename TTypes<T>::Matrix output);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_PARTIAL_REDUCTION_OPS_H_
