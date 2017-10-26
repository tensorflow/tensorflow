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

#ifndef TENSORFLOW_KERNELS_CWISE_OP_CLIP_H_
#define TENSORFLOW_KERNELS_CWISE_OP_CLIP_H_

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
namespace functor {
// Unary functor for clip [Tensor, Scalar, Scalar]
template <typename Device, typename T>
struct UnaryClipOp {
  void operator()(const Device &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const;
};

// Binary functor for clip [Tensor, Scalar, Tensor]
template <typename Device, typename T>
struct BinaryRightClipOp {
  void operator()(const Device &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const;
};

// Binary functor for clip [Tensor, Tensor, Scalar]
template <typename Device, typename T>
struct BinaryLeftClipOp {
  void operator()(const Device &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const;
};

// Ternary functor for clip [Tensor, Tensor, Tensor]
template <typename Device, typename T>
struct TernaryClipOp {
  void operator()(const Device &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const;
};
}
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CWISE_OP_CLIP_H_
