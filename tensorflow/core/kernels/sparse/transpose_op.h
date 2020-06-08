/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_TRANSPOSE_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_TRANSPOSE_OP_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct maybe_conj_inplace {
  static void run(const Device& d, Tensor* t) {}
};

template <typename Device>
struct maybe_conj_inplace<Device, complex64> {
  static void run(const Device& d, Tensor* t) {
    functor::UnaryFunctor<Device, functor::conj<complex64>> conj;
    conj(d, t->flat<complex64>() /*out*/,
         const_cast<const Tensor*>(t)->flat<complex64>() /*in*/);
  }
};

template <typename Device>
struct maybe_conj_inplace<Device, complex128> {
  static void run(const Device& d, Tensor* t) {
    functor::UnaryFunctor<Device, functor::conj<complex128>> conj;
    conj(d, t->flat<complex128>() /*out*/,
         const_cast<const Tensor*>(t)->flat<complex128>() /*in*/);
  }
};

template <typename Device, typename T>
struct maybe_conj {
  static void run(const Device& d, const Tensor& in, Tensor* out) { *out = in; }
};

template <typename Device>
struct maybe_conj<Device, complex64> {
  static void run(const Device& d, const Tensor& in, Tensor* out) {
    functor::UnaryFunctor<Device, functor::conj<complex64>> conj;
    conj(d, out->flat<complex64>() /*out*/, in.flat<complex64>() /*in*/);
  }
};

template <typename Device>
struct maybe_conj<Device, complex128> {
  static void run(const Device& d, const Tensor& in, Tensor* out) {
    functor::UnaryFunctor<Device, functor::conj<complex128>> conj;
    conj(d, out->flat<complex128>() /*out*/, in.flat<complex128>() /*in*/);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_TRANSPOSE_OP_H_
