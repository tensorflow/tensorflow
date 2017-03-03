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

#include "tensorflow/core/kernels/fill_functor.h"

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace functor {

template <typename T>
void SetZeroFunctor<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<T>::Flat out) {
  out.device(d) = out.constant(T(0));
}

void SetZeroFunctor<Eigen::ThreadPoolDevice, string>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<string>::Flat out) {
  out.device(d) = out.constant(string());
}

// Explicit instantiations.
#define DEFINE_SETZERO_CPU(T) \
  template struct SetZeroFunctor<Eigen::ThreadPoolDevice, T>;
DEFINE_SETZERO_CPU(bool);
DEFINE_SETZERO_CPU(Eigen::half);
DEFINE_SETZERO_CPU(float);
DEFINE_SETZERO_CPU(double);
DEFINE_SETZERO_CPU(uint8);
DEFINE_SETZERO_CPU(int8);
DEFINE_SETZERO_CPU(uint16);
DEFINE_SETZERO_CPU(int16);
DEFINE_SETZERO_CPU(int32);
DEFINE_SETZERO_CPU(int64);
DEFINE_SETZERO_CPU(complex64);
DEFINE_SETZERO_CPU(complex128);
#undef DEFINE_SETZERO_CPU

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
void SetZeroFunctor<Eigen::SyclDevice, T>::operator()(
    const Eigen::SyclDevice& d, typename TTypes<T>::Flat out) {
  out.device(d) = out.constant(T(0));
}

#define DEFINE_SETZERO_SYCL(T) \
  template struct SetZeroFunctor<Eigen::SyclDevice, T>;
DEFINE_SETZERO_SYCL(float);
DEFINE_SETZERO_SYCL(bool);
DEFINE_SETZERO_SYCL(double);
#undef DEFINE_SETZERO_SYCL
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor
}  // namespace tensorflow
