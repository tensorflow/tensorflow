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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_encode_decode.h"

namespace tensorflow {
namespace functor {

template <typename T>
void SetZeroFunctor<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<T>::Flat out) {
  out.device(d) = out.constant(T(0));
}

void SetZeroFunctor<Eigen::ThreadPoolDevice, tstring>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<tstring>::Flat out) {
  out.device(d) = out.constant(tstring());
}

// Explicit instantiations.
#define DEFINE_SETZERO_CPU(T) \
  template struct SetZeroFunctor<Eigen::ThreadPoolDevice, T>;
DEFINE_SETZERO_CPU(bool);
DEFINE_SETZERO_CPU(Eigen::half);
DEFINE_SETZERO_CPU(bfloat16);
DEFINE_SETZERO_CPU(float);
DEFINE_SETZERO_CPU(double);
DEFINE_SETZERO_CPU(uint32);
DEFINE_SETZERO_CPU(uint64);
DEFINE_SETZERO_CPU(uint8);
DEFINE_SETZERO_CPU(int8);
DEFINE_SETZERO_CPU(uint16);
DEFINE_SETZERO_CPU(int16);
DEFINE_SETZERO_CPU(int32);
DEFINE_SETZERO_CPU(int64_t);
DEFINE_SETZERO_CPU(quint8);
DEFINE_SETZERO_CPU(qint8);
DEFINE_SETZERO_CPU(quint16);
DEFINE_SETZERO_CPU(qint16);
DEFINE_SETZERO_CPU(qint32);
DEFINE_SETZERO_CPU(complex64);
DEFINE_SETZERO_CPU(complex128);
DEFINE_SETZERO_CPU(Variant);
#undef DEFINE_SETZERO_CPU


template <typename T>
void SetOneFunctor<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<T>::Flat out) {
  out.device(d) = out.constant(T(1));
}

// Explicit instantiations.
#define DEFINE_SETONE_CPU(T) \
  template struct SetOneFunctor<Eigen::ThreadPoolDevice, T>;
DEFINE_SETONE_CPU(bool);
DEFINE_SETONE_CPU(Eigen::half);
DEFINE_SETONE_CPU(bfloat16);
DEFINE_SETONE_CPU(float);
DEFINE_SETONE_CPU(double);
DEFINE_SETONE_CPU(uint32);
DEFINE_SETONE_CPU(uint64);
DEFINE_SETONE_CPU(uint8);
DEFINE_SETONE_CPU(int8);
DEFINE_SETONE_CPU(uint16);
DEFINE_SETONE_CPU(int16);
DEFINE_SETONE_CPU(int32);
DEFINE_SETONE_CPU(int64_t);
DEFINE_SETONE_CPU(complex64);
DEFINE_SETONE_CPU(complex128);
#undef DEFINE_SETONE_CPU

template <typename T>
void SetNanFunctor<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<T>::Flat out) {
  out.device(d) = out.constant(Eigen::NumTraits<T>::quiet_NaN());
}

// Explicit instantiations.
#define DEFINE_SETNAN_CPU(T) \
  template struct SetNanFunctor<Eigen::ThreadPoolDevice, T>;
TF_CALL_NUMBER_TYPES(DEFINE_SETNAN_CPU);
TF_CALL_bool(DEFINE_SETNAN_CPU);
#undef DEFINE_SETNAN_CPU

template <typename T>
struct FillFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d,
                  typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
    out.device(d) = out.constant(in());
  }
};

// Explicit instantiations.
#define DEFINE_FILL_CPU(T) \
  template struct FillFunctor<Eigen::ThreadPoolDevice, T>;

TF_CALL_ALL_TYPES(DEFINE_FILL_CPU);
DEFINE_FILL_CPU(quint8);
DEFINE_FILL_CPU(quint16);
DEFINE_FILL_CPU(qint8);
DEFINE_FILL_CPU(qint16);
DEFINE_FILL_CPU(qint32);
#undef DEFINE_FILL_CPU


}  // namespace functor
}  // namespace tensorflow
