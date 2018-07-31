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

void SetZeroFunctor<Eigen::ThreadPoolDevice, string>::operator()(
    const Eigen::ThreadPoolDevice& d, typename TTypes<string>::Flat out) {
  out.device(d) = out.constant(string());
}

// Explicit instantiations.
#define DEFINE_SETZERO_CPU(T) \
  template struct SetZeroFunctor<Eigen::ThreadPoolDevice, T>;
DEFINE_SETZERO_CPU(bool);
DEFINE_SETZERO_CPU(Eigen::half);
DEFINE_SETZERO_CPU(bfloat16);
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
DEFINE_SETZERO_CPU(Variant);
#undef DEFINE_SETZERO_CPU

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
void SetZeroFunctor<Eigen::SyclDevice, T>::operator()(
    const Eigen::SyclDevice& d, typename TTypes<T>::Flat out) {
  To32Bit(out).device(d) = To32Bit(out).constant(T(0));
}

#define DEFINE_SETZERO_SYCL(T) \
  template struct SetZeroFunctor<Eigen::SyclDevice, T>;
DEFINE_SETZERO_SYCL(bool);
DEFINE_SETZERO_SYCL(float);
DEFINE_SETZERO_SYCL(double);
DEFINE_SETZERO_SYCL(uint8);
DEFINE_SETZERO_SYCL(int8);
DEFINE_SETZERO_SYCL(uint16);
DEFINE_SETZERO_SYCL(int16);
DEFINE_SETZERO_SYCL(int32);
DEFINE_SETZERO_SYCL(int64);
#undef DEFINE_SETZERO_SYCL
#endif  // TENSORFLOW_USE_SYCL

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
DEFINE_SETONE_CPU(uint8);
DEFINE_SETONE_CPU(int8);
DEFINE_SETONE_CPU(uint16);
DEFINE_SETONE_CPU(int16);
DEFINE_SETONE_CPU(int32);
DEFINE_SETONE_CPU(int64);
DEFINE_SETONE_CPU(complex64);
DEFINE_SETONE_CPU(complex128);
#undef DEFINE_SETONE_CPU

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
void SetOneFunctor<Eigen::SyclDevice, T>::operator()(
    const Eigen::SyclDevice& d, typename TTypes<T>::Flat out) {
  out.device(d) = out.constant(T(1));
}

#define DEFINE_SETONE_SYCL(T) \
  template struct SetOneFunctor<Eigen::SyclDevice, T>;
DEFINE_SETONE_SYCL(float);
DEFINE_SETONE_SYCL(bool);
DEFINE_SETONE_SYCL(double);
#undef DEFINE_SETONE_SYCL
#endif  // TENSORFLOW_USE_SYCL

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
#undef DEFINE_FILL_CPU

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct FillFunctor<Eigen::SyclDevice, T> {
  void operator()(const Eigen::SyclDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> rank1{1};
#else
    Eigen::IndexList<Eigen::type2index<1> > rank1;
#endif
    const int size = out.dimension(0);
    Eigen::array<int, 1> broadcast_dims{size};

    To32Bit(out).device(d) = in.reshape(rank1).broadcast(broadcast_dims);
  }
};

#define DEFINE_FILL_SYCL(T) template struct FillFunctor<Eigen::SyclDevice, T>;
DEFINE_FILL_SYCL(float);
DEFINE_FILL_SYCL(double);
TF_CALL_INTEGRAL_TYPES(DEFINE_FILL_SYCL)
#undef DEFINE_FILL_SYCL
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor
}  // namespace tensorflow
