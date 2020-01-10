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

#ifndef TENSORFLOW_CORE_FRAMEWORK_REGISTER_TYPES_TRAITS_H_
#define TENSORFLOW_CORE_FRAMEWORK_REGISTER_TYPES_TRAITS_H_
// This file is used by cuda code and must remain compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Remap POD types by size to equivalent proxy types. This works
// since all we are doing is copying data around.
struct UnusableProxyType;
template <typename Device, int size>
struct proxy_type_pod {
  typedef UnusableProxyType type;
};
template <>
struct proxy_type_pod<CPUDevice, 16> {
  typedef ::tensorflow::complex128 type;
};
template <>
struct proxy_type_pod<CPUDevice, 8> {
  typedef ::tensorflow::int64 type;
};
template <>
struct proxy_type_pod<CPUDevice, 4> {
  typedef ::tensorflow::int32 type;
};
template <>
struct proxy_type_pod<CPUDevice, 2> {
  typedef ::tensorflow::int16 type;
};
template <>
struct proxy_type_pod<CPUDevice, 1> {
  typedef ::tensorflow::int8 type;
};
template <>
struct proxy_type_pod<GPUDevice, 8> {
  typedef double type;
};
template <>
struct proxy_type_pod<GPUDevice, 4> {
  typedef float type;
};
template <>
struct proxy_type_pod<GPUDevice, 2> {
  typedef Eigen::half type;
};
template <>
struct proxy_type_pod<GPUDevice, 1> {
  typedef ::tensorflow::int8 type;
};

#ifdef TENSORFLOW_USE_SYCL
template <>
struct proxy_type_pod<SYCLDevice, 8> {
  typedef double type;
};
template <>
struct proxy_type_pod<SYCLDevice, 4> {
  typedef float type;
};
#endif  // TENSORFLOW_USE_SYCL

/// If POD we use proxy_type_pod, otherwise this maps to identity.
template <typename Device, typename T>
struct proxy_type {
  typedef typename std::conditional<
      std::is_arithmetic<T>::value,
      typename proxy_type_pod<Device, sizeof(T)>::type, T>::type type;
  static_assert(sizeof(type) == sizeof(T), "proxy_type_pod is not valid");
};

/// The active proxy types
#define TF_CALL_CPU_PROXY_TYPES(m)                                     \
  TF_CALL_int64(m) TF_CALL_int32(m) TF_CALL_uint16(m) TF_CALL_int16(m) \
      TF_CALL_int8(m) TF_CALL_complex128(m)
#define TF_CALL_GPU_PROXY_TYPES(m)                                    \
  TF_CALL_double(m) TF_CALL_float(m) TF_CALL_half(m) TF_CALL_int32(m) \
      TF_CALL_int8(m)
#ifdef TENSORFLOW_USE_SYCL
#define TF_CALL_SYCL_PROXY_TYPES(m) \
  TF_CALL_double(m) TF_CALL_float(m) TF_CALL_int32(m)
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_REGISTER_TYPES_TRAITS_H_
