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

#if !TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
// For DMA helper
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
inline void const* GetBase(const Tensor* src) { return DMAHelper::base(src); }
inline void* GetBase(Tensor* dst) { return DMAHelper::base(dst); }

inline void SYCLmemcpy(Eigen::SyclDevice const& device,
                       Tensor const& src_tensor, Tensor* dst_tensor) {
  const size_t size = src_tensor.TotalBytes();
  void* dst_ptr = GetBase(dst_tensor);
  void const* src_ptr = GetBase(&src_tensor);

#define COPY_WITH_TYPE(T) \
  device.memcpy(dst_ptr, static_cast<T const*>(src_ptr), size);
  switch (src_tensor.dtype()) {
    case DT_COMPLEX128:
      COPY_WITH_TYPE(cl::sycl::cl_ulong2);
      break;
    case DT_DOUBLE:
    case DT_COMPLEX64:
    case DT_INT64:
      COPY_WITH_TYPE(cl::sycl::cl_ulong);
      break;
    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
      COPY_WITH_TYPE(cl::sycl::cl_uint);
      break;
    case DT_INT16:
    case DT_UINT16:
    case DT_BFLOAT16:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_HALF:
      COPY_WITH_TYPE(cl::sycl::cl_ushort);
      break;
    case DT_BOOL:
      COPY_WITH_TYPE(bool);
      break;
    case DT_UINT8:
    case DT_INT8:
    case DT_QINT8:
    case DT_QUINT8:
      COPY_WITH_TYPE(cl::sycl::cl_uchar);
      break;
    default:
      LOG(FATAL) << "Unknown data type " << src_tensor.dtype();
      break;
  }
#undef COPY_WITH_TYPE
}
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_
