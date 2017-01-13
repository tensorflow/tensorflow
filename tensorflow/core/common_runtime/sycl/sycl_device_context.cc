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

#if TENSORFLOW_USE_SYCL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/common_runtime/sycl/sycl_device_context.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

void SYCLDeviceContext::CopyCPUTensorToDevice(const Tensor *cpu_tensor,
                                              Device *device,
                                              Tensor *device_tensor,
                                              StatusCallback done) const {
  const int64 total_bytes = cpu_tensor->TotalBytes();
  if (total_bytes > 0) {
    const void *src_ptr = DMAHelper::base(cpu_tensor);
    void *dst_ptr = DMAHelper::base(device_tensor);
    switch (cpu_tensor->dtype()) {
    case DT_FLOAT:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<float *>(dst_ptr), static_cast<const float *>(src_ptr),
          total_bytes);
      break;
    case DT_DOUBLE:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<double *>(dst_ptr), static_cast<const double *>(src_ptr),
          total_bytes);
      break;
    case DT_INT32:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<int32 *>(dst_ptr), static_cast<const int32 *>(src_ptr),
          total_bytes);
      break;
    case DT_INT64:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<int64 *>(dst_ptr), static_cast<const int64 *>(src_ptr),
          total_bytes);
      break;
    case DT_HALF:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<Eigen::half *>(dst_ptr),
          static_cast<const Eigen::half *>(src_ptr), total_bytes);
      break;
    case DT_COMPLEX64:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<std::complex<float> *>(dst_ptr),
          static_cast<const std::complex<float> *>(src_ptr), total_bytes);
      break;
    case DT_COMPLEX128:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<std::complex<double> *>(dst_ptr),
          static_cast<const std::complex<double> *>(src_ptr), total_bytes);
      break;
    case DT_INT8:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<int8 *>(dst_ptr), static_cast<const int8 *>(src_ptr),
          total_bytes);
      break;
    case DT_INT16:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<int16 *>(dst_ptr), static_cast<const int16 *>(src_ptr),
          total_bytes);
      break;
    case DT_UINT8:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<uint8 *>(dst_ptr), static_cast<const uint8 *>(src_ptr),
          total_bytes);
      break;
    case DT_UINT16:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<uint16 *>(dst_ptr), static_cast<const uint16 *>(src_ptr),
          total_bytes);
      break;
    case DT_BOOL:
      device->eigen_sycl_device()->memcpyHostToDevice(
          static_cast<bool *>(dst_ptr), static_cast<const bool *>(src_ptr),
          total_bytes);
      break;
    default:
      assert(false && "unsupported type");
    }
  }
  device->eigen_sycl_device()->synchronize();
  done(Status::OK());
}

void SYCLDeviceContext::CopyDeviceTensorToCPU(const Tensor *device_tensor,
                                              StringPiece edge_name,
                                              Device *device,
                                              Tensor *cpu_tensor,
                                              StatusCallback done) {
  const int64 total_bytes = device_tensor->TotalBytes();
  if (total_bytes > 0) {
    const void* src_ptr = DMAHelper::base(device_tensor);
    void* dst_ptr = DMAHelper::base(cpu_tensor);
    switch (device_tensor->dtype()) {
    case DT_FLOAT:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<float *>(dst_ptr), static_cast<const float *>(src_ptr),
          total_bytes);
      break;
    case DT_DOUBLE:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<double *>(dst_ptr), static_cast<const double *>(src_ptr),
          total_bytes);
      break;
    case DT_INT32:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<int32 *>(dst_ptr), static_cast<const int32 *>(src_ptr),
          total_bytes);
      break;
    case DT_INT64:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<int64 *>(dst_ptr), static_cast<const int64 *>(src_ptr),
          total_bytes);
      break;
    case DT_HALF:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<Eigen::half *>(dst_ptr),
          static_cast<const Eigen::half *>(src_ptr), total_bytes);
      break;
    case DT_COMPLEX64:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<std::complex<float> *>(dst_ptr),
          static_cast<const std::complex<float> *>(src_ptr), total_bytes);
      break;
    case DT_COMPLEX128:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<std::complex<double> *>(dst_ptr),
          static_cast<const std::complex<double> *>(src_ptr), total_bytes);
      break;
    case DT_INT8:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<int8 *>(dst_ptr), static_cast<const int8 *>(src_ptr),
          total_bytes);
      break;
    case DT_INT16:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<int16 *>(dst_ptr), static_cast<const int16 *>(src_ptr),
          total_bytes);
      break;
    case DT_UINT8:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<uint8 *>(dst_ptr), static_cast<const uint8 *>(src_ptr),
          total_bytes);
      break;
    case DT_UINT16:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<uint16 *>(dst_ptr), static_cast<const uint16 *>(src_ptr),
          total_bytes);
      break;
    case DT_BOOL:
      device->eigen_sycl_device()->memcpyDeviceToHost(
          static_cast<bool *>(dst_ptr), static_cast<const bool *>(src_ptr),
          total_bytes);
      break;
    default:
      assert(false && "unsupported type");
    }
  }
  device->eigen_sycl_device()->synchronize();
  done(Status::OK());
}

}  // namespace tensorflow
#endif // TENSORFLOW_USE_SYCL
