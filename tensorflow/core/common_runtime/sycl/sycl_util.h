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

// For DMA helper
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
  inline void* GetBase(const Tensor* src) {
    return const_cast<void*>(DMAHelper::base(src));
  }

  inline void* GetBase(Tensor* dst) { return DMAHelper::base(dst); }

  inline cl::sycl::device GetSYCLDevice() {
    // Obtain list of supported devices from Eigen
    for (const auto& device :Eigen::get_sycl_supported_devices()) {
      if(device.is_gpu()) {
        // returns first found GPU
        return device;
      }
    }

    // Currently Intel GPU is not supported
    LOG(WARNING) << "No OpenCL GPU found that is supported by ComputeCpp, trying OpenCL CPU";

    for (const auto& device :Eigen::get_sycl_supported_devices()) {
      if(device.is_cpu()) {
        // returns first found CPU
        return device;
      }
    }
    // Currently Intel GPU is not supported
    LOG(FATAL) << "No OpenCL GPU nor CPU found that is supported by ComputeCpp";
  }
}

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_
