/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_GPU_VARIABLES_TABLE_H_
#define TENSORFLOW_CORE_TFRT_UTILS_GPU_VARIABLES_TABLE_H_

#include "tensorflow/core/tfrt/utils/device_variables_table.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"

namespace tfrt {
namespace gpu {

// This is for creating/getting GpuVariablesTable object in the execution
// context at runtime.
constexpr char kGpuVariablesTableResourceName[] = "GpuVariablesTableResource";

// A variable table that keeps track of the device copies of GPU host tensors.
class GpuVariablesTable
    : public DeviceVariablesTable<tensorflow::tfrt_stub::FallbackTensor,
                                  tensorflow::tfrt_stub::FallbackTensor> {
 private:
  const void* GetHostTensorDataPtr(
      const tensorflow::tfrt_stub::FallbackTensor& host_tensor) override {
    return host_tensor.tensor().data();
  }
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_UTILS_GPU_VARIABLES_TABLE_H_
