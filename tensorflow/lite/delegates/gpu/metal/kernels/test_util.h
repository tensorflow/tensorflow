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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_TEST_UTIL_H_

#import <Metal/Metal.h>

#include <map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"

namespace tflite {
namespace gpu {
namespace metal {

class MetalExecutionEnvironment : public TestExecutionEnvironment {
 public:
  MetalExecutionEnvironment() = default;
  ~MetalExecutionEnvironment() = default;

  std::vector<CalculationsPrecision> GetSupportedPrecisions() const override;
  std::vector<TensorStorageType> GetSupportedStorages(
      DataType data_type) const override;
  std::vector<TensorStorageType> GetSupportedStoragesWithHWZeroClampSupport(
      DataType data_type) const override;

  const GpuInfo& GetGpuInfo() const { return device_.GetInfo(); }

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWC>& dst_sizes,
      const std::vector<TensorFloat32*>& dst_cpu) override;

  absl::Status ExecuteGPUOperation(
      const std::vector<Tensor5DFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWDC>& dst_sizes,
      const std::vector<Tensor5DFloat32*>& dst_cpu) override;

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation) override;

 private:
  MetalDevice device_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_TEST_UTIL_H_
