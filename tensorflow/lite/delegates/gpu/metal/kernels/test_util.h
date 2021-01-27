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

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"

namespace tflite {
namespace gpu {
namespace metal {

class SingleOpModel {
 public:
  SingleOpModel() = delete;
  SingleOpModel(Operation&& operation,
                const std::vector<TensorRef<BHWC>>& inputs,
                const std::vector<TensorRef<BHWC>>& outputs);
  virtual ~SingleOpModel() = default;

  bool PopulateTensor(int index, std::vector<float>&& data) {
    inputs_[index].data = data;
    return true;
  }

  absl::Status Invoke();

  const std::vector<float>& GetOutput(int index) const {
    return outputs_[index].data;
  }

 protected:
  GraphFloat32 graph_;
  std::vector<TensorFloat32> inputs_;
  std::vector<TensorFloat32> outputs_;
};

absl::Status CompareVectors(const std::vector<float>& reference,
                            const std::vector<float>& output, float max_error);

class MetalExecutionEnvironment : public TestExecutionEnvironment {
 public:
  MetalExecutionEnvironment() = default;
  ~MetalExecutionEnvironment() = default;

  std::vector<CalculationsPrecision> GetSupportedPrecisions() const override;
  std::vector<TensorStorageType> GetSupportedStorages() const override;
  std::vector<TensorStorageType> GetSupportedStoragesWithHWZeroClampSupport()
      const override;

  const GpuInfo& GetGpuInfo() const { return device_.GetInfo(); }

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWC>& dst_sizes,
      const std::vector<TensorFloat32*>& dst_cpu) override;

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<ComputeTaskDescriptor>&& operation,
      const std::vector<BHWC>& dst_sizes,
      const std::vector<TensorFloat32*>& dst_cpu);

  absl::Status ExecuteGPUOperation(
      const TensorFloat32& src_cpu,
      std::unique_ptr<ComputeTaskDescriptor>&& operation, const BHWC& dst_size,
      TensorFloat32* result) {
    return ExecuteGPUOperation(std::vector<TensorFloat32>{src_cpu},
                               std::move(operation), dst_size, result);
  }

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<ComputeTaskDescriptor>&& operation, const BHWC& dst_size,
      TensorFloat32* result) {
    return ExecuteGPUOperation(
        std::vector<TensorFloat32>{src_cpu}, std::move(operation),
        std::vector<BHWC>{dst_size}, std::vector<TensorFloat32*>{result});
  }

 private:
  MetalDevice device_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_TEST_UTIL_H_
