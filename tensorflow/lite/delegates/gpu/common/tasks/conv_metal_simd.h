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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_METAL_SIMD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_METAL_SIMD_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"

namespace tflite {
namespace gpu {

class ConvolutionMetalSimd : public GPUOperation {
 public:
  ConvolutionMetalSimd() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  ConvolutionMetalSimd(ConvolutionMetalSimd&& kernel) = default;
  ConvolutionMetalSimd& operator=(ConvolutionMetalSimd&& kernel) = default;
  ConvolutionMetalSimd(const ConvolutionMetalSimd&) = delete;
  ConvolutionMetalSimd& operator=(const ConvolutionMetalSimd&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = DeduceDataTypeFromPrecision(definition_.precision);
    desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
    desc.output_group_size = 4;
    return desc;
  }

  struct ConvParams {
    int3 work_group_size;
    int3 work_group_launch_order;
    bool linear_spatial;  // spatial dimensions are Width/Height/Depth
    int slices_per_thread;
    bool x_kernel_is_1 = true;
    bool y_kernel_is_1 = true;
    bool z_kernel_is_1 = true;

    // must be 32 * k
    int GetSpatialThreadsCount() const {
      if (linear_spatial) {
        return work_group_size.x;
      } else {
        return work_group_size.x * work_group_size.y;
      }
    }

    int GetX4SlicesCount() const {
      if (linear_spatial) {
        return work_group_size.y;
      } else {
        return work_group_size.z;
      }
    }
  };

  ConvParams params_;

 private:
  explicit ConvolutionMetalSimd(const OperationDef& definition)
      : GPUOperation(definition) {}
  friend ConvolutionMetalSimd CreateConvolutionMetalSimd(
      const OperationDef& definition, const BHWC& dst_shape,
      const Convolution2DAttributes& attr, const GpuInfo& gpu_info);
};

ConvolutionMetalSimd CreateConvolutionMetalSimd(
    const OperationDef& definition, const BHWC& dst_shape,
    const Convolution2DAttributes& attr, const GpuInfo& gpu_info);

bool IsConvolutionMetalSimdSupported(const GpuInfo& gpu_info,
                                     const OperationDef& definition,
                                     const Convolution2DAttributes& attr);

bool IsGoodTaskSizeForAppleConvSimd(const BHWC& dst_shape,
                                    const GpuInfo& gpu_info);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_METAL_SIMD_H_
