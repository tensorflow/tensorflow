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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_DEPTHWISE_CONV_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_DEPTHWISE_CONV_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {
namespace metal {

// Depth Wise Convolution for kernel 3x3
// require:
//   channels_multiplier = 1;
//   kernel_size = 3x3;
//   dilation.y = 1;
//   stride.y = 2;
class DepthWiseConv3x3Stride2 : public GPUOperation {
 public:
  DepthWiseConv3x3Stride2() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  DepthWiseConv3x3Stride2(DepthWiseConv3x3Stride2&& kernel) = default;
  DepthWiseConv3x3Stride2& operator=(DepthWiseConv3x3Stride2&& kernel) =
      default;
  DepthWiseConv3x3Stride2(const DepthWiseConv3x3Stride2&) = delete;
  DepthWiseConv3x3Stride2& operator=(const DepthWiseConv3x3Stride2&) = delete;

 private:
  explicit DepthWiseConv3x3Stride2(const OperationDef& definition)
      : GPUOperation(definition) {}
  friend DepthWiseConv3x3Stride2 CreateDepthWiseConv3x3Stride2(
      const OperationDef& definition,
      const DepthwiseConvolution2DAttributes& attr);
};

DepthWiseConv3x3Stride2 CreateDepthWiseConv3x3Stride2(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr);

// TODO(impjdi): Move it inside module.
bool CheckDepthWiseConv3x3Stride2Support(
    const DepthwiseConvolution2DAttributes& attr);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_DEPTHWISE_CONV_H_
