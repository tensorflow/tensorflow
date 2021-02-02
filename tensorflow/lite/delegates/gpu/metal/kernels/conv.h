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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONV_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONV_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {
namespace metal {

class ConvolutionGeneric : public GPUOperation {
 public:
  enum class WeightsUploadType {
    PRIVATE_MEM_SIMD8_BROADCAST,
    PRIVATE_MEM_SIMD16_BROADCAST,
    PRIVATE_MEM_SIMD32_BROADCAST,
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
  };

  enum class WeightsInnerBlockLayout {
    O4I4,
    I4O4,
  };

  struct ConvParams {
    int3 block_size;
    int3 work_group_size;
    int3 work_group_launch_order;
    int src_depth_loop_size;
    bool need_src_loop = true;
    bool need_dst_loop = true;
    bool linear_wh;
    bool linear_whs;
    WeightsUploadType weights_upload_type;
    WeightsInnerBlockLayout weight_layout;
    bool different_weights_for_height = false;
    bool x_kernel_is_1;
    bool y_kernel_is_1;
  };

  ConvolutionGeneric() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;

  // Move only
  ConvolutionGeneric(ConvolutionGeneric&& kernel) = default;
  ConvolutionGeneric& operator=(ConvolutionGeneric&& kernel) = default;
  ConvolutionGeneric(const ConvolutionGeneric&) = delete;
  ConvolutionGeneric& operator=(const ConvolutionGeneric&) = delete;

 private:
  explicit ConvolutionGeneric(const OperationDef& definition)
      : GPUOperation(definition) {}
  friend ConvolutionGeneric CreateConvolutionGeneric(
      const OperationDef& definition, const BHWC& dst_shape,
      const Convolution2DAttributes& attr, const GpuInfo& gpu_info);

  friend ConvolutionGeneric CreateConvolutionWino4x4To6x6(
      const OperationDef& definition, const BHWC& dst_shape,
      const Convolution2DAttributes& attr, const GpuInfo& gpu_info);

  ConvParams params_;
};

ConvolutionGeneric CreateConvolutionGeneric(const OperationDef& definition,
                                            const BHWC& dst_shape,
                                            const Convolution2DAttributes& attr,
                                            const GpuInfo& gpu_info);

ConvolutionGeneric CreateConvolutionWino4x4To6x6(
    const OperationDef& definition, const BHWC& dst_shape,
    const Convolution2DAttributes& attr, const GpuInfo& gpu_info);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_CONV_H_
