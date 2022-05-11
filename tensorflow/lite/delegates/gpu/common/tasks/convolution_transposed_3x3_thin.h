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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class ConvolutionTransposed3x3Thin : public GPUOperation {
 public:
  ConvolutionTransposed3x3Thin() = default;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed3x3Thin(ConvolutionTransposed3x3Thin&& operation) =
      default;
  ConvolutionTransposed3x3Thin& operator=(
      ConvolutionTransposed3x3Thin&& operation) = default;
  ConvolutionTransposed3x3Thin(const ConvolutionTransposed3x3Thin&) = delete;
  ConvolutionTransposed3x3Thin& operator=(const ConvolutionTransposed3x3Thin&) =
      delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = DeduceDataTypeFromPrecision(definition_.precision);
    desc.layout = weights_layout_;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

 private:
  ConvolutionTransposed3x3Thin(const GpuInfo& gpu_info,
                               const OperationDef& definition,
                               const ConvolutionTransposedAttributes& attr);

  friend ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed3x3Thin
  CreateConvolutionTransposed3x3ThinDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  void UploadWeights(
      const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights);

  std::vector<int> GetSpatialWeightsRemap() const;

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                int src_depth, int dst_depth);

  WeightsLayout weights_layout_;
};

bool IsConvolutionTransposed3x3ThinSupported(
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3ThinDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
