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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class ConvolutionTransposed3x3Thin : public GPUOperation {
 public:
  ConvolutionTransposed3x3Thin() = default;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed3x3Thin(ConvolutionTransposed3x3Thin&& operation);
  ConvolutionTransposed3x3Thin& operator=(
      ConvolutionTransposed3x3Thin&& operation);
  ConvolutionTransposed3x3Thin(const ConvolutionTransposed3x3Thin&) = delete;
  ConvolutionTransposed3x3Thin& operator=(const ConvolutionTransposed3x3Thin&) =
      delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.layout = WeightsLayout::kOICustomSSpatialI4O4;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

 private:
  explicit ConvolutionTransposed3x3Thin(
      const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  friend ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed3x3Thin
  CreateConvolutionTransposed3x3ThinDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);

  std::vector<int> GetSpatialWeightsRemap() const;

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                int src_depth, int dst_depth);
};

template <DataType T>
void ConvolutionTransposed3x3Thin::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = 3;  //  This operation support only 3x3 kernel
  const int kernel_y = 3;
  const int flt4_count = kernel_x * kernel_y * src_depth * dst_depth * 4;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  const int flt4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = flt4_size * flt4_count;
  desc.data.resize(desc.size);

  if (f32_weights) {
    float4* gpu_data = reinterpret_cast<float4*>(desc.data.data());
    RearrangeWeightsToOICustomSpatialI4O4(weights, GetSpatialWeightsRemap(),
                                          absl::MakeSpan(gpu_data, flt4_count));
  } else {
    half4* gpu_data = reinterpret_cast<half4*>(desc.data.data());
    RearrangeWeightsToOICustomSpatialI4O4(weights, GetSpatialWeightsRemap(),
                                          absl::MakeSpan(gpu_data, flt4_count));
  }

  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvolutionTransposed3x3ThinSupported(
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3ThinDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_3X3_THIN_H_
