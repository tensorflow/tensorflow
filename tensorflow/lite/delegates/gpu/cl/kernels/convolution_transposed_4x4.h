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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_

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

class ConvolutionTransposed4x4 : public GPUOperation {
 public:
  ConvolutionTransposed4x4() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed4x4(ConvolutionTransposed4x4&& operation);
  ConvolutionTransposed4x4& operator=(ConvolutionTransposed4x4&& operation);
  ConvolutionTransposed4x4(const ConvolutionTransposed4x4&) = delete;
  ConvolutionTransposed4x4& operator=(const ConvolutionTransposed4x4&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.layout = WeightsLayout::kOICustomSSpatialI4O4;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC,
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
  };

 private:
  ConvolutionTransposed4x4(const OperationDef& definition,
                           const GpuInfo& gpu_info);

  friend ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                     WeightsUploadType weights_upload_type);

  std::vector<int> GetSpatialWeightsRemap() const;

  std::string GenerateConvolutionTransposedCode(
      const OperationDef& op_def, WeightsUploadType weights_upload_type);
};

template <DataType T>
void ConvolutionTransposed4x4::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights,
    WeightsUploadType weights_upload_type) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = 4;  //  This operation support only 4x4 kernel
  const int kernel_y = 4;
  const int flt4_count = kernel_x * kernel_y * src_depth * dst_depth * 4;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  const int flt4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type =
      weights_upload_type ==
              ConvolutionTransposed4x4::WeightsUploadType::CONSTANT_MEM
          ? MemoryType::CONSTANT
          : MemoryType::GLOBAL;
  desc.size = flt4_size * flt4_count;
  desc.data.resize(desc.size);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(desc.data.data());
    RearrangeWeightsToOICustomSpatialI4O4(weights, GetSpatialWeightsRemap(),
                                          absl::MakeSpan(ptr, flt4_count));
  } else {
    half4* ptr = reinterpret_cast<half4*>(desc.data.data());
    RearrangeWeightsToOICustomSpatialI4O4(weights, GetSpatialWeightsRemap(),
                                          absl::MakeSpan(ptr, flt4_count));
  }

  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvolutionTransposed4x4Supported(
    const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_4X4_H_
