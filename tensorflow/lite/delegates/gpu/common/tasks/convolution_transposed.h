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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_H_

#include <cstdint>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class ConvolutionTransposed : public GPUOperation {
 public:
  ConvolutionTransposed() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed(ConvolutionTransposed&& operation) = default;
  ConvolutionTransposed& operator=(ConvolutionTransposed&& operation) = default;
  ConvolutionTransposed(const ConvolutionTransposed&) = delete;
  ConvolutionTransposed& operator=(const ConvolutionTransposed&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.layout = weights_layout_;
    desc.output_group_size = block_size_.w;
    return desc;
  }

 private:
  friend ConvolutionTransposed CreateConvolutionTransposed(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed CreateConvolutionTransposed3D(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposed3DAttributes& attr);
  friend ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  ConvolutionTransposed(const OperationDef& definition,
                        const ConvolutionTransposedAttributes& attr,
                        const GpuInfo& gpu_info, bool weights_are_buffer);
  ConvolutionTransposed(const OperationDef& definition,
                        const ConvolutionTransposed3DAttributes& attr,
                        const GpuInfo& gpu_info, bool weights_are_buffer);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                     bool weights_are_buffer);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights,
                     bool weights_are_buffer);

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                const GpuInfo& gpu_info,
                                                bool weights_are_buffer,
                                                const int4& block_size);
  int4 stride_;
  int4 block_size_ = int4(1, 1, 1, 1);  // WHDS
  WeightsLayout weights_layout_;
};

template <DataType T>
void ConvolutionTransposed::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, bool weights_are_buffer) {
  const int flt_count =
      GetTotalElementsCountForLayout(GetWeightsDescription(), weights.shape);
  DataType weights_type = definition_.precision == CalculationsPrecision::F32
                              ? DataType::FLOAT32
                              : DataType::FLOAT16;

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_type));
  RearrangeWeights(weights, GetWeightsDescription(), weights_type,
                   absl::MakeSpan(weights_data));

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = weights_type;
    desc.element_size = 16;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    const int dst_depth =
        AlignByN(DivideRoundUp(weights.shape.o, 4), block_size_.w);
    const int src_depth = DivideRoundUp(weights.shape.i, 4);
    const int kernel_x = weights.shape.w;
    const int kernel_y = weights.shape.h;
    int texture_width = dst_depth;
    int texture_height = src_depth * kernel_x * kernel_y;
    int sub_size = SizeOf(weights_type) * 4 * texture_width * texture_height;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = weights_type;
      desc.size = int2(texture_width, texture_height);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), weights_data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

template <DataType T>
void ConvolutionTransposed::UploadWeights(
    const tflite::gpu::Tensor<OHWDI, T>& weights, bool weights_are_buffer) {
  const int dst_depth =
      AlignByN(DivideRoundUp(weights.shape.o, 4), block_size_.w);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;
  const int kernel_z = weights.shape.d;

  const int elements_count =
      kernel_x * kernel_y * kernel_z * src_depth * dst_depth * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;
  std::vector<uint8_t> data(float4_size * elements_count);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    if (weights_are_buffer) {
      RearrangeWeightsToODHWIOGroupI4O4(weights, block_size_.w,
                                        absl::MakeSpan(ptr, elements_count));
    } else {
      RearrangeWeightsToI4DHWIOOGroupO4(weights, block_size_.w,
                                        absl::MakeSpan(ptr, elements_count));
    }
  } else {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    if (weights_are_buffer) {
      RearrangeWeightsToODHWIOGroupI4O4(weights, block_size_.w,
                                        absl::MakeSpan(ptr, elements_count));
    } else {
      RearrangeWeightsToI4DHWIOOGroupO4(weights, block_size_.w,
                                        absl::MakeSpan(ptr, elements_count));
    }
  }

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 16;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    int texture_width = dst_depth;
    int texture_height = src_depth * kernel_x * kernel_y * kernel_z;
    int sub_size = float4_size * texture_width * texture_height;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
      desc.size = int2(texture_width, texture_height);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

ConvolutionTransposed CreateConvolutionTransposed(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed CreateConvolutionTransposed3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposed3DAttributes& attr);

ConvolutionTransposed CreateConvolutionTransposedDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_H_
