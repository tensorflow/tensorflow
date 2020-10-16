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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_H_

#include <cstdint>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class ConvolutionTransposed : public GPUOperation {
 public:
  ConvolutionTransposed() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const DeviceInfo& device_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed(ConvolutionTransposed&& operation);
  ConvolutionTransposed& operator=(ConvolutionTransposed&& operation);
  ConvolutionTransposed(const ConvolutionTransposed&) = delete;
  ConvolutionTransposed& operator=(const ConvolutionTransposed&) = delete;

 private:
  friend ConvolutionTransposed CreateConvolutionTransposed(
      const DeviceInfo& device_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed CreateConvolutionTransposed3D(
      const DeviceInfo& device_info, const OperationDef& definition,
      const ConvolutionTransposed3DAttributes& attr);
  ConvolutionTransposed(const OperationDef& definition,
                        const ConvolutionTransposedAttributes& attr,
                        const DeviceInfo& device_info);
  ConvolutionTransposed(const OperationDef& definition,
                        const ConvolutionTransposed3DAttributes& attr,
                        const DeviceInfo& device_info);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                     bool weights_are_buffer);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights,
                     bool weights_are_buffer);

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                const DeviceInfo& device_info,
                                                bool weights_are_buffer,
                                                const int4& block_size);
  int4 stride_;
  int4 block_size_ = int4(1, 1, 1, 1);  // WHDS
};

template <DataType T>
void ConvolutionTransposed::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, bool weights_are_buffer) {
  const int dst_depth =
      AlignByN(DivideRoundUp(weights.shape.o, 4), block_size_.w);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int elements_count = kernel_x * kernel_y * src_depth * dst_depth * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;
  std::vector<uint8_t> data(float4_size * elements_count);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    if (weights_are_buffer) {
      RearrangeWeightsToOHWIOGroupI4O4(weights, block_size_.w,
                                       absl::MakeSpan(ptr, elements_count));
    } else {
      RearrangeWeightsToI4HWIOOGroupO4(weights, block_size_.w,
                                       absl::MakeSpan(ptr, elements_count));
    }
  } else {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    if (weights_are_buffer) {
      RearrangeWeightsToOHWIOGroupI4O4(weights, block_size_.w,
                                       absl::MakeSpan(ptr, elements_count));
    } else {
      RearrangeWeightsToI4HWIOOGroupO4(weights, block_size_.w,
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
    int texture_height = src_depth * kernel_x * kernel_y;
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
    const DeviceInfo& device_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed CreateConvolutionTransposed3D(
    const DeviceInfo& device_info, const OperationDef& definition,
    const ConvolutionTransposed3DAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_H_
