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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_TEXTURE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_TEXTURE_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
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
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {
namespace cl {

// This convolution process BLOCK_SIZE(XxYxZ) of FLT4 values per thread.
class ConvTexture : public GPUOperation {
 public:
  ConvTexture() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const DeviceInfo& device_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments() override;
  int3 GetGridSize() const override;

  // Move only
  ConvTexture(ConvTexture&& operation);
  ConvTexture& operator=(ConvTexture&& operation);
  ConvTexture(const ConvTexture&) = delete;
  ConvTexture& operator=(const ConvTexture&) = delete;

 private:
  friend ConvTexture CreateConvTexture(const DeviceInfo& device_info,
                                       const OperationDef& definition,
                                       const Convolution2DAttributes& attr);
  friend ConvTexture CreateConvTexture(const DeviceInfo& device_info,
                                       const OperationDef& definition,
                                       const FullyConnectedAttributes& attr);

  friend ConvTexture CreateConvTextureWino4x4To6x6(
      const DeviceInfo& device_info, const OperationDef& definition,
      const Convolution2DAttributes& attr);

  ConvTexture(const OperationDef& definition,
              const Convolution2DAttributes& attr);
  explicit ConvTexture(const OperationDef& definition);
  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                  const tflite::gpu::Tensor<Linear, T>& biases);

  template <DataType T>
  void UploadDataForWinograd4x4To6x6(
      const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);

  void GenerateCode(const DeviceInfo& device_info);

  std::string GenerateConvCode(const OperationDef& op_def,
                               const int3& block_size, bool is1x1,
                               bool adreno4xx_optimization,
                               bool stride_correction,
                               bool different_weights_for_height);

  int2 kernel_size_;
  int2 stride_;
  int2 padding_;
  int2 dilation_;

  // By default in 2d convolution we have the same weights for WH dims, but in
  // some cases we need separate weights for H dimension and convolution kernel
  // requires very small modifications to support it.
  bool different_weights_for_height_;

  int3 block_size_ = int3(2, 2, 2);
};

template <DataType T>
void ConvTexture::UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                             const tflite::gpu::Tensor<Linear, T>& biases) {
  UploadWeights(weights);

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition_.GetDataType();
  desc.UploadLinearData(biases);
  args_.AddObject("biases",
                  absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
}

template <DataType T>
void ConvTexture::UploadDataForWinograd4x4To6x6(
    const tflite::gpu::Tensor<OHWI, T>& weights) {
  tflite::gpu::Tensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  UploadWeights(wino_weights);

  tflite::gpu::Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape = Linear(1);
  bias.data = {0.0f};
  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::TEXTURE_2D;
  desc.element_type = definition_.GetDataType();
  desc.UploadLinearData(bias);
  args_.AddObject("biases",
                  absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
}

template <DataType T>
void ConvTexture::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights) {
  int dst_depth = DivideRoundUp(weights.shape.o, 4);
  dst_depth = AlignByN(dst_depth, block_size_.z);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  DataType data_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;

  const int elements_count = dst_depth * src_depth * kernel_x * kernel_y * 4;
  const int float4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  std::vector<uint8_t> data(float4_size * elements_count);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    RearrangeWeightsToI4HWIOOGroupO4(weights, block_size_.z,
                                     absl::MakeSpan(ptr, elements_count));
  } else {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    RearrangeWeightsToI4HWIOOGroupO4(weights, block_size_.z,
                                     absl::MakeSpan(ptr, elements_count));
  }

  const int texture_width = dst_depth;
  const int texture_height = src_depth * kernel_x * kernel_y;
  const int sub_size = float4_size * texture_width * texture_height;
  for (int i = 0; i < 4; ++i) {
    Texture2DDescriptor desc;
    desc.element_type = data_type;
    desc.size = int2(texture_width, texture_height);
    desc.data.resize(sub_size);
    memcpy(desc.data.data(), data.data() + sub_size * i, sub_size);
    const std::string name = "weights" + std::to_string(i);
    args_.AddObject(name,
                    absl::make_unique<Texture2DDescriptor>(std::move(desc)));
  }
}

ConvTexture CreateConvTexture(const DeviceInfo& device_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr);

ConvTexture CreateConvTexture(const DeviceInfo& device_info,
                              const OperationDef& definition,
                              const FullyConnectedAttributes& attr);

ConvTexture CreateConvTextureWino4x4To6x6(const DeviceInfo& device_info,
                                          const OperationDef& definition,
                                          const Convolution2DAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_TEXTURE_H_
