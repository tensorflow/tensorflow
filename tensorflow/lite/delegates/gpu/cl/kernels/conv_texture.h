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

  template <DataType S, typename T>
  void RearrangeWeightsData(const tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst_0, absl::Span<T> dst_1,
                            absl::Span<T> dst_2, absl::Span<T> dst_3);

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

  int texture_width = dst_depth;
  int texture_height = src_depth * kernel_x * kernel_y;

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  DataType data_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;

  const int elements_count = texture_width * texture_height;
  const int float4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  Texture2DDescriptor desc0;
  desc0.element_type = data_type;
  desc0.size = int2(texture_width, texture_height);
  desc0.data.resize(elements_count * float4_size);

  Texture2DDescriptor desc1;
  desc1.element_type = data_type;
  desc1.size = int2(texture_width, texture_height);
  desc1.data.resize(elements_count * float4_size);

  Texture2DDescriptor desc2;
  desc2.element_type = data_type;
  desc2.size = int2(texture_width, texture_height);
  desc2.data.resize(elements_count * float4_size);

  Texture2DDescriptor desc3;
  desc3.element_type = data_type;
  desc3.size = int2(texture_width, texture_height);
  desc3.data.resize(elements_count * float4_size);

  if (f32_weights) {
    float4* ptr0 = reinterpret_cast<float4*>(desc0.data.data());
    float4* ptr1 = reinterpret_cast<float4*>(desc1.data.data());
    float4* ptr2 = reinterpret_cast<float4*>(desc2.data.data());
    float4* ptr3 = reinterpret_cast<float4*>(desc3.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(ptr0, elements_count),
                         absl::MakeSpan(ptr1, elements_count),
                         absl::MakeSpan(ptr2, elements_count),
                         absl::MakeSpan(ptr3, elements_count));
  } else {
    half4* ptr0 = reinterpret_cast<half4*>(desc0.data.data());
    half4* ptr1 = reinterpret_cast<half4*>(desc1.data.data());
    half4* ptr2 = reinterpret_cast<half4*>(desc2.data.data());
    half4* ptr3 = reinterpret_cast<half4*>(desc3.data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(ptr0, elements_count),
                         absl::MakeSpan(ptr1, elements_count),
                         absl::MakeSpan(ptr2, elements_count),
                         absl::MakeSpan(ptr3, elements_count));
  }

  args_.AddObject("weights0",
                  absl::make_unique<Texture2DDescriptor>(std::move(desc0)));
  args_.AddObject("weights1",
                  absl::make_unique<Texture2DDescriptor>(std::move(desc1)));
  args_.AddObject("weights2",
                  absl::make_unique<Texture2DDescriptor>(std::move(desc2)));
  args_.AddObject("weights3",
                  absl::make_unique<Texture2DDescriptor>(std::move(desc3)));
}

template <DataType S, typename T>
void ConvTexture::RearrangeWeightsData(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst_0,
    absl::Span<T> dst_1, absl::Span<T> dst_2, absl::Span<T> dst_3) {
  int dst_depth = DivideRoundUp(weights.shape.o, 4);
  dst_depth = AlignByN(dst_depth, block_size_.z);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int texture_width = dst_depth;

  for (int d = 0; d < dst_depth / block_size_.z; ++d) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int s = 0; s < src_depth; ++s) {
          for (int sub_d = 0; sub_d < block_size_.z; ++sub_d) {
            T filters[4];
            for (int i = 0; i < 4; ++i) {
              for (int j = 0; j < 4; ++j) {
                const int s_ch = s * 4 + j;
                const int d_ch = (d * block_size_.z + sub_d) * 4 + i;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                  const int f_index =
                      weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  filters[j][i] = weights.data[f_index];
                } else {
                  filters[j][i] = 0.0f;
                }
              }
            }
            int x_coord = d * block_size_.z + sub_d;
            int y_coord = (y * kernel_x + x) * src_depth + s;
            int offset = y_coord * texture_width + x_coord;
            dst_0[offset] = filters[0];
            dst_1[offset] = filters[1];
            dst_2[offset] = filters[2];
            dst_3[offset] = filters[3];
          }
        }
      }
    }
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
