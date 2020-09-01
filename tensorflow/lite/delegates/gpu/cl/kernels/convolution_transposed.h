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

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
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
  explicit ConvolutionTransposed(const OperationDef& definition,
                                 const ConvolutionTransposedAttributes& attr,
                                 const DeviceInfo& device_info);
  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                     bool weights_are_buffer);

  template <DataType S, typename T>
  void RearrangeWeightsData(const tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst, bool weights_are_buffer);

  std::string GenerateConvolutionTransposedCode(const OperationDef& op_def,
                                                const DeviceInfo& device_info,
                                                bool weights_are_buffer,
                                                const int3& block_size);
  int2 stride_;
  int3 block_size_ = int3(1, 1, 1);
};

template <DataType T>
void ConvolutionTransposed::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, bool weights_are_buffer) {
  const int dst_depth =
      AlignByN(DivideRoundUp(weights.shape.o, 4), block_size_.z);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int elements_count = kernel_x * kernel_y * src_depth * dst_depth * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;
  std::vector<uint8_t> data(float4_size * elements_count);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(ptr, elements_count),
                         weights_are_buffer);
  } else {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    RearrangeWeightsData(weights, absl::MakeSpan(ptr, elements_count),
                         weights_are_buffer);
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
    int sub_size = float4_size * elements_count / 4;
    Texture2DDescriptor desc0;
    desc0.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc0.size = int2(dst_depth, src_depth * kernel_x * kernel_y);
    desc0.data.resize(sub_size);
    memcpy(desc0.data.data(), data.data(), sub_size);
    args_.AddObject("weights0",
                    absl::make_unique<Texture2DDescriptor>(std::move(desc0)));

    Texture2DDescriptor desc1;
    desc1.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc1.size = int2(dst_depth, src_depth * kernel_x * kernel_y);
    desc1.data.resize(sub_size);
    memcpy(desc1.data.data(), data.data() + sub_size, sub_size);
    args_.AddObject("weights1",
                    absl::make_unique<Texture2DDescriptor>(std::move(desc1)));

    Texture2DDescriptor desc2;
    desc2.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc2.size = int2(dst_depth, src_depth * kernel_x * kernel_y);
    desc2.data.resize(sub_size);
    memcpy(desc2.data.data(), data.data() + sub_size * 2, sub_size);
    args_.AddObject("weights2",
                    absl::make_unique<Texture2DDescriptor>(std::move(desc2)));

    Texture2DDescriptor desc3;
    desc3.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc3.size = int2(dst_depth, src_depth * kernel_x * kernel_y);
    desc3.data.resize(sub_size);
    memcpy(desc3.data.data(), data.data() + sub_size * 3, sub_size);
    args_.AddObject("weights3",
                    absl::make_unique<Texture2DDescriptor>(std::move(desc3)));
  }
}

template <DataType S, typename T>
void ConvolutionTransposed::RearrangeWeightsData(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst,
    bool weights_are_buffer) {
  const int dst_depth =
      AlignByN(DivideRoundUp(weights.shape.o, 4), block_size_.z);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;
  int texture_width = dst_depth;
  int texture_height = src_depth * kernel_x * kernel_y;

  int counter = 0;
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
            if (weights_are_buffer) {
              dst[counter++] = filters[0];
              dst[counter++] = filters[1];
              dst[counter++] = filters[2];
              dst[counter++] = filters[3];
            } else {
              int x_coord = d * block_size_.z + sub_d;
              int y_coord = (y * kernel_x + x) * src_depth + s;
              int offset = y_coord * dst_depth + x_coord;
              dst[offset + texture_width * texture_height * 0] = filters[0];
              dst[offset + texture_width * texture_height * 1] = filters[1];
              dst[offset + texture_width * texture_height * 2] = filters[2];
              dst[offset + texture_width * texture_height * 3] = filters[3];
            }
          }
        }
      }
    }
  }
}

ConvolutionTransposed CreateConvolutionTransposed(
    const DeviceInfo& device_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_H_
