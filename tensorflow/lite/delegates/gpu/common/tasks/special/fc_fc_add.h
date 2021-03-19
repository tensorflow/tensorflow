/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_FC_FC_ADD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_FC_FC_ADD_H_

#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

template <DataType T, typename S>
void RearrangeFCWeightsToIOO4I4(const tflite::gpu::Tensor<OHWI, T>& weights,
                                S* dst) {
  const int src_channels = weights.shape.i;
  const int padded_src_channels = AlignByN(src_channels, 4);
  const int dst_channels = weights.shape.o;
  const int padded_dst_channels = AlignByN(dst_channels, 4);

  for (int block_y = 0; 4 * block_y < padded_dst_channels; block_y++) {
    for (int y_in_block = 0; y_in_block < 4; y_in_block++) {
      for (int block_x = 0; 4 * block_x < padded_src_channels; block_x++) {
        for (int x_in_block = 0; x_in_block < 4; x_in_block++) {
          int y = 4 * block_y + y_in_block;
          int x = 4 * block_x + x_in_block;
          int dst_index = block_x * padded_dst_channels * 4 + block_y * 16 +
                          x_in_block * 4 + y_in_block;
          if (x < src_channels && y < dst_channels) {
            dst[dst_index] = weights.data[src_channels * y + x];
          } else {
            dst[dst_index] = 0.0f;
          }
        }
      }
    }
  }
}

template <DataType T, typename S>
void RearrangeFCWeightsToOIO4I4(const tflite::gpu::Tensor<OHWI, T>& weights,
                                S* dst) {
  const int src_channels = weights.shape.i;
  const int src_depth = DivideRoundUp(src_channels, 4);
  const int dst_channels = weights.shape.o;
  const int dst_depth = DivideRoundUp(dst_channels, 4);

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int s = 0; s < src_depth; ++s) {
      for (int i = 0; i < 4; ++i) {
        const int src_ch = s * 4 + i;
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + j;
          if (src_ch < src_channels && dst_ch < dst_channels) {
            dst[counter++] = weights.data[dst_ch * src_channels + src_ch];
          } else {
            dst[counter++] = 0.0f;
          }
        }
      }
    }
  }
}

class FCFCAdd : public GPUOperation {
 public:
  FCFCAdd() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  FCFCAdd(FCFCAdd&& kernel);
  FCFCAdd& operator=(FCFCAdd&& kernel);
  FCFCAdd(const FCFCAdd&) = delete;
  FCFCAdd& operator=(const FCFCAdd&) = delete;

 private:
  FCFCAdd(const OperationDef& definition, const GpuInfo& gpu_info);
  friend FCFCAdd CreateFCFCAdd(const GpuInfo& gpu_info,
                               const OperationDef& definition,
                               const FullyConnectedAttributes& attr0,
                               const FullyConnectedAttributes& attr1);
  friend FCFCAdd CreateFCFCAdd(const GpuInfo& gpu_info,
                               const OperationDef& definition,
                               const FullyConnectedInt8Attributes& attr0,
                               const FullyConnectedInt8Attributes& attr1);

  void UploadQuantizedWeights(
      const tflite::gpu::Tensor<OHWI, DataType::INT8>& weights, float scale,
      float zero_point, int index);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                     const std::string& name, bool weights_are_buffer);

  std::string GetFCFCAddKernelCode(const OperationDef& op_def,
                                   const GpuInfo& gpu_info,
                                   bool weights_are_buffer, bool quantized_0,
                                   bool quantized_1);
};

template <DataType T>
void FCFCAdd::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                            const std::string& name, bool weights_are_buffer) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);

  const int elements_count = src_depth * dst_depth * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 16;
    desc.size = float4_size * elements_count;
    desc.data.resize(desc.size);

    if (f32_weights) {
      float* ptr = reinterpret_cast<float*>(desc.data.data());
      RearrangeFCWeightsToIOO4I4(weights, ptr);
    } else {
      half* ptr = reinterpret_cast<half*>(desc.data.data());
      RearrangeFCWeightsToIOO4I4(weights, ptr);
    }

    args_.AddObject(name, absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    Texture2DDescriptor desc;
    desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    // desc.element_type = DataType::UINT8;
    // desc.normalized = true;
    // desc.normalized_type = f32_weights ? DataType::FLOAT32 :
    // DataType::FLOAT16;
    desc.size = int2(src_depth * 4, dst_depth);
    desc.data.resize(float4_size * elements_count);

    if (f32_weights) {
      float* ptr = reinterpret_cast<float*>(desc.data.data());
      RearrangeFCWeightsToOIO4I4(weights, ptr);
    } else {
      half* ptr = reinterpret_cast<half*>(desc.data.data());
      RearrangeFCWeightsToOIO4I4(weights, ptr);
    }

    args_.AddObject(name,
                    absl::make_unique<Texture2DDescriptor>(std::move(desc)));
  }
}

FCFCAdd CreateFCFCAdd(const GpuInfo& gpu_info, const OperationDef& definition,
                      const FullyConnectedAttributes& attr0,
                      const FullyConnectedAttributes& attr1);

FCFCAdd CreateFCFCAdd(const GpuInfo& gpu_info, const OperationDef& definition,
                      const FullyConnectedInt8Attributes& attr0,
                      const FullyConnectedInt8Attributes& attr1);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_SPECIAL_FC_FC_ADD_H_
