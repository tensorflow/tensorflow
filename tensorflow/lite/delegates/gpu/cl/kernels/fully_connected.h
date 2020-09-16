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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/device_info.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/tuning_parameters.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

template <DataType T, typename S>
void RearrangeFCWeightsToIOO4I4(const tflite::gpu::Tensor<OHWI, T>& weights,
                                S* dst) {
  const int src_channels = weights.shape.i;
  const int padded_src_channels = AlignByN(src_channels, 4);
  const int dst_channels = weights.shape.o;
  const int padded_dst_channels = AlignByN(dst_channels, 4);

  // Change the travelsal order of the weight matrix in such a way that the
  // first 4 elements of all rows are scanned first, followed by elements 5 to 8
  // of all rows, then elements 9 to 12 of all rows, and so on. As an example,
  // an 8x8 matrix would be traversed as below.
  //
  //  |  0  1  2  3 32 33 34 35 |
  //  |  4  5  6  7 36 37 38 39 |
  //  |  8  9 10 11 40 41 42 43 |
  //  | 12 13 14 15 44 45 46 47 |
  //  | 16 17 18 19 48 49 50 51 |
  //  | 20 21 22 23 52 53 54 55 |
  //  | 24 25 26 27 56 57 58 59 |
  //  | 28 29 30 31 60 61 62 63 |
  //
  // If (any) dimension of the weight matrix size is not divisible by 4, then
  // the output is padded with zeros.
  //
  // The benefit of doing this is that reading contigous 16 elements gives a 4x4
  // block of the matrix, where the first 4 elements is the first row of the
  // block, second 4 elements is the second row of the block, etc.

  for (int y = 0; y < padded_dst_channels; y++) {
    for (int block_x = 0; 4 * block_x < padded_src_channels; block_x++) {
      for (int x_in_block = 0; x_in_block < 4; x_in_block++) {
        int x = 4 * block_x + x_in_block;
        int dst_index = padded_dst_channels * 4 * block_x + 4 * y + x_in_block;
        if (x < src_channels && y < dst_channels) {
          dst[dst_index] = weights.data[src_channels * y + x];
        } else {
          dst[dst_index] = 0.0f;
        }
      }
    }
  }
}

class FullyConnected : public GPUOperation {
 public:
  FullyConnected() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const DeviceInfo& device_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  FullyConnected(FullyConnected&& kernel);
  FullyConnected& operator=(FullyConnected&& kernel);
  FullyConnected(const FullyConnected&) = delete;
  FullyConnected& operator=(const FullyConnected&) = delete;

 private:
  FullyConnected(const OperationDef& definition, const DeviceInfo& device_info);
  friend FullyConnected CreateFullyConnected(
      const DeviceInfo& device_info, const OperationDef& definition,
      const FullyConnectedAttributes& attr);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);

  std::string GetFullyConnectedKernelCode(const OperationDef& op_def,
                                          const int3& work_group_size);
};

template <DataType T>
void FullyConnected::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);

  const int elements_count = src_depth * dst_depth * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;

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

  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

FullyConnected CreateFullyConnected(const DeviceInfo& device_info,
                                    const OperationDef& definition,
                                    const FullyConnectedAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_H_
