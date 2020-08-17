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

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
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

template <DataType T, typename S>
void RearrangeFCWeightsToIOO4I4(const tflite::gpu::Tensor<OHWI, T>& weights,
                                absl::Span<S> dst) {
  const int src_channels = weights.shape.i;
  const int padded_src_channels = AlignByN(src_channels, 4);
  const int dst_channels = weights.shape.o;
  const int padded_dst_channels = AlignByN(dst_channels, 4);

  // The weights are to be rearranged in such a way that the first 4 elements of
  // each row, starting from row_0, are copied onto the destination buffer. The
  // next set of 4 elements are then copied and so on. As an example, an 8x8
  // matrix would be rearranged as below.
  //
  //  | a0 a1 a2 a3 a4 a5 a6 a7 |              | a0 a1 a2 a3 b0 b1 b2 b3 |
  //  | b0 b1 b2 b3 b4 b5 b6 b7 |              | c0 c1 c2 c3 d0 d1 d2 d3 |
  //  | c0 c1 c2 c3 c4 c5 c6 c7 |              | e0 e1 e2 e3 f0 f1 f2 f3 |
  //  | d0 d1 d2 d3 d4 d5 d6 d7 |  --------->  | g0 g1 g2 g3 h0 h1 h2 h3 |
  //  | e0 e1 e2 e3 e4 e5 e6 e7 |              | a4 a5 a6 a7 b4 b5 b6 b7 |
  //  | f0 f1 f2 f3 f4 f5 f6 f7 |              | c4 c5 c6 c7 d4 d5 d6 d7 |
  //  | g0 g1 g2 g3 g4 g5 g6 g7 |              | e4 e5 e6 e7 f4 f5 f6 f7 |
  //  | h0 h1 h2 h3 h4 h5 h6 h7 |              | g4 g5 g6 g7 h4 h5 h6 h7 |

  for (int y = 0; y < dst_channels; y++) {
    int x = 0;
    for (; x + 4 <= src_channels; x += 4) {
      const int idx_data_0 = src_channels * y + x;
      S filter = S(weights.data[idx_data_0], weights.data[idx_data_0 + 1],
                   weights.data[idx_data_0 + 2], weights.data[idx_data_0 + 3]);
      dst[y + padded_dst_channels * x / 4] = filter;
    }

    // If the width is not a multiple of 4, padding is required and the padded
    // region is filled with zeros.
    if (src_channels != padded_src_channels) {
      const int idx_data_0 = src_channels * y + x;

      S filter = S(x < src_channels ? weights.data[idx_data_0] : 0.0,
                   x + 1 < src_channels ? weights.data[idx_data_0 + 1] : 0.0,
                   x + 2 < src_channels ? weights.data[idx_data_0 + 2] : 0.0,
                   x + 3 < src_channels ? weights.data[idx_data_0 + 3] : 0.0);
      dst[y + padded_dst_channels * x / 4] = filter;
    }
  }

  // Fill the padded columns with zeros.
  for (int y = dst_channels; y < padded_dst_channels; y++) {
    for (int x = 0; x < padded_src_channels; x += 4) {
      dst[y + padded_dst_channels * x / 4] = S(0.0);
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
  friend absl::Status CreateFullyConnected(
      const CreationContext& creation_context, const OperationDef& definition,
      const FullyConnectedAttributes& attr, FullyConnected* result);

  template <DataType T>
  absl::Status UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                             CLContext* context);

  std::string GetFullyConnectedKernelCode(const OperationDef& op_def,
                                          const int3& work_group_size);
};

template <DataType T>
absl::Status FullyConnected::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
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
    float4* ptr = reinterpret_cast<float4*>(desc.data.data());
    RearrangeFCWeightsToIOO4I4(weights, absl::MakeSpan(ptr, elements_count));
  } else {
    half4* ptr = reinterpret_cast<half4*>(desc.data.data());
    RearrangeFCWeightsToIOO4I4(weights, absl::MakeSpan(ptr, elements_count));
  }

  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
  return absl::OkStatus();
}

absl::Status CreateFullyConnected(const CreationContext& creation_context,
                                  const OperationDef& definition,
                                  const FullyConnectedAttributes& attr,
                                  FullyConnected* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_H_
