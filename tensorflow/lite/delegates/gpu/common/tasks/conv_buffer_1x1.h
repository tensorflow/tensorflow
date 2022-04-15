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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_BUFFER_1X1_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_BUFFER_1X1_H_

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {

class ConvBuffer1x1 : public GPUOperation {
 public:
  ConvBuffer1x1() = default;

  // Move only
  ConvBuffer1x1(ConvBuffer1x1&& operation);
  ConvBuffer1x1& operator=(ConvBuffer1x1&& operation);
  ConvBuffer1x1(const ConvBuffer1x1&) = delete;
  ConvBuffer1x1& operator=(const ConvBuffer1x1&) = delete;

  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = DeduceDataTypeFromPrecision(definition_.precision);
    desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    desc.output_group_size = conv_params_.block_size.z;
    return desc;
  }

  struct ConvParams {
    int3 block_size = int3(1, 1, 1);
    int element_size = 4;  // can be 4, 8 or 16

    // By default in 2d convolution we have the same weights for WH dims, but in
    // some cases we need separate weights for H dimension and convolution
    // kernel requires very small modifications to support it.
    bool different_weights_for_height = false;
  };

 private:
  ConvBuffer1x1(const OperationDef& definition, const ConvParams& conv_params,
                const GpuInfo& gpu_info);
  friend ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                           const OperationDef& definition,
                                           const Convolution2DAttributes& attr,
                                           const BHWC* shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                           const OperationDef& definition,
                                           const FullyConnectedAttributes& attr,
                                           const BHWC* shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1Wino4x4To6x6(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC* shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1DynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC& weights_shape,
      const BHWC* dst_shape);

  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                  const tflite::gpu::Tensor<Linear, T>& biases);
  template <DataType T>
  void UploadDataForWinograd4x4To6x6(
      const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadBiases(const tflite::gpu::Tensor<Linear, T>& biases);

  std::string GenerateConvBuffer1x1(
      const OperationDef& op_def, const ConvBuffer1x1::ConvParams& conv_params,
      const GpuInfo& gpu_info, Arguments* args);

  ConvParams conv_params_;
};

template <DataType T>
void ConvBuffer1x1::UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                               const tflite::gpu::Tensor<Linear, T>& biases) {
  UploadWeights(weights);
  UploadBiases(biases);
}

template <DataType T>
void ConvBuffer1x1::UploadDataForWinograd4x4To6x6(
    const tflite::gpu::Tensor<OHWI, T>& weights) {
  tflite::gpu::Tensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  UploadWeights(wino_weights);
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape = Linear(weights.shape.o);
  bias.data.resize(weights.shape.o, 0.0f);
  UploadBiases(bias);
}

template <DataType T>
void ConvBuffer1x1::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor desc;
  desc.element_type = weights_desc.type;
  desc.element_size = 16;
  desc.memory_type = MemoryType::GLOBAL;
  desc.size = flt_count * SizeOf(desc.element_type);
  desc.data.resize(desc.size);

  RearrangeWeights(weights, weights_desc, absl::MakeSpan(desc.data));

  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType T>
void ConvBuffer1x1::UploadBiases(const tflite::gpu::Tensor<Linear, T>& biases) {
  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::BUFFER;
  desc.element_type = definition_.GetDataType();
  int depth = AlignByN(biases.shape.v, 4 * conv_params_.block_size.z) / 4;
  desc.UploadLinearData(biases, depth);
  args_.AddObject("biases",
                  absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
}

bool IsConvBuffer1x1Supported(const OperationDef& definition,
                              const Convolution2DAttributes& attr);

bool IsConvBuffer1x1Supported(const OperationDef& definition,
                              const BHWC& weights_shape,
                              const Convolution2DAttributes& attr);

ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const Convolution2DAttributes& attr,
                                  const BHWC* shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const FullyConnectedAttributes& attr,
                                  const BHWC* shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC* dst_shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1Wino4x4To6x6(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC* shape = nullptr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_BUFFER_1X1_H_
