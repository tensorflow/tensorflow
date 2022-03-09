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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_POWERVR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_POWERVR_H_

#include <cstring>
#include <string>
#include <utility>
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
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {

class ConvPowerVR : public GPUOperation {
 public:
  ConvPowerVR() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = conv_params_.weights_data_type;
    desc.layout = conv_params_.weights_layout;
    desc.output_group_size = conv_params_.block_size.w;
    return desc;
  }

  // Move only
  ConvPowerVR(ConvPowerVR&& operation);
  ConvPowerVR& operator=(ConvPowerVR&& operation);
  ConvPowerVR(const ConvPowerVR&) = delete;
  ConvPowerVR& operator=(const ConvPowerVR&) = delete;

 private:
  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC_SUBGROUP,  // we use it for PowerVR with workgroup size = 32
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
    PRIVATE_MEM_SIMD_BROADCAST,
    TEXTURES_MEM_X4,  // 4 textures for weights
  };

  struct ConvParams {
    // Usually we use this combinations for CalculationPrecision:
    // F32: all F32
    // F16: all F16
    // F32_F16: all besides accumulator is F16, including weights
    // But for PowerVR we can achieve better performance in F32_F16 with F32
    // weights, so for PowerVR in this kernel we have F32 weights for
    // F32_F16 precision mode
    DataType weights_data_type;  // used for weights and biases
    int4 block_size;             // WHDS
    bool fixed_work_group_size;
    bool linear_spatial;  // spatial dimensions are Width/Height/Depth
    bool linear_all;  // linear_spatial & linear_all can not be used together,
                      // linear_all can not be used with WeightsUploadTypes
                      // that use workgroups(subgroups) for
                      // uploading(LOCAL_MEM_BY_THREADS for example).
    bool different_weights_for_height;
    bool groups_support = false;  // convolution groups
    int src_depth_loop_size;
    WeightsUploadType weights_upload_type;
    bool x_kernel_is_1 = false;
    bool y_kernel_is_1 = false;
    bool z_kernel_is_1 = false;
    WeightsLayout weights_layout;

    // used only with PRIVATE_MEM_SIMD_BROADCAST
    int simd_size = 1;

    bool AreWeightsBuffer() const {
      return weights_upload_type != WeightsUploadType::TEXTURES_MEM_X4;
    }

    bool IsPrivateMemBroadcast() const {
      return weights_upload_type ==
             WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
    }
  };

  ConvPowerVR(const OperationDef& definition,
              const Convolution2DAttributes& attr, const GpuInfo& gpu_info,
              const BHWC* dst_shape = nullptr);
  ConvPowerVR(const OperationDef& definition,
              const Convolution2DAttributes& attr, const BHWC& weights_shape,
              const GpuInfo& gpu_info, const BHWC* dst_shape = nullptr);
  ConvPowerVR(const OperationDef& definition,
              const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
              const BHWC* dst_shape = nullptr);
  explicit ConvPowerVR(const OperationDef& definition);
  ConvPowerVR(const OperationDef& definition,
              const Convolution3DAttributes& attr, const GpuInfo& gpu_info,
              const BHWDC* dst_shape = nullptr);

  void GenerateCode(const GpuInfo& gpu_info);

  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                  const tflite::gpu::Tensor<Linear, T>& biases);
  template <DataType T>
  void UploadDataForWinograd4x4To6x6(
      const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights);

  template <DataType T>
  void UploadBias(const tflite::gpu::Tensor<Linear, T>& bias);

  friend ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const Convolution2DAttributes& attr,
                                       const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const FullyConnectedAttributes& attr,
                                       const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVRDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC& weights_shape,
      const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVRWino4x4To6x6(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVR3D(const GpuInfo& gpu_info,
                                         const OperationDef& definition,
                                         const Convolution3DAttributes& attr,
                                         const BHWDC* dst_shape);

  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const Convolution2DAttributes& attr,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const Convolution2DAttributes& attr,
                             const BHWC& weights_shape,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const FullyConnectedAttributes& attr,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParamsWinograd(const GpuInfo& gpu_info,
                                     const OperationDef& definition,
                                     const Convolution2DAttributes& attr,
                                     const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const Convolution3DAttributes& attr,
                             const BHWDC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition, int src_depth,
                             int dst_depth, bool x_kernel_is_1,
                             bool y_kernel_is_1,
                             bool different_weights_for_height,
                             const BHWC* dst_shape = nullptr);

  std::string GenerateConv(const GpuInfo& gpu_info, const OperationDef& op_def,
                           bool stride_correction,
                           const ConvParams& conv_params);

  int4 stride_;
  int4 padding_;
  int4 kernel_size_;
  int4 dilation_;
  ConvParams conv_params_;
};

template <DataType T>
void ConvPowerVR::UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                             const tflite::gpu::Tensor<Linear, T>& biases) {
  UploadWeights(weights);
  UploadBias(biases);
}

template <DataType T>
void ConvPowerVR::UploadDataForWinograd4x4To6x6(
    const tflite::gpu::Tensor<OHWI, T>& weights) {
  tflite::gpu::Tensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  UploadWeights(wino_weights);
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(weights.shape.o);
  biases.data.resize(weights.shape.o, 0.0f);
  UploadBias(biases);
}

template <DataType T>
void ConvPowerVR::UploadBias(const tflite::gpu::Tensor<Linear, T>& bias) {
  BufferDescriptor desc;
  desc.element_type = conv_params_.weights_data_type;
  desc.element_size = 4;
  desc.memory_type = conv_params_.weights_upload_type ==
                             ConvPowerVR::WeightsUploadType::CONSTANT_MEM
                         ? MemoryType::CONSTANT
                         : MemoryType::GLOBAL;
  const int float_size = conv_params_.weights_data_type == DataType::FLOAT32
                             ? sizeof(float)
                             : sizeof(half);
  int aligned_channels = AlignByN(bias.shape.v, 4 * conv_params_.block_size.w);
  desc.size = float_size * aligned_channels;
  desc.data.resize(desc.size);
  if (conv_params_.weights_data_type == DataType::FLOAT32) {
    float* gpu_data = reinterpret_cast<float*>(desc.data.data());
    for (int i = 0; i < aligned_channels; ++i) {
      gpu_data[i] = i < bias.shape.v ? bias.data[i] : 0.0f;
    }
  } else {
    half* gpu_data = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < aligned_channels; ++i) {
      gpu_data[i] = i < bias.shape.v ? bias.data[i] : 0.0f;
    }
  }
  args_.AddObject("biases",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType T>
void ConvPowerVR::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (conv_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 4;
    desc.memory_type = conv_params_.weights_upload_type ==
                               ConvPowerVR::WeightsUploadType::CONSTANT_MEM
                           ? MemoryType::CONSTANT
                           : MemoryType::GLOBAL;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = weights_desc.type;
      desc.size = int2(tex_size.x, tex_size.y);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), weights_data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

template <DataType T>
void ConvPowerVR::UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (conv_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 4;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = weights_desc.type;
      desc.size = int2(tex_size.x, tex_size.y);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), weights_data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr,
                              const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const FullyConnectedAttributes& attr,
                              const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVRDynamicWeights(const GpuInfo& gpu_info,
                                            const OperationDef& definition,
                                            const Convolution2DAttributes& attr,
                                            const BHWC& weights_shape,
                                            const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVRWino4x4To6x6(const GpuInfo& gpu_info,
                                          const OperationDef& definition,
                                          const Convolution2DAttributes& attr,
                                          const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVR3D(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                const Convolution3DAttributes& attr,
                                const BHWDC* dst_shape = nullptr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_POWERVR_H_
