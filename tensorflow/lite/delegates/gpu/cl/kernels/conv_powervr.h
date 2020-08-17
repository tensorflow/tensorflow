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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_POWERVR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_POWERVR_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_common.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
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

class ConvPowerVR : public GPUOperation {
 public:
  ConvPowerVR() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const DeviceInfo& device_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments() override;
  int3 GetGridSize() const override;

  ConvWeightsDescription GetConvWeightsDescription() const {
    ConvWeightsDescription desc;
    desc.layout = ConvWeightsLayout::kOHWIOGroupI4O4;
    desc.output_group_size = conv_params_.block_size.z;
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
    PRIVATE_MEM_SIMD8_BROADCAST,
    PRIVATE_MEM_SIMD16_BROADCAST,
    PRIVATE_MEM_SIMD32_BROADCAST,
    PRIVATE_MEM_SIMD64_BROADCAST,
    PRIVATE_MEM_SIMD128_BROADCAST,
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
    int3 block_size;
    int3 work_group_launch_order;
    bool fixed_work_group_size;
    bool linear_hw;
    bool different_weights_for_height;
    int src_depth_loop_size;
    WeightsUploadType weights_upload_type;
    bool x_kernel_is_1;
    bool y_kernel_is_1;

    bool IsPrivateMemBroadcast() const {
      return weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD8_BROADCAST ||
             weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD16_BROADCAST ||
             weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD32_BROADCAST ||
             weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD64_BROADCAST ||
             weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD128_BROADCAST;
    }

    int GetSimdSize() const {
      if (weights_upload_type ==
          WeightsUploadType::PRIVATE_MEM_SIMD8_BROADCAST) {
        return 8;
      } else if (weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD16_BROADCAST) {
        return 16;
      } else if (weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD32_BROADCAST) {
        return 32;
      } else if (weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD64_BROADCAST) {
        return 64;
      } else if (weights_upload_type ==
                 WeightsUploadType::PRIVATE_MEM_SIMD128_BROADCAST) {
        return 128;
      }
      return 1;
    }
  };

  ConvPowerVR(const OperationDef& definition,
              const Convolution2DAttributes& attr, const CLDevice& device,
              const BHWC* dst_shape = nullptr);
  ConvPowerVR(const OperationDef& definition,
              const Convolution2DAttributes& attr, const BHWC& weights_shape,
              const CLDevice& device, const BHWC* dst_shape = nullptr);
  ConvPowerVR(const OperationDef& definition,
              const FullyConnectedAttributes& attr, const CLDevice& device,
              const BHWC* dst_shape = nullptr);
  explicit ConvPowerVR(const OperationDef& definition);

  void GenerateCode(const DeviceInfo& device_info);

  template <DataType T>
  absl::Status UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                          const tflite::gpu::Tensor<Linear, T>& biases,
                          CLContext* context);
  template <DataType T>
  absl::Status UploadDataForWinograd4x4To6x6(
      const tflite::gpu::Tensor<OHWI, T>& weights, const CLDevice& device,
      CLContext* context);

  template <DataType T>
  absl::Status UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights,
                             CLContext* context);

  template <DataType T>
  absl::Status UploadBias(const tflite::gpu::Tensor<Linear, T>& bias,
                          CLContext* context);

  friend absl::Status CreateConvPowerVR(const CreationContext& creation_context,
                                        const OperationDef& definition,
                                        const Convolution2DAttributes& attr,
                                        ConvPowerVR* result,
                                        const BHWC* dst_shape);

  friend absl::Status CreateConvPowerVR(const CreationContext& creation_context,
                                        const OperationDef& definition,
                                        const FullyConnectedAttributes& attr,
                                        ConvPowerVR* result,
                                        const BHWC* dst_shape);

  friend absl::Status CreateConvPowerVRDynamicWeights(
      const CreationContext& creation_context, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC& weights_shape,
      ConvPowerVR* result, const BHWC* dst_shape);

  friend absl::Status CreateConvPowerVRWino4x4To6x6(
      const CreationContext& creation_context, const OperationDef& definition,
      const Convolution2DAttributes& attr, ConvPowerVR* result,
      const BHWC* dst_shape);

  ConvParams GuessBestParams(const CLDevice& device,
                             const OperationDef& definition,
                             const Convolution2DAttributes& attr,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const CLDevice& device,
                             const OperationDef& definition,
                             const Convolution2DAttributes& attr,
                             const BHWC& weights_shape,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const CLDevice& device,
                             const OperationDef& definition,
                             const FullyConnectedAttributes& attr,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParamsWinograd(const CLDevice& device,
                                     const OperationDef& definition,
                                     const Convolution2DAttributes& attr,
                                     const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const CLDevice& device,
                             const OperationDef& definition, int src_depth,
                             int dst_depth, bool x_kernel_is_1,
                             bool y_kernel_is_1,
                             bool different_weights_for_height,
                             const BHWC* dst_shape = nullptr);

  std::string GenerateConv(const DeviceInfo& device_info,
                           const OperationDef& op_def, bool stride_correction,
                           const ConvParams& conv_params);

  int4 stride_padding_;
  int4 kernel_dilation_;
  ConvParams conv_params_;
};

template <DataType T>
absl::Status ConvPowerVR::UploadData(
    const tflite::gpu::Tensor<OHWI, T>& weights,
    const tflite::gpu::Tensor<Linear, T>& biases, CLContext* context) {
  RETURN_IF_ERROR(UploadWeights(weights, context));
  RETURN_IF_ERROR(UploadBias(biases, context));
  return absl::OkStatus();
}

template <DataType T>
absl::Status ConvPowerVR::UploadDataForWinograd4x4To6x6(
    const tflite::gpu::Tensor<OHWI, T>& weights, const CLDevice& device,
    CLContext* context) {
  tflite::gpu::Tensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  RETURN_IF_ERROR(UploadWeights(wino_weights, context));
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(weights.shape.o);
  biases.data.resize(weights.shape.o, 0.0f);
  RETURN_IF_ERROR(UploadBias(biases, context));
  return absl::OkStatus();
}

template <DataType T>
absl::Status ConvPowerVR::UploadBias(const tflite::gpu::Tensor<Linear, T>& bias,
                                     CLContext* context) {
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
  int aligned_channels = AlignByN(bias.shape.v, 4 * conv_params_.block_size.z);
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
  return absl::OkStatus();
}

template <DataType T>
absl::Status ConvPowerVR::UploadWeights(
    const tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);

  const bool f32_weights = conv_params_.weights_data_type == DataType::FLOAT32;
  const int float4_size = f32_weights ? sizeof(float4) : sizeof(half4);

  const int dst_depth_aligned = AlignByN(dst_depth, conv_params_.block_size.z);
  const int elements_count =
      weights.shape.h * weights.shape.w * src_depth * dst_depth_aligned * 4;

  BufferDescriptor desc;
  desc.element_type = conv_params_.weights_data_type;
  desc.element_size = 4;
  desc.memory_type = conv_params_.weights_upload_type ==
                             ConvPowerVR::WeightsUploadType::CONSTANT_MEM
                         ? MemoryType::CONSTANT
                         : MemoryType::GLOBAL;
  desc.size = float4_size * elements_count;
  desc.data.resize(desc.size);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(desc.data.data());
    RearrangeWeightsToOHWIOGroupI4O4(weights, conv_params_.block_size.z,
                                     absl::MakeSpan(ptr, elements_count));
  } else {
    half4* ptr = reinterpret_cast<half4*>(desc.data.data());
    RearrangeWeightsToOHWIOGroupI4O4(weights, conv_params_.block_size.z,
                                     absl::MakeSpan(ptr, elements_count));
  }
  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
  return absl::OkStatus();
}

absl::Status CreateConvPowerVR(const CreationContext& creation_context,
                               const OperationDef& definition,
                               const Convolution2DAttributes& attr,
                               ConvPowerVR* result,
                               const BHWC* dst_shape = nullptr);

absl::Status CreateConvPowerVR(const CreationContext& creation_context,
                               const OperationDef& definition,
                               const FullyConnectedAttributes& attr,
                               ConvPowerVR* result,
                               const BHWC* dst_shape = nullptr);

absl::Status CreateConvPowerVRDynamicWeights(
    const CreationContext& creation_context, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    ConvPowerVR* result, const BHWC* dst_shape = nullptr);

absl::Status CreateConvPowerVRWino4x4To6x6(
    const CreationContext& creation_context, const OperationDef& definition,
    const Convolution2DAttributes& attr, ConvPowerVR* result,
    const BHWC* dst_shape = nullptr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_POWERVR_H_
