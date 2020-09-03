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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_3D_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_3D_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
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

class Conv3D : public GPUOperation {
 public:
  Conv3D() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const DeviceInfo& device_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments() override;
  int3 GetGridSize() const override;

  // Move only
  Conv3D(Conv3D&& operation);
  Conv3D& operator=(Conv3D&& operation);
  Conv3D(const Conv3D&) = delete;
  Conv3D& operator=(const Conv3D&) = delete;

 private:
  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC_SUBGROUP,  // we use it for PowerVR with workgroup size = 32
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    TEXTURES_MEM,
  };

  struct ConvParams {
    int4 block_size;  // WHDS
    int3 work_group_launch_order;
    int src_depth_loop_size;
    WeightsUploadType weights_upload_type;
    bool AreWeightsBuffer() const {
      return weights_upload_type != WeightsUploadType::TEXTURES_MEM;
    }
    bool x_kernel_is_1;
    bool y_kernel_is_1;
    bool z_kernel_is_1;
  };

  Conv3D(const OperationDef& definition, const Convolution3DAttributes& attr,
         const DeviceInfo& device_info);

  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<OHWDI, T>& weights,
                  const tflite::gpu::Tensor<Linear, T>& biases);
  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights);

  friend Conv3D CreateConv3D(const DeviceInfo& device_info,
                             const OperationDef& definition,
                             const Convolution3DAttributes& attr);

  friend std::string GenerateConv3D(const OperationDef& op_def,
                                    bool stride_correction,
                                    const ConvParams& conv_params,
                                    Arguments* args);

  ConvParams GuessBestParams(const DeviceInfo& device_info,
                             const OperationDef& definition,
                             const Convolution3DAttributes& attr);

  ConvParams GuessBestParams(const DeviceInfo& device_info,
                             const OperationDef& definition, int src_slices,
                             int dst_slices, bool x_kernel_is_1,
                             bool y_kernel_is_1, bool z_kernel_is_1);

  std::string GenerateConv3D(const OperationDef& op_def, bool stride_correction,
                             const Conv3D::ConvParams& conv_params);

  int3 stride_;
  int3 padding_;
  int3 kernel_size_;
  int3 dilation_;
  ConvParams conv_params_;
};

template <DataType T>
void Conv3D::UploadData(const tflite::gpu::Tensor<OHWDI, T>& weights,
                        const tflite::gpu::Tensor<Linear, T>& biases) {
  UploadWeights(weights);
  TensorLinearDescriptor desc;
  desc.storage_type = conv_params_.AreWeightsBuffer()
                          ? LinearStorageType::BUFFER
                          : LinearStorageType::TEXTURE_2D;
  desc.element_type = definition_.GetDataType();
  desc.UploadLinearData(biases);
  args_.AddObject("biases",
                  absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
}

template <DataType T>
void Conv3D::UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights) {
  const int block_size = conv_params_.block_size.w;
  const int dst_slices =
      AlignByN(DivideRoundUp(weights.shape.o, 4), block_size);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = kernel_size_.x;
  const int kernel_y = kernel_size_.y;
  const int kernel_z = kernel_size_.z;

  const int elements_count =
      kernel_x * kernel_y * kernel_z * src_slices * dst_slices * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;

  std::vector<uint8_t> data(float4_size * elements_count);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    if (conv_params_.AreWeightsBuffer()) {
      RearrangeWeightsToODHWIOGroupI4O4(weights, conv_params_.block_size.w,
                                        absl::MakeSpan(ptr, elements_count));
    } else {
      RearrangeWeightsToI4DHWIOOGroupO4(weights, conv_params_.block_size.w,
                                        absl::MakeSpan(ptr, elements_count));
    }
  } else {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    if (conv_params_.AreWeightsBuffer()) {
      RearrangeWeightsToODHWIOGroupI4O4(weights, conv_params_.block_size.w,
                                        absl::MakeSpan(ptr, elements_count));
    } else {
      RearrangeWeightsToI4DHWIOOGroupO4(weights, conv_params_.block_size.w,
                                        absl::MakeSpan(ptr, elements_count));
    }
  }

  if (conv_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 4;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    const int texture_width = dst_slices;
    const int texture_height = src_slices * kernel_x * kernel_y * kernel_z;
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

Conv3D CreateConv3D(const DeviceInfo& device_info,
                    const OperationDef& definition,
                    const Convolution3DAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONV_3D_H_
