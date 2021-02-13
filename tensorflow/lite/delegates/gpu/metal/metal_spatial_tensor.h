/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the Licensgoe is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_

#import <Metal/Metal.h>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_tensor.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/gpu_object.h"

namespace tflite {
namespace gpu {
namespace metal {

class MetalSpatialTensor : public GPUObject, public GpuSpatialTensor {
 public:
  MetalSpatialTensor()
      : memory_(nullptr),
        texture_mem_(nullptr),
        memory_owner_(true),
        texture_mem_owner_(true) {}
  MetalSpatialTensor(id<MTLBuffer> buffer, id<MTLTexture> texture,
                     bool memory_owner, bool texture_mem_owner,
                     const BHWC& shape, const TensorDescriptor& descriptor);
  MetalSpatialTensor(id<MTLBuffer> buffer, id<MTLTexture> texture,
                     bool memory_owner, bool texture_mem_owner,
                     const BHWDC& shape, const TensorDescriptor& descriptor);

  // Move only
  MetalSpatialTensor(MetalSpatialTensor&& tensor);
  MetalSpatialTensor& operator=(MetalSpatialTensor&& tensor);
  MetalSpatialTensor(const MetalSpatialTensor&) = delete;
  MetalSpatialTensor& operator=(const MetalSpatialTensor&) = delete;

  ~MetalSpatialTensor() override { Release(); }

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  int Width() const override { return shape_.w; }
  int Height() const override { return shape_.h; }
  int Depth() const override { return shape_.d; }
  int Channels() const override { return shape_.c; }
  int Slices() const override { return DivideRoundUp(shape_.c, 4); }
  int Batch() const override { return shape_.b; }

  TensorDescriptor GetDescriptor() const { return descriptor_; }
  DataType GetDataType() const { return descriptor_.data_type; }
  TensorStorageType GetStorageType() const { return descriptor_.storage_type; }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const;

  absl::Status WriteData(id<MTLDevice> device, const TensorFloat32& src);
  absl::Status WriteData(
      id<MTLDevice> device,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);
  absl::Status WriteData(
      id<MTLDevice> device,
      const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  absl::Status WriteData(id<MTLDevice> device, const Tensor5DFloat32& src);
  absl::Status ReadData(id<MTLDevice> device, TensorFloat32* dst) const;
  absl::Status ReadData(id<MTLDevice> device, Tensor5DFloat32* dst) const;

  absl::Status CreateFromDescriptor(const TensorDescriptor& desc,
                                    id<MTLDevice> device);

  absl::Status SetBufferHandle(id<MTLBuffer> buffer);
  id<MTLBuffer> GetBufferHandle() const;

 private:
  absl::Status IsValid(const BHWC& shape) const;
  absl::Status IsValid(const BHWDC& shape) const;

  absl::Status WriteDataBHWDC(id<MTLDevice> device, const float* in);
  absl::Status ReadDataBHWDC(id<MTLDevice> device, float* out) const;

  int GetAlignedChannels() const;
  int3 GetFullTensorRegion() const;
  void Release();

  id<MTLBuffer> memory_;
  id<MTLTexture> texture_mem_;
  bool memory_owner_;
  bool texture_mem_owner_;
  BHWDC shape_;
  TensorDescriptor descriptor_;
};

absl::Status CreateTensor(id<MTLDevice> device, const BHWC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateTensor(id<MTLDevice> device, const BHWDC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result);

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_
