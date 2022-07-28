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
#include "tensorflow/lite/delegates/gpu/metal/common.h"
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
                     const TensorDescriptor& descriptor);

  // Move only
  MetalSpatialTensor(MetalSpatialTensor&& tensor);
  MetalSpatialTensor& operator=(MetalSpatialTensor&& tensor);
  MetalSpatialTensor(const MetalSpatialTensor&) = delete;
  MetalSpatialTensor& operator=(const MetalSpatialTensor&) = delete;

  ~MetalSpatialTensor() override { Release(); }

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  int Width() const override { return descriptor_.GetBHWDCShape().w; }
  int Height() const override { return descriptor_.GetBHWDCShape().h; }
  int Depth() const override { return descriptor_.GetBHWDCShape().d; }
  int Channels() const override { return descriptor_.GetBHWDCShape().c; }
  int Slices() const override {
    return DivideRoundUp(descriptor_.GetBHWDCShape().c, 4);
  }
  int Batch() const override { return descriptor_.GetBHWDCShape().b; }

  TensorDescriptor GetDescriptor() const override { return descriptor_; }
  DataType GetDataType() const { return descriptor_.GetDataType(); }
  TensorStorageType GetStorageType() const {
    return descriptor_.GetStorageType();
  }

  uint64_t GetMemorySizeInBytes() const;

  absl::Status CreateFromDescriptor(const TensorDescriptor& desc,
                                    id<MTLDevice> device);
  absl::Status UploadDescriptorData(const TensorDescriptor& desc,
                                    id<MTLDevice> device);
  absl::Status ToDescriptor(TensorDescriptor* desc, id<MTLDevice> device) const;

  absl::Status SetBufferHandle(id<MTLBuffer> buffer);
  id<MTLBuffer> GetBufferHandle() const;

 private:
  friend absl::Status CreateTensorSharedBuffer(
      id<MTLBuffer> buffer, const TensorDescriptor& descriptor,
      MetalSpatialTensor* result, uint64_t buffer_offset);

  friend absl::Status CreateTensorSharedImage2DBuffer(
      id<MTLBuffer> buffer, const TensorDescriptor& descriptor,
      int row_bytes_alignment, MetalSpatialTensor* result,
      uint64_t buffer_offset);

  absl::Status WriteData(id<MTLDevice> device, const void* ptr);
  absl::Status ReadData(id<MTLDevice> device, void* ptr) const;

  int3 GetFullTensorRegion() const;
  void Release();

  id<MTLBuffer> memory_;
  id<MTLTexture> texture_mem_;
  bool memory_owner_;
  bool texture_mem_owner_;
  TensorDescriptor descriptor_;
  // for use with TEXTURE_2D and when texture created from buffer.
  int aligned_texture_width_;
  // used when created from shared buffer
  uint64_t buffer_offset_ = 0;
};

absl::Status CreateTensor(id<MTLDevice> device,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateTensorSharedBuffer(id<MTLBuffer> buffer,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result,
                                      uint64_t buffer_offset = 0);

absl::Status CreateTensorSharedImage2DBuffer(id<MTLBuffer> buffer,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment,
                                             MetalSpatialTensor* result,
                                             uint64_t buffer_offset = 0);

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_
