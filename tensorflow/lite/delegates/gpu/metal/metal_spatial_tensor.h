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

  TensorDescriptor GetDescriptor() const override { return descriptor_; }
  DataType GetDataType() const { return descriptor_.GetDataType(); }
  TensorStorageType GetStorageType() const {
    return descriptor_.GetStorageType();
  }

  uint64_t GetMemorySizeInBytes() const;

  absl::Status WriteData(
      id<MTLDevice> device,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);
  absl::Status WriteData(
      id<MTLDevice> device,
      const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  template <DataType T>
  absl::Status WriteData(id<MTLDevice> device, const tflite::gpu::Tensor<BHWC, T>& src);
  template <DataType T>
  absl::Status WriteData(id<MTLDevice> device, const tflite::gpu::Tensor<BHWDC, T>& src);
  template <DataType T>
  absl::Status ReadData(id<MTLDevice> device, tflite::gpu::Tensor<BHWC, T>* dst) const;
  template <DataType T>
  absl::Status ReadData(id<MTLDevice> device, tflite::gpu::Tensor<BHWDC, T>* dst) const;

  absl::Status CreateFromDescriptor(const TensorDescriptor& desc,
                                    id<MTLDevice> device);
  absl::Status ToDescriptor(TensorDescriptor* desc, id<MTLDevice> device) const;

  absl::Status SetBufferHandle(id<MTLBuffer> buffer);
  id<MTLBuffer> GetBufferHandle() const;

 private:
  friend absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                               const TensorDescriptor& descriptor,
                                               MetalSpatialTensor* result, uint64_t buffer_offset);

  friend absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      int row_bytes_alignment,
                                                      MetalSpatialTensor* result,
                                                      uint64_t buffer_offset);

  absl::Status IsValid(const BHWC& shape) const;
  absl::Status IsValid(const BHWDC& shape) const;

  template <typename T>
  absl::Status WriteDataBHWDC(id<MTLDevice> device, const T* in);
  absl::Status WriteData(id<MTLDevice> device, const void* ptr);
  template <typename T>
  absl::Status ReadDataBHWDC(id<MTLDevice> device, T* out) const;
  absl::Status ReadData(id<MTLDevice> device, void* ptr) const;

  int GetAlignedChannels() const;
  int3 GetFullTensorRegion() const;
  void Release();

  id<MTLBuffer> memory_;
  id<MTLTexture> texture_mem_;
  bool memory_owner_;
  bool texture_mem_owner_;
  BHWDC shape_;
  TensorDescriptor descriptor_;
  // for use with TEXTURE_2D and when texture created from buffer.
  int aligned_texture_width_;
  // used when created from shared buffer
  uint64_t buffer_offset_ = 0;
};

absl::Status CreateTensor(id<MTLDevice> device, const BHWC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateTensor(id<MTLDevice> device, const BHWDC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result, uint64_t buffer_offset = 0);

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result, uint64_t buffer_offset = 0);

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment, MetalSpatialTensor* result,
                                             uint64_t buffer_offset = 0);

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment, MetalSpatialTensor* result,
                                             uint64_t buffer_offset = 0);

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);

template <DataType T>
absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const tflite::gpu::Tensor<BHWC, T>& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(device, src.data.data());
}

template <DataType T>
absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const tflite::gpu::Tensor<BHWDC, T>& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(device, src.data.data());
}

template <DataType T>
absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          tflite::gpu::Tensor<BHWC, T>* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(device, dst->data.data());
}

template <DataType T>
absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          tflite::gpu::Tensor<BHWDC, T>* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(device, dst->data.data());
}

template <typename T>
absl::Status MetalSpatialTensor::WriteDataBHWDC(id<MTLDevice> device, const T* in) {
  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[GetMemorySizeInBytes()]);
  if (descriptor_.GetDataType() == DataType::FLOAT16) {
    // rearrangement and conversion from float32 to float16
    DataFromBHWDC(reinterpret_cast<const float*>(in), shape_, descriptor_,
                  reinterpret_cast<half*>(data_copy.get()));
  } else {
    // rearrangement
    DataFromBHWDC(in, shape_, descriptor_, reinterpret_cast<T*>(data_copy.get()));
  }

  return WriteData(device, data_copy.get());
}

template <typename T>
absl::Status MetalSpatialTensor::ReadDataBHWDC(id<MTLDevice> device, T* out) const {
  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[GetMemorySizeInBytes()]);

  RETURN_IF_ERROR(ReadData(device, data_copy.get()));

  if (descriptor_.GetDataType() == DataType::FLOAT16) {
    // rearrangement and conversion from float32 to float16
    DataToBHWDC(reinterpret_cast<half*>(data_copy.get()), shape_, descriptor_,
                reinterpret_cast<float*>(out));
  } else {
    // rearrangement
    DataToBHWDC(reinterpret_cast<T*>(data_copy.get()), shape_, descriptor_, out);
  }

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_
