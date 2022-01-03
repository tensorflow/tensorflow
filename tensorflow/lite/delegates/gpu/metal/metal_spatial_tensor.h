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
  DataType GetDataType() const { return descriptor_.data_type; }
  TensorStorageType GetStorageType() const { return descriptor_.storage_type; }

  // for profiling and memory statistics
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

  absl::Status SetBufferHandle(id<MTLBuffer> buffer);
  id<MTLBuffer> GetBufferHandle() const;

 private:
  friend absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      int row_bytes_alignment,
                                                      MetalSpatialTensor* result);

  absl::Status IsValid(const BHWC& shape) const;
  absl::Status IsValid(const BHWDC& shape) const;

  template <typename T>
  absl::Status WriteDataBHWDC(id<MTLDevice> device, const T* in);
  template <typename T>
  absl::Status ReadDataBHWDC(id<MTLDevice> device, T* out) const;

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

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment, MetalSpatialTensor* result);

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment, MetalSpatialTensor* result);

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
  const int aligned_channels = GetAlignedChannels();
  const int elements_count = shape_.b * shape_.w * shape_.h * shape_.d * aligned_channels;

  const size_t data_size = elements_count * SizeOf(descriptor_.data_type);
  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[data_size]);
  if (descriptor_.data_type == DataType::FLOAT16) {
    // rearrangement and conversion from float32 to float16
    DataFromBHWDC(reinterpret_cast<const float*>(in), shape_, descriptor_,
                  reinterpret_cast<half*>(data_copy.get()));
  } else {
    // rearrangement
    DataFromBHWDC(in, shape_, descriptor_, reinterpret_cast<T*>(data_copy.get()));
  }

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy([memory_ contents], data_copy.get(), data_size);
      break;
    case TensorStorageType::TEXTURE_2D:
      WriteDataToTexture2D(texture_mem_, device, data_copy.get());
      break;
    case TensorStorageType::TEXTURE_3D:
      WriteDataToTexture3D(texture_mem_, device, data_copy.get());
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      WriteDataToTexture2DArray(texture_mem_, device, data_copy.get());
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }

  return absl::OkStatus();
}

template <typename T>
absl::Status MetalSpatialTensor::ReadDataBHWDC(id<MTLDevice> device, T* out) const {
  const int aligned_channels = GetAlignedChannels();
  const int elements_count = shape_.b * shape_.w * shape_.h * shape_.d * aligned_channels;
  const size_t data_size = elements_count * SizeOf(descriptor_.data_type);
  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[data_size]);

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy(data_copy.get(), [memory_ contents], data_size);
      break;
    case TensorStorageType::TEXTURE_2D:
      ReadDataFromTexture2D(texture_mem_, device, data_copy.get());
      break;
    case TensorStorageType::TEXTURE_3D:
      ReadDataFromTexture3D(texture_mem_, device, data_copy.get());
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      ReadDataFromTexture2DArray(texture_mem_, device, data_copy.get());
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }

  if (descriptor_.data_type == DataType::FLOAT16) {
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
