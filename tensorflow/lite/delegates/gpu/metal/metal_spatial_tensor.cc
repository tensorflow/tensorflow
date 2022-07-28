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

#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

absl::Status CreateTextureBuffer(id<MTLBuffer> buffer, uint64_t buffer_offset,
                                 const TensorDescriptor& descriptor,
                                 id<MTLTexture>* texture) {
  const BHWDC& shape = descriptor.GetBHWDCShape();
  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    const int slices = DivideRoundUp(shape.c, 4);
    const size_t flt4_count = shape.b * shape.w * shape.h * shape.d * slices;
    const size_t data_size = flt4_count * 4 * SizeOf(descriptor.GetDataType());
    MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
    texture_desc.width = flt4_count;
    texture_desc.pixelFormat =
        DataTypeToRGBAPixelFormat(descriptor.GetDataType(), false);
    texture_desc.textureType = MTLTextureTypeTextureBuffer;
    texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    texture_desc.storageMode = buffer.storageMode;
    *texture = [buffer newTextureWithDescriptor:texture_desc
                                         offset:buffer_offset
                                    bytesPerRow:data_size];
    if (!*texture) {
      return absl::UnknownError("Failed to allocate id<MTLTexture>");
    }
  } else {
    return absl::UnknownError(
        "TensorStorageType::IMAGE_BUFFER available only in iOS 12/tvOS "
        "12/macOS 10.14 and higher.");
  }
  return absl::OkStatus();
}

absl::Status AllocateTensorMemory(id<MTLDevice> device,
                                  const TensorDescriptor& descriptor,
                                  id<MTLBuffer>* buffer,
                                  id<MTLTexture>* texture) {
  const BHWDC& shape = descriptor.GetBHWDCShape();
  const void* data_ptr =
      descriptor.GetData().empty() ? nullptr : descriptor.GetData().data();
  const int slices = DivideRoundUp(shape.c, 4);
  switch (descriptor.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size = shape.b * shape.w * shape.h * shape.d * slices *
                               4 * SizeOf(descriptor.GetDataType());
      if (data_ptr) {
        *buffer = [device newBufferWithBytes:data_ptr
                                      length:data_size
                                     options:MTLResourceStorageModeShared];
      } else {
        *buffer = [device newBufferWithLength:data_size
                                      options:MTLResourceStorageModeShared];
      }
      if (!*buffer) {
        return absl::UnknownError("Failed to allocate id<MTLBuffer>");
      }
      if (descriptor.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
        RETURN_IF_ERROR(CreateTextureBuffer(*buffer, 0, descriptor, texture));
      }
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_2D: {
      MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:DataTypeToRGBAPixelFormat(
                                                 descriptor.GetDataType(),
                                                 false)
                                       width:shape.w * shape.b * shape.d
                                      height:shape.h * slices
                                   mipmapped:NO];
      texture_desc.textureType = MTLTextureType2D;
      texture_desc.usage =
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      texture_desc.storageMode = MTLStorageModePrivate;

      *texture = [device newTextureWithDescriptor:texture_desc];
      if (!*texture) {
        return absl::UnknownError("Failed to allocate id<MTLTexture>");
      }
      if (data_ptr) {
        WriteDataToTexture2D(*texture, device, data_ptr);
      }
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_3D: {
      MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
      texture_desc.width = shape.w * shape.b;
      texture_desc.height = shape.h;
      texture_desc.depth = slices * shape.d;
      texture_desc.pixelFormat =
          DataTypeToRGBAPixelFormat(descriptor.GetDataType(), false);
      texture_desc.textureType = MTLTextureType3D;
      texture_desc.usage =
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      texture_desc.storageMode = MTLStorageModePrivate;

      *texture = [device newTextureWithDescriptor:texture_desc];
      if (!*texture) {
        return absl::UnknownError("Failed to allocate id<MTLTexture>");
      }
      if (data_ptr) {
        WriteDataToTexture3D(*texture, device, data_ptr);
      }
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
      texture_desc.width = shape.w * shape.b;
      texture_desc.height = shape.h;
      texture_desc.arrayLength = slices * shape.d;
      texture_desc.pixelFormat =
          DataTypeToRGBAPixelFormat(descriptor.GetDataType(), false);
      texture_desc.textureType = MTLTextureType2DArray;
      texture_desc.usage =
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      texture_desc.storageMode = MTLStorageModePrivate;

      *texture = [device newTextureWithDescriptor:texture_desc];
      if (!*texture) {
        return absl::UnknownError("Failed to allocate id<MTLTexture>");
      }
      if (data_ptr) {
        WriteDataToTexture2DArray(*texture, device, data_ptr);
      }
      return absl::OkStatus();
    }
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
}
}  // namespace

MetalSpatialTensor::MetalSpatialTensor(id<MTLBuffer> buffer,
                                       id<MTLTexture> texture,
                                       bool memory_owner,
                                       bool texture_mem_owner,
                                       const TensorDescriptor& descriptor)
    : memory_(buffer),
      texture_mem_(texture),
      memory_owner_(memory_owner),
      texture_mem_owner_(texture_mem_owner),
      descriptor_(descriptor) {}

MetalSpatialTensor::MetalSpatialTensor(MetalSpatialTensor&& tensor)
    : memory_(tensor.memory_),
      texture_mem_(tensor.texture_mem_),
      memory_owner_(tensor.memory_owner_),
      texture_mem_owner_(tensor.texture_mem_owner_),
      descriptor_(std::move(tensor.descriptor_)),
      aligned_texture_width_(tensor.aligned_texture_width_),
      buffer_offset_(tensor.buffer_offset_) {
  tensor.memory_ = nullptr;
}

MetalSpatialTensor& MetalSpatialTensor::operator=(MetalSpatialTensor&& tensor) {
  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(texture_mem_, tensor.texture_mem_);
    std::swap(memory_owner_, tensor.memory_owner_);
    std::swap(texture_mem_owner_, tensor.texture_mem_owner_);
    descriptor_ = std::move(tensor.descriptor_);
    std::swap(aligned_texture_width_, tensor.aligned_texture_width_);
    std::swap(buffer_offset_, tensor.buffer_offset_);
  }
  return *this;
}

void MetalSpatialTensor::Release() {
  if (memory_owner_ && memory_) {
    memory_ = nullptr;
  }
  if (texture_mem_owner_ && texture_mem_) {
    texture_mem_ = nullptr;
  }
}

absl::Status MetalSpatialTensor::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (buffer_desc) {
    if (descriptor_.GetStorageType() != TensorStorageType::BUFFER) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only wtih "
          "TensorStorageType::BUFFER.");
    }
    resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
    return absl::OkStatus();
  }
  const auto* texture2d_desc =
      dynamic_cast<const Texture2DDescriptor*>(obj_ptr);
  if (texture2d_desc) {
    if (descriptor_.GetStorageType() != TensorStorageType::TEXTURE_2D) {
      return absl::InvalidArgumentError(
          "Tensor can be used with Texture2DDescriptor only wtih "
          "TensorStorageType::TEXTURE_2D.");
    }
    resources->images2d.push_back({"tex2d", texture_mem_});
    return absl::OkStatus();
  }
  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(obj_ptr);
  if (!tensor_desc) {
    return absl::InvalidArgumentError("Expected TensorDescriptor on input.");
  }
  tensor_desc->GetGpuResources(descriptor_.GetBHWDCShape(),
                               &resources->generic);

  if (descriptor_.GetStorageType() == TensorStorageType::BUFFER) {
    resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
  } else if (descriptor_.GetStorageType() == TensorStorageType::TEXTURE_2D) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->GetUseBufferForWriteOnlyTexture2d()) {
      resources->AddInt("aligned_texture_width", aligned_texture_width_);
      resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
    } else {
      resources->images2d.push_back({"image2d", texture_mem_});
    }
  } else if (descriptor_.GetStorageType() == TensorStorageType::TEXTURE_3D) {
    resources->images3d.push_back({"image3d", texture_mem_});
  } else if (descriptor_.GetStorageType() == TensorStorageType::TEXTURE_ARRAY) {
    resources->image2d_arrays.push_back({"image2d_array", texture_mem_});
  } else if (descriptor_.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->GetUseBufferForWriteOnlyImageBuffer()) {
      resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
    } else {
      resources->image_buffers.push_back({"image_buffer", texture_mem_});
    }
  }

  return absl::OkStatus();
}

int3 MetalSpatialTensor::GetFullTensorRegion() const {
  const BHWDC& shape = descriptor_.GetBHWDCShape();
  switch (descriptor_.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::IMAGE_BUFFER:
      return {shape.w * shape.b, shape.h, shape.d * Slices()};
    case TensorStorageType::TEXTURE_2D:
      return {shape.w * shape.b * shape.d, shape.h * Slices(), 1};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {shape.w * shape.b * shape.d, shape.h, 1};
    case TensorStorageType::UNKNOWN:
      return {-1, -1, -1};
  }
}

uint64_t MetalSpatialTensor::GetMemorySizeInBytes() const {
  const BHWDC& shape = descriptor_.GetBHWDCShape();
  const int flt_size = SizeOf(descriptor_.GetDataType());
  const int flt4_size = 4 * flt_size;
  switch (descriptor_.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
      return flt4_size * shape.b * shape.w * shape.h * shape.d * Slices();
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return flt_size * shape.w * shape.h * shape.c * shape.b * shape.d;
    default:
      return 0;
  }
}

absl::Status MetalSpatialTensor::CreateFromDescriptor(
    const TensorDescriptor& desc, id<MTLDevice> device) {
  desc.CopyWithoutData(&descriptor_);
  memory_owner_ = true;
  id<MTLBuffer> buffer;
  id<MTLTexture> texture;
  RETURN_IF_ERROR(AllocateTensorMemory(device, desc, &buffer, &texture));
  memory_ = buffer;
  texture_mem_ = texture;
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::UploadDescriptorData(
    const TensorDescriptor& desc, id<MTLDevice> device) {
  return WriteData(device, desc.GetData().data());
}

absl::Status MetalSpatialTensor::ToDescriptor(TensorDescriptor* desc,
                                              id<MTLDevice> device) const {
  *desc = descriptor_;
  std::vector<uint8_t> data(GetMemorySizeInBytes());
  RETURN_IF_ERROR(ReadData(device, data.data()));
  desc->SetData(std::move(data));
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const void* ptr) {
  switch (descriptor_.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy(
          reinterpret_cast<uint8_t*>([memory_ contents]) + buffer_offset_, ptr,
          GetMemorySizeInBytes());
      break;
    case TensorStorageType::TEXTURE_2D:
      WriteDataToTexture2D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_3D:
      WriteDataToTexture3D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      WriteDataToTexture2DArray(texture_mem_, device, ptr);
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          void* ptr) const {
  switch (descriptor_.GetStorageType()) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy(
          ptr, reinterpret_cast<uint8_t*>([memory_ contents]) + buffer_offset_,
          GetMemorySizeInBytes());
      break;
    case TensorStorageType::TEXTURE_2D:
      ReadDataFromTexture2D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_3D:
      ReadDataFromTexture3D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      ReadDataFromTexture2DArray(texture_mem_, device, ptr);
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::SetBufferHandle(id<MTLBuffer> buffer) {
  if (memory_owner_) {
    return absl::InvalidArgumentError(
        "SetBufferHandle can be used only with shared "
        "Tensors(CreateSharedBufferTensor).");
  }
  if (memory_ == buffer) {
    return absl::OkStatus();
  }
  memory_ = buffer;
  if (descriptor_.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    id<MTLTexture> texture_buffer = nullptr;
    RETURN_IF_ERROR(
        CreateTextureBuffer(memory_, 0, descriptor_, &texture_buffer));
    texture_mem_ = texture_buffer;
  }
  return absl::OkStatus();
}

id<MTLBuffer> MetalSpatialTensor::GetBufferHandle() const { return memory_; }

absl::Status CreateTensor(id<MTLDevice> device,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result) {
  id<MTLBuffer> buffer;
  id<MTLTexture> texture;
  RETURN_IF_ERROR(AllocateTensorMemory(device, descriptor, &buffer, &texture));
  *result = MetalSpatialTensor(buffer, texture, true, true, descriptor);
  return absl::OkStatus();
}

absl::Status CreateTensorSharedBuffer(id<MTLBuffer> buffer,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result,
                                      uint64_t buffer_offset) {
  id<MTLTexture> texture_buffer = nullptr;
  if (buffer &&
      descriptor.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
    RETURN_IF_ERROR(CreateTextureBuffer(buffer, buffer_offset, descriptor,
                                        &texture_buffer));
  }
  *result = MetalSpatialTensor(buffer, texture_buffer, false, true, descriptor);
  result->buffer_offset_ = buffer_offset;
  return absl::OkStatus();
}

absl::Status CreateTensorSharedImage2DBuffer(id<MTLBuffer> buffer,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment,
                                             MetalSpatialTensor* result,
                                             uint64_t buffer_offset) {
  const BHWDC shape = descriptor.GetBHWDCShape();
  const int width = shape.b * shape.w * shape.d;
  const int height = shape.h * DivideRoundUp(shape.c, 4);
  const int channels =
      descriptor.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D
          ? shape.c
          : 4;
  MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
  texture_desc.width = width;
  texture_desc.height = height;
  texture_desc.depth = 1;
  texture_desc.textureType = MTLTextureType2D;
  texture_desc.arrayLength = 1;
  texture_desc.mipmapLevelCount = 1;
  texture_desc.sampleCount = 1;
  texture_desc.pixelFormat =
      DataTypeToRGBAPixelFormat(descriptor.GetDataType(), false);
  texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  texture_desc.storageMode = buffer.storageMode;
  const size_t pixel_size = channels * SizeOf(descriptor.GetDataType());
  const size_t bytes_per_row = width * pixel_size;
  const size_t bytes_per_row_aligned =
      AlignByN(bytes_per_row, row_bytes_alignment);
  id<MTLTexture> texture_buffer =
      [buffer newTextureWithDescriptor:texture_desc
                                offset:buffer_offset
                           bytesPerRow:bytes_per_row_aligned];
  if (!texture_buffer) {
    return absl::UnknownError("Failed to allocate id<MTLTexture>.");
  }
  if (bytes_per_row_aligned % pixel_size != 0) {
    return absl::UnknownError("Alignment mismatch.");
  }
  *result = MetalSpatialTensor(buffer, texture_buffer, false, true, descriptor);
  result->aligned_texture_width_ = bytes_per_row_aligned / pixel_size;
  result->buffer_offset_ = buffer_offset;
  return absl::OkStatus();
}

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info) {
  const bool a7_or_a8 =
      gpu_info.IsApple() && (gpu_info.apple_info.IsA7GenerationGpu() ||
                             gpu_info.apple_info.IsA8GenerationGpu());
  if (a7_or_a8) {
    return TensorStorageType::TEXTURE_2D;
  } else {
    return TensorStorageType::BUFFER;
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
