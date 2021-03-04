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

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

absl::Status CreateTextureBuffer(id<MTLBuffer> buffer, const BHWDC& shape,
                                 const TensorDescriptor& descriptor,
                                 id<MTLTexture>* texture) {
  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    const int slices = DivideRoundUp(shape.c, 4);
    const size_t flt4_count = shape.b * shape.w * shape.h * shape.d * slices;
    const size_t data_size = flt4_count * 4 * SizeOf(descriptor.data_type);
    MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
    texture_desc.width = flt4_count;
    texture_desc.pixelFormat =
        DataTypeToRGBAPixelFormat(descriptor.data_type, false);
    texture_desc.textureType = MTLTextureTypeTextureBuffer;
    texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    texture_desc.storageMode = buffer.storageMode;
    *texture = [buffer newTextureWithDescriptor:texture_desc
                                         offset:0
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

absl::Status AllocateTensorMemory(id<MTLDevice> device, const BHWDC& shape,
                                  const TensorDescriptor& descriptor,
                                  const void* data_ptr, id<MTLBuffer>* buffer,
                                  id<MTLTexture>* texture) {
  const int slices = DivideRoundUp(shape.c, 4);
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size = shape.b * shape.w * shape.h * shape.d * slices *
                               4 * SizeOf(descriptor.data_type);
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
      if (descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
        RETURN_IF_ERROR(
            CreateTextureBuffer(*buffer, shape, descriptor, texture));
      }
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_2D: {
      MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:DataTypeToRGBAPixelFormat(
                                                 descriptor.data_type, false)
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
          DataTypeToRGBAPixelFormat(descriptor.data_type, false);
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
          DataTypeToRGBAPixelFormat(descriptor.data_type, false);
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

absl::Status CreateTensor(id<MTLDevice> device, const BHWDC& shape,
                          const TensorDescriptor& descriptor,
                          id<MTLBuffer> buffer, id<MTLTexture> texture,
                          MetalSpatialTensor* result) {
  const bool user_provided = buffer != nullptr || texture != nullptr;
  const bool memory_owner = !user_provided;
  if (memory_owner) {
    RETURN_IF_ERROR(AllocateTensorMemory(device, shape, descriptor, nullptr,
                                         &buffer, &texture));
  }

  *result = MetalSpatialTensor(buffer, texture, memory_owner, memory_owner,
                               shape, descriptor);
  return absl::OkStatus();
}
}  // namespace

MetalSpatialTensor::MetalSpatialTensor(id<MTLBuffer> buffer,
                                       id<MTLTexture> texture,
                                       bool memory_owner,
                                       bool texture_mem_owner,
                                       const BHWC& shape,
                                       const TensorDescriptor& descriptor)
    : memory_(buffer),
      texture_mem_(texture),
      memory_owner_(memory_owner),
      texture_mem_owner_(texture_mem_owner),
      shape_(shape.b, shape.h, shape.w, 1, shape.c),
      descriptor_(descriptor) {}

MetalSpatialTensor::MetalSpatialTensor(id<MTLBuffer> buffer,
                                       id<MTLTexture> texture,
                                       bool memory_owner,
                                       bool texture_mem_owner,
                                       const BHWDC& shape,
                                       const TensorDescriptor& descriptor)
    : memory_(buffer),
      texture_mem_(texture),
      memory_owner_(memory_owner),
      texture_mem_owner_(texture_mem_owner),
      shape_(shape),
      descriptor_(descriptor) {}

MetalSpatialTensor::MetalSpatialTensor(MetalSpatialTensor&& tensor)
    : memory_(tensor.memory_),
      texture_mem_(tensor.texture_mem_),
      memory_owner_(tensor.memory_owner_),
      texture_mem_owner_(tensor.texture_mem_owner_),
      shape_(tensor.shape_),
      descriptor_(tensor.descriptor_) {
  tensor.memory_ = nullptr;
}

MetalSpatialTensor& MetalSpatialTensor::operator=(MetalSpatialTensor&& tensor) {
  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(texture_mem_, tensor.texture_mem_);
    std::swap(memory_owner_, tensor.memory_owner_);
    std::swap(texture_mem_owner_, tensor.texture_mem_owner_);
    std::swap(shape_, tensor.shape_);
    std::swap(descriptor_, tensor.descriptor_);
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
    if (descriptor_.storage_type != TensorStorageType::BUFFER) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only wtih "
          "TensorStorageType::BUFFER.");
    }
    resources->buffers.push_back({"buffer", memory_});
    return absl::OkStatus();
  }
  const auto* texture2d_desc =
      dynamic_cast<const Texture2DDescriptor*>(obj_ptr);
  if (texture2d_desc) {
    if (descriptor_.storage_type != TensorStorageType::TEXTURE_2D) {
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
  resources->ints.push_back(
      {"slice_stride", tensor_desc->GetSliceStrideSize(shape_)});
  if (descriptor_.HasAxis(Axis::WIDTH)) {
    resources->ints.push_back({"width", Width()});
    resources->ints.push_back({"width_div2", Width() / 2});
    resources->ints.push_back({"width_div4", Width() / 4});
    resources->ints.push_back({"width_batched", Width() * Batch()});
    resources->ints.push_back({"width_batched_div2", Width() * Batch() / 2});
    resources->ints.push_back({"width_batched_div4", Width() * Batch() / 4});
  }
  if (descriptor_.HasAxis(Axis::HEIGHT)) {
    resources->ints.push_back({"height", Height()});
  }
  if (descriptor_.HasAxis(Axis::CHANNELS)) {
    resources->ints.push_back({"slices", Slices()});
    resources->ints.push_back({"channels", Channels()});
  }
  if (descriptor_.HasAxis(Axis::BATCH)) {
    resources->ints.push_back({"batch", Batch()});
  }
  if (descriptor_.HasAxis(Axis::DEPTH)) {
    resources->ints.push_back({"depth", Depth()});
  }

  if (descriptor_.storage_type == TensorStorageType::BUFFER) {
    resources->buffers.push_back({"buffer", memory_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_2D) {
    resources->images2d.push_back({"image2d", texture_mem_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_3D) {
    resources->images3d.push_back({"image3d", texture_mem_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_ARRAY) {
    resources->image2d_arrays.push_back({"image2d_array", texture_mem_});
  } else if (descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (obj_ptr->GetAccess() == AccessType::READ) {
      resources->image_buffers.push_back({"image_buffer", texture_mem_});
    } else {
      resources->buffers.push_back({"buffer", memory_});
    }
  }

  return absl::OkStatus();
}

int3 MetalSpatialTensor::GetFullTensorRegion() const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::IMAGE_BUFFER:
      return {shape_.w * shape_.b, shape_.h, shape_.d * Slices()};
    case TensorStorageType::TEXTURE_2D:
      return {shape_.w * shape_.b * shape_.d, shape_.h * Slices(), 1};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {shape_.w * shape_.b * shape_.d, shape_.h, 1};
    case TensorStorageType::UNKNOWN:
      return {-1, -1, -1};
  }
}

absl::Status MetalSpatialTensor::IsValid(const BHWC& shape) const {
  if (shape.b != shape_.b) {
    return absl::InvalidArgumentError(
        "Shape batch does not match tensor batch");
  }
  if (shape.w != shape_.w) {
    return absl::InvalidArgumentError(
        "Shape width does not match tensor width");
  }
  if (shape.h != shape_.h) {
    return absl::InvalidArgumentError(
        "Shape height does not match tensor height");
  }
  if (shape.c != shape_.c) {
    return absl::InvalidArgumentError(
        "Shape channels does not match tensor channels");
  }
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::IsValid(const BHWDC& shape) const {
  if (shape.b != shape_.b) {
    return absl::InvalidArgumentError(
        "Shape batch does not match tensor batch");
  }
  if (shape.w != shape_.w) {
    return absl::InvalidArgumentError(
        "Shape width does not match tensor width");
  }
  if (shape.h != shape_.h) {
    return absl::InvalidArgumentError(
        "Shape height does not match tensor height");
  }
  if (shape.d != shape_.d) {
    return absl::InvalidArgumentError(
        "Shape depth does not match tensor depth");
  }
  if (shape.c != shape_.c) {
    return absl::InvalidArgumentError(
        "Shape channels does not match tensor channels");
  }
  return absl::OkStatus();
}

uint64_t MetalSpatialTensor::GetMemorySizeInBytes() const {
  const int flt_size = SizeOf(descriptor_.data_type);
  const int flt4_size = 4 * flt_size;
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
      return flt4_size * shape_.b * shape_.w * shape_.h * shape_.d * Slices();
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return flt_size * shape_.w * shape_.h * shape_.c * shape_.b * shape_.d;
    default:
      return 0;
  }
}

int MetalSpatialTensor::GetAlignedChannels() const {
  return descriptor_.storage_type == TensorStorageType::SINGLE_TEXTURE_2D
             ? shape_.c
             : AlignByN(shape_.c, 4);
}

absl::Status MetalSpatialTensor::WriteDataBHWDC(id<MTLDevice> device,
                                                const float* in) {
  void* data_ptr = nullptr;
  const int aligned_channels = GetAlignedChannels();
  const int elements_count =
      shape_.b * shape_.w * shape_.h * shape_.d * aligned_channels;

  const size_t data_size = elements_count * SizeOf(descriptor_.data_type);
  std::unique_ptr<float[]> data_f;
  std::unique_ptr<half[]> data_h;
  if (descriptor_.data_type == DataType::FLOAT32) {
    data_f.reset(new float[elements_count]);
    data_ptr = data_f.get();
    DataFromBHWDC(in, shape_, descriptor_, data_f.get());
  } else {
    data_h.reset(new half[elements_count]);
    data_ptr = data_h.get();
    DataFromBHWDC(in, shape_, descriptor_, data_h.get());
  }

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy([memory_ contents], data_ptr, data_size);
      break;
    case TensorStorageType::TEXTURE_2D:
      WriteDataToTexture2D(texture_mem_, device, data_ptr);
      break;
    case TensorStorageType::TEXTURE_3D:
      WriteDataToTexture3D(texture_mem_, device, data_ptr);
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      WriteDataToTexture2DArray(texture_mem_, device, data_ptr);
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }

  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const TensorFloat32& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(device, src.data.data());
}

absl::Status MetalSpatialTensor::WriteData(
    id<MTLDevice> device,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
  return WriteDataBHWDC(device, src.data.data());
}

absl::Status MetalSpatialTensor::WriteData(
    id<MTLDevice> device,
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
  return WriteDataBHWDC(device, src.data.data());
}

absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const Tensor5DFloat32& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(device, src.data.data());
}

absl::Status MetalSpatialTensor::ReadDataBHWDC(id<MTLDevice> device,
                                               float* out) const {
  void* data_ptr = nullptr;
  const int aligned_channels = GetAlignedChannels();
  const int elements_count =
      shape_.b * shape_.w * shape_.h * shape_.d * aligned_channels;
  const size_t data_size = elements_count * SizeOf(descriptor_.data_type);
  std::unique_ptr<float[]> data_f;
  std::unique_ptr<half[]> data_h;
  if (descriptor_.data_type == DataType::FLOAT32) {
    data_f.reset(new float[elements_count]);
    data_ptr = data_f.get();
  } else {
    data_h.reset(new half[elements_count]);
    data_ptr = data_h.get();
  }

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy(data_ptr, [memory_ contents], data_size);
      break;
    case TensorStorageType::TEXTURE_2D:
      ReadDataFromTexture2D(texture_mem_, device, data_ptr);
      break;
    case TensorStorageType::TEXTURE_3D:
      ReadDataFromTexture3D(texture_mem_, device, data_ptr);
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      ReadDataFromTexture2DArray(texture_mem_, device, data_ptr);
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }

  if (descriptor_.data_type == DataType::FLOAT32) {
    DataToBHWDC(data_f.get(), shape_, descriptor_, out);
  } else {
    DataToBHWDC(data_h.get(), shape_, descriptor_, out);
  }

  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          TensorFloat32* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(device, dst->data.data());
}

absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          Tensor5DFloat32* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(device, dst->data.data());
}

absl::Status MetalSpatialTensor::CreateFromDescriptor(
    const TensorDescriptor& desc, id<MTLDevice> device) {
  shape_ = desc.shape;
  descriptor_.data_type = desc.data_type;
  descriptor_.storage_type = desc.storage_type;
  descriptor_.layout = desc.layout;
  memory_owner_ = true;
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  id<MTLBuffer> buffer;
  id<MTLTexture> texture;
  RETURN_IF_ERROR(AllocateTensorMemory(device, shape_, descriptor_, data_ptr,
                                       &buffer, &texture));
  memory_ = buffer;
  texture_mem_ = texture;
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
  if (descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER) {
    id<MTLTexture> texture_buffer = nullptr;
    RETURN_IF_ERROR(
        CreateTextureBuffer(memory_, shape_, descriptor_, &texture_buffer));
    texture_mem_ = texture_buffer;
  }
  return absl::OkStatus();
}

id<MTLBuffer> MetalSpatialTensor::GetBufferHandle() const { return memory_; }

absl::Status CreateTensor(id<MTLDevice> device, const BHWC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result) {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateTensor(device, shape5D, descriptor, nullptr, nullptr, result);
}

absl::Status CreateTensor(id<MTLDevice> device, const BHWDC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result) {
  return CreateTensor(device, shape, descriptor, nullptr, nullptr, result);
}

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result) {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  id<MTLTexture> texture_buffer = nullptr;
  if (buffer && descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    RETURN_IF_ERROR(
        CreateTextureBuffer(buffer, shape5D, descriptor, &texture_buffer));
  }
  *result = MetalSpatialTensor(buffer, texture_buffer, false, true, shape5D,
                               descriptor);
  return absl::OkStatus();
}

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result) {
  id<MTLTexture> texture_buffer = nullptr;
  if (buffer && descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    RETURN_IF_ERROR(
        CreateTextureBuffer(buffer, shape, descriptor, &texture_buffer));
  }
  *result = MetalSpatialTensor(buffer, texture_buffer, false, true, shape,
                               descriptor);
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
