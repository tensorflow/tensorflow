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

#include "tensorflow/lite/delegates/gpu/cl/tensor.h"

#include <cstring>
#include <memory>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
absl::Status AllocateTensorMemory(const CLContext& context, const BHWDC& shape,
                                  const TensorDescriptor& descriptor,
                                  const void* data_ptr, CLMemory* result) {
  const int slices = DivideRoundUp(shape.c, 4);
  cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
  if (data_ptr) {
    mem_flags |= CL_MEM_COPY_HOST_PTR;
  }
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size = shape.b * shape.w * shape.h * shape.d * slices *
                               4 * SizeOf(descriptor.data_type);
      cl_int error_code;
      cl_mem memory = clCreateBuffer(context.context(), mem_flags, data_size,
                                     const_cast<void*>(data_ptr), &error_code);
      if (!memory) {
        return absl::UnknownError(
            absl::StrCat("Failed to allocate device memory (clCreateBuffer): ",
                         CLErrorCodeToString(error_code)));
      }
      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_2D: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = shape.w * shape.b * shape.d;
      desc.image_height = shape.h * slices;
      desc.image_depth = 0;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = ToImageChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory =
          CreateImage2DLegacy(context.context(), mem_flags, &format, &desc,
                              const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create 2D texture (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_3D: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE3D;
      desc.image_width = shape.w * shape.b;
      desc.image_height = shape.h;
      desc.image_depth = slices * shape.d;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = ToImageChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory =
          CreateImage3DLegacy(context.context(), mem_flags, &format, &desc,
                              const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create 3D texture (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
      desc.image_width = shape.w * shape.b;
      desc.image_height = shape.h;
      desc.image_depth = 0;
      desc.image_array_size = slices * shape.d;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = ToImageChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory =
          clCreateImage(context.context(), mem_flags, &format, &desc,
                        const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create 2D texture array (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }

    case TensorStorageType::SINGLE_TEXTURE_2D: {
      if (slices != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SINGLE_TEXTURE_2D support only channels in range [1-4], but ",
            shape.c, "was provided"));
      }
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = shape.w * shape.b * shape.d;
      desc.image_height = shape.h;
      desc.image_depth = 0;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      if (context.IsFloatTexture2DSupported(shape.c, descriptor.data_type)) {
        format.image_channel_order = ToChannelOrder(shape.c);
        format.image_channel_data_type =
            ToImageChannelType(descriptor.data_type);
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "This device doesn't support ", shape.c, "-channel textures."));
      }

      cl_int error_code;
      cl_mem memory =
          CreateImage2DLegacy(context.context(), mem_flags, &format, &desc,
                              const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create single 2D texture (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }

    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
}

absl::Status CreateImageBufferFromBuffer(const CLContext& context,
                                         cl_mem memory, DataType data_type,
                                         int width, cl_mem* result) {
  cl_image_format format;
  cl_image_desc desc;
  std::memset(&desc, 0, sizeof(desc));
  desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  desc.image_width = width;
  desc.mem_object = memory;

  format.image_channel_data_type = ToImageChannelType(data_type);
  format.image_channel_order = CL_RGBA;

  cl_int error_code;
  *result = clCreateImage(context.context(), CL_MEM_READ_WRITE, &format, &desc,
                          nullptr, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create Image from Buffer (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateTensor(const CLContext& context, const BHWDC& shape,
                          const TensorDescriptor& descriptor, cl_mem memory,
                          Tensor* result) {
  const bool memory_owner = memory == nullptr;
  if (memory_owner) {
    CLMemory mem;
    RETURN_IF_ERROR(
        AllocateTensorMemory(context, shape, descriptor, nullptr, &mem));
    memory = mem.Release();
  }
  if (descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    cl_mem image_memory;
    RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        context, memory, descriptor.data_type,
        shape.b * shape.w * shape.h * shape.d * DivideRoundUp(shape.c, 4),
        &image_memory));
    *result = Tensor(memory, memory_owner, image_memory, shape, descriptor);
  } else {
    *result = Tensor(memory, memory_owner, shape, descriptor);
  }
  return absl::OkStatus();
}

absl::Status CreateTensorShared(const CLContext& context, const BHWDC& shape,
                                const TensorDescriptor& descriptor,
                                cl_mem memory, Tensor* result) {
  const bool memory_owner = false;
  if (descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    cl_mem image_memory;
    RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        context, memory, descriptor.data_type,
        shape.b * shape.w * shape.h * shape.d * DivideRoundUp(shape.c, 4),
        &image_memory));
    *result = Tensor(memory, memory_owner, image_memory, shape, descriptor);
  } else {
    *result = Tensor(memory, memory_owner, shape, descriptor);
  }
  return absl::OkStatus();
}

}  // namespace

Tensor::Tensor(cl_mem memory, bool memory_owner, const BHWC& shape,
               const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(nullptr),
      memory_owner_(memory_owner),
      shape_(shape.b, shape.h, shape.w, 1, shape.c),
      descriptor_(descriptor) {}

Tensor::Tensor(cl_mem memory, bool memory_owner, const BHWDC& shape,
               const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(nullptr),
      memory_owner_(memory_owner),
      shape_(shape),
      descriptor_(descriptor) {}

Tensor::Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
               const BHWC& shape, const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(image_buffer_memory),
      memory_owner_(memory_owner),
      shape_(shape.b, shape.h, shape.w, 1, shape.c),
      descriptor_(descriptor) {}

Tensor::Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
               const BHWDC& shape, const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(image_buffer_memory),
      memory_owner_(memory_owner),
      shape_(shape),
      descriptor_(descriptor) {}

Tensor::Tensor(Tensor&& tensor)
    : memory_(tensor.memory_),
      image_buffer_memory_(tensor.image_buffer_memory_),
      memory_owner_(tensor.memory_owner_),
      shape_(tensor.shape_),
      descriptor_(tensor.descriptor_) {
  tensor.memory_ = nullptr;
  tensor.image_buffer_memory_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) {
  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(image_buffer_memory_, tensor.image_buffer_memory_);
    std::swap(memory_owner_, tensor.memory_owner_);
    std::swap(shape_, tensor.shape_);
    std::swap(descriptor_, tensor.descriptor_);
  }
  return *this;
}

void Tensor::Release() {
  // image_buffer_memory_ always owned by object
  if (image_buffer_memory_) {
    clReleaseMemObject(image_buffer_memory_);
    image_buffer_memory_ = nullptr;
  }
  if (memory_owner_ && memory_) {
    clReleaseMemObject(memory_);
    memory_ = nullptr;
  }
}

absl::Status Tensor::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GPUResourcesWithValue* resources) const {
  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (buffer_desc) {
    if (descriptor_.storage_type != TensorStorageType::BUFFER) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only with "
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
          "Tensor can be used with Texture2DDescriptor only with "
          "TensorStorageType::TEXTURE_2D.");
    }
    resources->images2d.push_back({"tex2d", memory_});
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
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_2D ||
             descriptor_.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
    resources->images2d.push_back({"image2d", memory_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_ARRAY) {
    resources->image2d_arrays.push_back({"image2d_array", memory_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_3D) {
    resources->images3d.push_back({"image3d", memory_});
  } else if (descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (obj_ptr->GetAccess() == AccessType::READ) {
      resources->image_buffers.push_back(
          {"image_buffer", image_buffer_memory_});
    } else {
      resources->buffers.push_back({"buffer", memory_});
    }
  }

  return absl::OkStatus();
}

int3 Tensor::GetFullTensorRegion() const {
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

absl::Status Tensor::IsValid(const BHWC& shape) const {
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

absl::Status Tensor::IsValid(const BHWDC& shape) const {
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

int Tensor::GetAlignedChannels() const {
  return descriptor_.storage_type == TensorStorageType::SINGLE_TEXTURE_2D
             ? shape_.c
             : AlignByN(shape_.c, 4);
}

uint64_t Tensor::GetMemorySizeInBytes() const {
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

cl_mem Tensor::GetMemoryPtr() const {
  return descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER
             ? image_buffer_memory_
             : memory_;
}

cl_mem Tensor::GetMemoryPtrForWriting() const { return memory_; }

absl::Status Tensor::WriteDataBHWDC(const float* in, CLCommandQueue* queue) {
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
      RETURN_IF_ERROR(queue->EnqueueWriteBuffer(memory_, data_size, data_ptr));
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      RETURN_IF_ERROR(
          queue->EnqueueWriteImage(memory_, GetFullTensorRegion(), data_ptr));
      break;
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }

  return absl::OkStatus();
}

absl::Status Tensor::WriteData(CLCommandQueue* queue,
                               const TensorFloat32& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(src.data.data(), queue);
}

absl::Status Tensor::WriteData(
    CLCommandQueue* queue,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
  return WriteDataBHWDC(src.data.data(), queue);
}

absl::Status Tensor::WriteData(
    CLCommandQueue* queue,
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
  return WriteDataBHWDC(src.data.data(), queue);
}

absl::Status Tensor::WriteData(CLCommandQueue* queue,
                               const Tensor5DFloat32& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(src.data.data(), queue);
}

absl::Status Tensor::ReadDataBHWDC(float* out, CLCommandQueue* queue) const {
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
      RETURN_IF_ERROR(queue->EnqueueReadBuffer(memory_, data_size, data_ptr));
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      RETURN_IF_ERROR(
          queue->EnqueueReadImage(memory_, GetFullTensorRegion(), data_ptr));
      break;
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

absl::Status Tensor::ReadData(CLCommandQueue* queue, TensorFloat32* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(dst->data.data(), queue);
}

absl::Status Tensor::ReadData(CLCommandQueue* queue,
                              Tensor5DFloat32* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(dst->data.data(), queue);
}

absl::Status Tensor::CreateFromDescriptor(const TensorDescriptor& desc,
                                          CLContext* context) {
  shape_ = desc.shape;
  descriptor_.data_type = desc.data_type;
  descriptor_.storage_type = desc.storage_type;
  descriptor_.layout = desc.layout;
  memory_owner_ = true;
  CLMemory memory;
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  RETURN_IF_ERROR(
      AllocateTensorMemory(*context, shape_, descriptor_, data_ptr, &memory));
  memory_ = memory.Release();
  if (desc.storage_type == TensorStorageType::IMAGE_BUFFER) {
    RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        *context, memory_, desc.data_type,
        shape_.b * shape_.w * shape_.h * shape_.d * DivideRoundUp(shape_.c, 4),
        &image_buffer_memory_));
  }
  return absl::OkStatus();
}

absl::Status CreateTensor(const CLContext& context, const BHWC& shape,
                          const TensorDescriptor& descriptor, Tensor* result) {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateTensor(context, shape5D, descriptor, nullptr, result);
}

absl::Status CreateTensor(const CLContext& context, const BHWDC& shape,
                          const TensorDescriptor& descriptor, Tensor* result) {
  return CreateTensor(context, shape, descriptor, nullptr, result);
}

absl::Status CreateSharedTensor(const CLContext& context, cl_mem memory,
                                const BHWC& shape,
                                const TensorDescriptor& descriptor,
                                Tensor* result) {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateTensorShared(context, shape5D, descriptor, memory, result);
}

absl::Status CreateSharedTensor(const CLContext& context, cl_mem memory,
                                const BHWDC& shape,
                                const TensorDescriptor& descriptor,
                                Tensor* result) {
  return CreateTensorShared(context, shape, descriptor, memory, result);
}

absl::Status AllocateTensorMemory(const CLContext& context, const BHWC& shape,
                                  const TensorDescriptor& descriptor,
                                  CLMemory* result) {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return AllocateTensorMemory(context, shape5D, descriptor, nullptr, result);
}

absl::Status AllocateTensorMemory(const CLContext& context, const BHWDC& shape,
                                  const TensorDescriptor& descriptor,
                                  CLMemory* result) {
  return AllocateTensorMemory(context, shape, descriptor, nullptr, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
