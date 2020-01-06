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

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
Status CreateImageBufferFromBuffer(const CLContext& context, cl_mem memory,
                                   enum DataType data_type, int width,
                                   cl_mem* result) {
  cl_image_format format;
  cl_image_desc desc;
  std::memset(&desc, 0, sizeof(desc));
  desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  desc.image_width = width;
  desc.mem_object = memory;

  format.image_channel_data_type = ToImageChannelType(data_type);
  format.image_channel_order = CL_RGBA;

  cl_int error;
  *result = clCreateImage(context.context(), CL_MEM_READ_WRITE, &format, &desc,
                          nullptr, &error);
  if (error != CL_SUCCESS) {
    return UnknownError(
        absl::StrCat("Failed to create Texture2D (clCreateImage)",
                     CLErrorCodeToString(error)));
  }
  return OkStatus();
}

Status CreateTensor(const CLContext& context, const CLDevice& device,
                    const BHWC& shape, const TensorDescriptor& descriptor,
                    cl_mem memory, Tensor* result) {
  const bool memory_owner = memory == nullptr;
  if (memory_owner) {
    CLMemory mem;
    RETURN_IF_ERROR(
        AllocateTensorMemory(context, device, shape, descriptor, &mem));
    memory = mem.Release();
  }
  if (descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    cl_mem image_memory;
    RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        context, memory, descriptor.data_type,
        shape.b * shape.w * shape.h * IntegralDivideRoundUp(shape.c, 4),
        &image_memory));
    *result = Tensor(memory, memory_owner, image_memory, shape, descriptor);
  } else {
    *result = Tensor(memory, memory_owner, shape, descriptor);
  }
  return OkStatus();
}
}  // namespace

Tensor::Tensor(cl_mem memory, bool memory_owner, const BHWC& shape,
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
      shape_(shape),
      descriptor_(descriptor) {}

Tensor::Tensor(Tensor&& tensor)
    : memory_(tensor.memory_),
      image_buffer_memory_(tensor.image_buffer_memory_),
      memory_owner_(tensor.memory_owner_),
      shape_(tensor.shape_),
      descriptor_(tensor.descriptor_) {
  tensor.memory_ = nullptr;
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
  if (image_buffer_memory_) {
    clReleaseMemObject(image_buffer_memory_);
    memory_ = nullptr;
  }
  if (memory_owner_ && memory_) {
    clReleaseMemObject(memory_);
    memory_ = nullptr;
  }
}

int3 Tensor::GetFullTensorRegion() const {
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::IMAGE_BUFFER:
      return {shape_.w * shape_.b, shape_.h, Slices()};
    case TensorStorageType::TEXTURE_2D:
      return {shape_.w * shape_.b, shape_.h * Slices(), 1};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {shape_.w * shape_.b, shape_.h, 1};
    case TensorStorageType::UNKNOWN:
      return {-1, -1, -1};
  }
}

Status Tensor::IsValid(const BHWC& shape) const {
  if (shape.b != shape_.b) {
    return InvalidArgumentError("Shape batch does not match tensor batch");
  }
  if (shape.w != shape_.w) {
    return InvalidArgumentError("Shape width does not match tensor width");
  }
  if (shape.h != shape_.h) {
    return InvalidArgumentError("Shape height does not match tensor height");
  }
  if (shape.c != shape_.c) {
    return InvalidArgumentError(
        "Shape channels does not match tensor channels");
  }
  return OkStatus();
}

int Tensor::GetChannelsAlignment() const {
  return descriptor_.storage_type == TensorStorageType::SINGLE_TEXTURE_2D
             ? shape_.c
             : 4;
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
      return flt4_size * shape_.b * shape_.w * shape_.h * Slices();
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return flt_size * shape_.w * shape_.h * shape_.c * shape_.b;
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

Status Tensor::WriteDataBHWC(absl::Span<const float> in,
                             CLCommandQueue* queue) {
  void* data_ptr = nullptr;
  const int aligned_channels = GetAlignedChannels();
  const int elements_count = shape_.b * shape_.w * shape_.h * aligned_channels;

  const size_t data_size = elements_count * SizeOf(descriptor_.data_type);
  std::vector<float> data_f;
  std::vector<half> data_h;
  if (descriptor_.data_type == DataType::FLOAT32) {
    data_f.resize(elements_count);
    data_ptr = data_f.data();
    DataFromBHWC(in, absl::MakeSpan(data_f.data(), data_f.size()));
  } else {
    data_h.resize(elements_count);
    data_ptr = data_h.data();
    DataFromBHWC(in, absl::MakeSpan(data_h.data(), data_h.size()));
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
      return InternalError("Unsupported tensor storage type");
  }

  return OkStatus();
}

Status Tensor::WriteData(CLCommandQueue* queue, const TensorFloat32& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWC(absl::MakeConstSpan(src.data), queue);
}

Status Tensor::ReadDataBHWC(absl::Span<float> out,
                            CLCommandQueue* queue) const {
  void* data_ptr = nullptr;
  const int aligned_channels = GetAlignedChannels();
  const int elements_count = shape_.b * shape_.w * shape_.h * aligned_channels;
  const size_t data_size = elements_count * SizeOf(descriptor_.data_type);
  std::vector<float> data_f;
  std::vector<half> data_h;
  if (descriptor_.data_type == DataType::FLOAT32) {
    data_f.resize(elements_count);
    data_ptr = data_f.data();
  } else {
    data_h.resize(elements_count);
    data_ptr = data_h.data();
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
      return InternalError("Unsupported tensor storage type");
  }

  if (descriptor_.data_type == DataType::FLOAT32) {
    DataToBHWC(absl::MakeConstSpan(data_f.data(), data_f.size()), out);
  } else {
    DataToBHWC(absl::MakeConstSpan(data_h.data(), data_h.size()), out);
  }

  return OkStatus();
}

Status Tensor::ReadData(CLCommandQueue* queue, TensorFloat32* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWC(absl::MakeSpan(dst->data), queue);
}

bool CanCreateTensorWithShape(const CLContext& context, const CLDevice& device,
                              const BHWC& shape,
                              const TensorDescriptor& descriptor) {
  const int slices = IntegralDivideRoundUp(shape.c, 4);
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER: {
      const int flt4_size =
          4 * (descriptor.data_type == DataType::FLOAT32 ? 4 : 2);
      const int buffer_size = shape.b * shape.w * shape.h * slices * flt4_size;
      return buffer_size <= device.GetInfo().buffer_max_size;
    }
    case TensorStorageType::IMAGE_BUFFER:
      return shape.b * shape.w * shape.h * slices <=
             device.GetInfo().image_buffer_max_size;
    case TensorStorageType::TEXTURE_3D:
      if (device.cl_version() < OpenCLVersion::CL_1_2 && slices == 1) {
        // clCreateImage3D (that used in CL 1.0/1.1) can not create image with
        // depth = 1 by specification;
        return false;
      }
      return shape.w * shape.b <= device.GetInfo().image3d_max_width &&
             shape.h <= device.GetInfo().image3d_max_height &&
             slices <= device.GetInfo().image3d_max_depth;
    case TensorStorageType::TEXTURE_ARRAY:
      // Bug on some Adreno. b/131099086
      if (slices == 1 && !device.SupportsOneLayerTextureArray()) {
        return false;
      }
      return shape.w * shape.b <= device.GetInfo().image2d_max_width &&
             shape.h <= device.GetInfo().image2d_max_height &&
             slices <= device.GetInfo().image_array_max_layers;
    case TensorStorageType::TEXTURE_2D:
      return shape.w * shape.b <= device.GetInfo().image2d_max_width &&
             shape.h * slices <= device.GetInfo().image2d_max_height;
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return shape.c <= 4 &&
             context.IsFloatTexture2DSupported(shape.c, descriptor.data_type) &&
             shape.w * shape.b <= device.GetInfo().image2d_max_width &&
             shape.h <= device.GetInfo().image2d_max_height;
    default:
      return false;
  }
}

Status CreateTensor(const CLContext& context, const CLDevice& device,
                    const BHWC& shape, const TensorDescriptor& descriptor,
                    Tensor* result) {
  return CreateTensor(context, device, shape, descriptor, nullptr, result);
}

Status CreateSharedTensor(const CLContext& context, const CLDevice& device,
                          cl_mem memory, const BHWC& shape,
                          const TensorDescriptor& descriptor, Tensor* result) {
  return CreateTensor(context, device, shape, descriptor, memory, result);
}

Status AllocateTensorMemory(const CLContext& context, const CLDevice& device,
                            const BHWC& shape,
                            const TensorDescriptor& descriptor,
                            CLMemory* result) {
  const int slices = IntegralDivideRoundUp(shape.c, 4);
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size = shape.b * shape.w * shape.h * slices * 4 *
                               SizeOf(descriptor.data_type);
      cl_int error_code;
      cl_mem memory = clCreateBuffer(context.context(), CL_MEM_READ_WRITE,
                                     data_size, nullptr, &error_code);
      if (!memory) {
        return UnknownError(
            absl::StrCat("Failed to allocate device memory with clCreateBuffer",
                         CLErrorCodeToString(error_code)));
      }
      *result = CLMemory(memory, true);
      return OkStatus();
    }
    case TensorStorageType::TEXTURE_2D: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = shape.w * shape.b;
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
      cl_mem memory = CreateImage2DLegacy(context.context(), CL_MEM_READ_WRITE,
                                          &format, &desc, nullptr, &error_code);
      if (error_code != CL_SUCCESS) {
        return UnknownError(
            absl::StrCat("Failed to create Texture2D (clCreateImage)",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return OkStatus();
    }
    case TensorStorageType::TEXTURE_3D: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE3D;
      desc.image_width = shape.w * shape.b;
      desc.image_height = shape.h;
      desc.image_depth = slices;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = ToImageChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory = CreateImage3DLegacy(context.context(), CL_MEM_READ_WRITE,
                                          &format, &desc, nullptr, &error_code);
      if (error_code != CL_SUCCESS) {
        return UnknownError(
            absl::StrCat("Failed to create Texture3D (clCreateImage)",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return OkStatus();
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
      desc.image_width = shape.w * shape.b;
      desc.image_height = shape.h;
      desc.image_depth = 0;
      desc.image_array_size = slices;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = ToImageChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory = clCreateImage(context.context(), CL_MEM_READ_WRITE,
                                    &format, &desc, nullptr, &error_code);
      if (error_code != CL_SUCCESS) {
        return UnknownError(
            absl::StrCat("Failed to create TextureArray (clCreateImage)",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return OkStatus();
    }

    case TensorStorageType::SINGLE_TEXTURE_2D: {
      if (slices != 1) {
        return InvalidArgumentError(absl::StrCat(
            "SINGLE_TEXTURE_2D support only cnannels in range [1-4], but ",
            shape.c, "was provided"));
      }
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = shape.w * shape.b;
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
        return InvalidArgumentError(absl::StrCat(
            "This device doesn't support ", shape.c, "-channel textures."));
      }

      cl_int error_code;
      cl_mem memory = CreateImage2DLegacy(context.context(), CL_MEM_READ_WRITE,
                                          &format, &desc, nullptr, &error_code);
      if (error_code != CL_SUCCESS) {
        return UnknownError(
            absl::StrCat("Failed to create Texture2D (clCreateImage)",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return OkStatus();
    }

    default:
      return InternalError("Unsupported tensor storage type");
  }
}

template <typename T>
void Tensor::DataFromBHWC(absl::Span<const float> src,
                          absl::Span<T> dst) const {
  const int channels_batch = GetChannelsAlignment();
  for (int b = 0; b < shape_.b; ++b) {
    for (int s = 0; s < Slices(); ++s) {
      for (int y = 0; y < shape_.h; ++y) {
        for (int x = 0; x < shape_.w; ++x) {
          for (int c = 0; c < channels_batch; ++c) {
            float value;
            if (s * 4 + c < shape_.c) {
              const int cpu_index = shape_.LinearIndex({b, y, x, s * 4 + c});
              value = src[cpu_index];
            } else {
              value = 0.0f;
            }
            const int gpu_index = GetLinearIndex(b, x, y, s, c);
            dst[gpu_index] = value;
          }
        }
      }
    }
  }
}

template void Tensor::DataFromBHWC<float>(absl::Span<const float> src,
                                          absl::Span<float> dst) const;
template void Tensor::DataFromBHWC<half>(absl::Span<const float> src,
                                         absl::Span<half> dst) const;

template <typename T>
void Tensor::DataToBHWC(absl::Span<const T> src, absl::Span<float> dst) const {
  const int channels_batch = GetChannelsAlignment();
  for (int b = 0; b < shape_.b; ++b) {
    for (int s = 0; s < Slices(); ++s) {
      for (int y = 0; y < shape_.h; ++y) {
        for (int x = 0; x < shape_.w; ++x) {
          for (int c = 0; c < channels_batch; ++c) {
            if (s * 4 + c >= shape_.c) {
              continue;
            }
            const int cpu_index = shape_.LinearIndex({b, y, x, s * 4 + c});
            const int gpu_index = GetLinearIndex(b, x, y, s, c);
            dst[cpu_index] = src[gpu_index];
          }
        }
      }
    }
  }
}

template void Tensor::DataToBHWC<float>(absl::Span<const float> src,
                                        absl::Span<float> dst) const;
template void Tensor::DataToBHWC<half>(absl::Span<const half> src,
                                       absl::Span<float> dst) const;

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
