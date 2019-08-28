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

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

Tensor::Tensor(cl_mem memory, int width, int height, int channels,
               enum DataType data_type, TensorStorageType storage_type)
    : memory_(memory),
      width_(width),
      height_(height),
      channels_(channels),
      data_type_(data_type),
      storage_type_(storage_type) {}

Tensor::Tensor(Tensor&& tensor)
    : memory_(tensor.memory_),
      width_(tensor.width_),
      height_(tensor.height_),
      channels_(tensor.channels_),
      data_type_(tensor.data_type_),
      storage_type_(tensor.storage_type_) {
  tensor.memory_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) {
  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(width_, tensor.width_);
    std::swap(height_, tensor.height_);
    std::swap(channels_, tensor.channels_);
    std::swap(data_type_, tensor.data_type_);
    std::swap(storage_type_, tensor.storage_type_);
  }
  return *this;
}

void Tensor::Release() {
  if (memory_) {
    clReleaseMemObject(memory_);
    memory_ = nullptr;
  }
}

int3 Tensor::GetFullTensorRegion() const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
      return {width_, height_, Depth()};
    case TensorStorageType::TEXTURE_2D:
      return {width_, height_ * Depth(), 1};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {width_, height_, 1};
    case TensorStorageType::UNKNOWN:
      return {-1, -1, -1};
  }
}

Status Tensor::IsValid(const BHWC& shape) const {
  if (shape.b != 1) {
    return InvalidArgumentError("Batch is not equal to 1.");
  }
  if (shape.w != width_) {
    return InvalidArgumentError("Shape width does not match tensor width");
  }
  if (shape.h != height_) {
    return InvalidArgumentError("Shape height does not match tensor height");
  }
  if (shape.c != channels_) {
    return InvalidArgumentError(
        "Shape channels does not match tensor channels");
  }
  return OkStatus();
}

Status Tensor::WriteDataBHWC(absl::Span<const float> in,
                             CLCommandQueue* queue) {
  if (in.size() != channels_ * width_ * height_) {
    return InvalidArgumentError("Input data size not match expected size");
  }

  void* data_ptr = nullptr;
  int channels = storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D
                     ? channels_
                     : AlignByN(channels_, 4);
  const int elements_count = width_ * height_ * channels;

  const size_t data_size = elements_count * SizeOf(data_type_);
  std::vector<float> data_f;
  std::vector<half> data_h;
  if (data_type_ == DataType::FLOAT32) {
    data_f.resize(elements_count);
    data_ptr = data_f.data();
    DataFromBHWC(in, absl::MakeSpan(data_f.data(), data_f.size()));
  } else {
    data_h.resize(elements_count);
    data_ptr = data_h.data();
    DataFromBHWC(in, absl::MakeSpan(data_h.data(), data_h.size()));
  }

  switch (storage_type_) {
    case TensorStorageType::BUFFER: {
      RETURN_IF_ERROR(queue->EnqueueWriteBuffer(memory_, data_size, data_ptr));
      break;
    }
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
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
  if (out.size() != channels_ * width_ * height_) {
    return InvalidArgumentError("Output data size not match expected size");
  }

  void* data_ptr = nullptr;
  int channels = storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D
                     ? channels_
                     : AlignByN(channels_, 4);
  const int elements_count = width_ * height_ * channels;
  const size_t data_size = elements_count * SizeOf(data_type_);
  std::vector<float> data_f;
  std::vector<half> data_h;
  if (data_type_ == DataType::FLOAT32) {
    data_f.resize(elements_count);
    data_ptr = data_f.data();
  } else {
    data_h.resize(elements_count);
    data_ptr = data_h.data();
  }

  switch (storage_type_) {
    case TensorStorageType::BUFFER:
      RETURN_IF_ERROR(queue->EnqueueReadBuffer(memory_, data_size, data_ptr));
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      RETURN_IF_ERROR(
          queue->EnqueueReadImage(memory_, GetFullTensorRegion(), data_ptr));
      break;
    default:
      return InternalError("Unsupported tensor storage type");
  }

  if (data_type_ == DataType::FLOAT32) {
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

Status CreateTensor(const CLContext& context, const CLDevice& device, int width,
                    int height, int channels, DataType data_type,
                    TensorStorageType storage_type, Tensor* result) {
  CLMemory memory;
  RETURN_IF_ERROR(AllocateTensorMemory(context, device, width, height, channels,
                                       data_type, storage_type, &memory));
  *result = Tensor(memory.Release(), width, height, channels, data_type,
                   storage_type);
  return OkStatus();
}

Status AllocateTensorMemory(const CLContext& context, const CLDevice& device,
                            int width, int height, int channels,
                            DataType data_type, TensorStorageType storage_type,
                            CLMemory* result) {
  switch (storage_type) {
    case TensorStorageType::BUFFER: {
      const size_t data_size =
          width * height * AlignByN(channels, 4) * SizeOf(data_type);
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
      desc.image_width = width;
      desc.image_height = height * IntegralDivideRoundUp(channels, 4);
      desc.image_depth = 0;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = ToImageChannelType(data_type);

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
    case TensorStorageType::TEXTURE_ARRAY: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
      desc.image_width = width;
      desc.image_height = height;
      desc.image_depth = 0;
      int layers_count = IntegralDivideRoundUp(channels, 4);
      // Adreno bug. b/131099086
      if (layers_count == 1 && !device.SupportsOneLayerTextureArray()) {
        layers_count = 2;
      }
      desc.image_array_size = layers_count;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = ToImageChannelType(data_type);

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
      if (IntegralDivideRoundUp(channels, 4) != 1) {
        return InvalidArgumentError(absl::StrCat(
            "SINGLE_TEXTURE_2D support only cnannels in range [1-4], but ",
            channels, "was provided"));
      }
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = width;
      desc.image_height = height;
      desc.image_depth = 0;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      if (context.IsFloatTexture2DSupported(channels, data_type)) {
        format.image_channel_order = ToChannelOrder(channels);
        format.image_channel_data_type = ToImageChannelType(data_type);
      } else {
        return InvalidArgumentError(absl::StrCat(
            "This device doesn't support ", channels, "-channel textures."));
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
  int channels =
      storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D ? channels_ : 4;
  BHWC src_shape;
  src_shape.b = 1;
  src_shape.h = height_;
  src_shape.w = width_;
  src_shape.c = channels_;
  for (int d = 0; d < Depth(); ++d) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        for (int c = 0; c < channels; ++c) {
          float value;
          if (d * 4 + c < channels_) {
            const int cpu_index = src_shape.LinearIndex({0, y, x, d * 4 + c});
            value = src[cpu_index];
          } else {
            value = 0.0f;
          }
          const int gpu_index = GetLinearIndex(x, y, d, c);
          dst[gpu_index] = value;
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
  int channels =
      storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D ? channels_ : 4;
  BHWC dst_shape;
  dst_shape.b = 1;
  dst_shape.h = height_;
  dst_shape.w = width_;
  dst_shape.c = channels_;
  for (int d = 0; d < Depth(); ++d) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        for (int c = 0; c < channels; ++c) {
          if (d * 4 + c >= channels_) continue;

          const int cpu_index = dst_shape.LinearIndex({0, y, x, d * 4 + c});
          const int gpu_index = GetLinearIndex(x, y, d, c);
          dst[cpu_index] = src[gpu_index];
        }
      }
    }
  }
}

template void Tensor::DataToBHWC<float>(absl::Span<const float> src,
                                        absl::Span<float> dst) const;
template void Tensor::DataToBHWC<half>(absl::Span<const half> src,
                                       absl::Span<float> dst) const;

TensorBHWC::TensorBHWC(TensorBHWC&& tensor)
    : Tensor(std::move(tensor)), owner_(tensor.owner_) {}

TensorBHWC& TensorBHWC::operator=(TensorBHWC&& tensor) {
  if (this != &tensor) {
    ReleaseBHWC();
    owner_ = tensor.owner_;
    Tensor::operator=(std::move(tensor));
  }
  return *this;
}

void TensorBHWC::ReleaseBHWC() {
  // Base class is handling deletion if we are not owners
  if (!owner_ && memory_) {
    memory_ = nullptr;
  }
}

Status CreateTensorBHWC(const CLContext& context, const HWC& shape,
                        DataType data_type, void* data, Tensor* result) {
  const size_t data_size = shape.w * shape.h * shape.c * SizeOf(data_type);
  cl_int error_code;
  int flags =
      data ? (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR) : CL_MEM_READ_WRITE;
  cl_mem memory =
      clCreateBuffer(context.context(), flags, data_size, data, &error_code);
  if (!memory) {
    return UnknownError(
        absl::StrCat("Failed to allocate device memory with clCreateBuffer",
                     CLErrorCodeToString(error_code)));
  }

  *result = TensorBHWC(memory, shape.w, shape.h, shape.c, data_type,
                       TensorStorageType::BUFFER);
  return OkStatus();
}

Status CreateTensorBHWCFromOpenGlObject(const CLContext& context,
                                        cl_int ssbo_id, const HWC& shape,
                                        bool is_readonly, TensorBHWC* tensor) {
  cl_int error_code;
  auto cl_buffer = clCreateFromGLBuffer(
      context.context(), is_readonly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE,
      ssbo_id, &error_code);
  if (error_code != CL_SUCCESS) {
    return ResourceExhaustedError(
        absl::StrCat("Unable to create CL buffer from GL buffer.",
                     CLErrorCodeToString(error_code)));
  }
  *tensor = TensorBHWC(cl_buffer, shape.w, shape.h, shape.c, DataType::FLOAT32,
                       TensorStorageType::BUFFER);
  tensor->owner_ = false;
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
