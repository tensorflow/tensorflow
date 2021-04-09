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

#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                      const BHWDC& shape,
                                      const TensorDescriptor& descriptor) {
  const int slices = DivideRoundUp(shape.c, 4);
  const uint64_t flt_size = descriptor.data_type == DataType::FLOAT32 ? 4 : 2;
  const uint64_t channels =
      descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D
          ? shape.c
          : slices * 4;
  const uint64_t allocation_size =
      flt_size * channels * shape.b * shape.w * shape.h * shape.d;
  const std::string common_desc = "Shape - " + ToString(shape) +
                                  ", data type - " +
                                  ToString(descriptor.data_type) + ".";
  if (allocation_size > gpu_info.GetMaxMemoryAllocationSize()) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Requested allocation size - ", allocation_size,
        " bytes. Max allocation size for this GPU - ",
        gpu_info.GetMaxMemoryAllocationSize(), " bytes. ", common_desc));
  }
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER: {
      const uint64_t flt4_size =
          4 * (descriptor.data_type == DataType::FLOAT32 ? 4 : 2);
      const uint64_t buffer_size =
          flt4_size * shape.b * shape.w * shape.h * shape.d * slices;
      if (buffer_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", buffer_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::IMAGE_BUFFER: {
      const uint64_t flt4_size =
          4 * (descriptor.data_type == DataType::FLOAT32 ? 4 : 2);
      const uint64_t buffer_size =
          flt4_size * shape.b * shape.w * shape.h * shape.d * slices;
      const uint64_t image_width = buffer_size / flt4_size;
      if (image_width > gpu_info.GetMaxImageBufferWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image buffer with width - ", image_width,
            " can not be created. Max image buffer width for this GPU - ",
            gpu_info.GetMaxImageBufferWidth(), ". ", common_desc));
      } else if (buffer_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", buffer_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_3D: {
      if (gpu_info.opencl_info.cl_version < OpenClVersion::kCl1_2 &&
          slices == 1) {
        return absl::InternalError(
            "clCreateImage3D (that used in CL 1.0/1.1) can not create image "
            "with depth = 1 by specification.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_depth = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage3DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with width - ", image_width,
            " can not be created. Max Image3D width for this GPU - ",
            gpu_info.GetMaxImage3DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage3DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with height - ", image_height,
            " can not be created. Max Image3D height for this GPU - ",
            gpu_info.GetMaxImage3DHeight(), ". ", common_desc));
      } else if (image_depth > gpu_info.GetMaxImage3DDepth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with depth - ", image_depth,
            " can not be created. Max Image3D depth for this GPU - ",
            gpu_info.GetMaxImage3DDepth(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      // Bug on some Adreno. b/131099086
      if (slices == 1 && gpu_info.IsAdreno() &&
          !gpu_info.adreno_info.support_one_layer_texture_array) {
        return absl::InternalError(
            "Image2DArray with layer = 1 works incorrect on some Adreno. Can "
            "not be created.");
      }
      const int image_width = shape.w * shape.b;
      const int image_height = shape.h;
      const int image_layers = slices * shape.d;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with width - ", image_width,
            " can not be created. Max Image2DArray width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with height - ", image_height,
            " can not be created. Max Image2DArray height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else if (image_layers > gpu_info.GetMaxImage2DArrayLayers()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with layers - ", image_layers,
            " can not be created. Max Image2DArray layers for this GPU - ",
            gpu_info.GetMaxImage2DArrayLayers(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h * slices;
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      const int image_width = shape.w * shape.b * shape.d;
      const int image_height = shape.h;
      if (shape.c > 4) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with channels - ", shape.c, " can not be created."));
      } else if (!gpu_info.SupportsFloatImage2D(descriptor.data_type,
                                                shape.c)) {
        return absl::ResourceExhaustedError(
            "Image2D doesn't support this pixel layout.");
      } else if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    default:
      return absl::UnimplementedError(
          "Can not create resources for unknown storage type.");
  }
}

absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                      const BHWC& shape,
                                      const TensorDescriptor& descriptor) {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CanCreateTensorWithShape(gpu_info, shape5D, descriptor);
}

absl::Status SelectBestStorageType(const GpuInfo& gpu_info, const BHWC& shape,
                                   TensorStorageType desired,
                                   DataType data_type, Layout layout,
                                   TensorStorageType* result) {
  if (CanCreateTensorWithShape(gpu_info, shape,
                               TensorDescriptor{data_type, desired, layout})
          .ok()) {
    *result = desired;
    return absl::OkStatus();
  }
  if (gpu_info.IsApiMetal()) {
    *result = TensorStorageType::BUFFER;
    return CanCreateTensorWithShape(
        gpu_info, shape,
        TensorDescriptor{data_type, TensorStorageType::BUFFER, layout});
  }
  auto GetBestTypeAfterTextureArray = [&]() {
    if (gpu_info.SupportsImageBuffer() &&
        CanCreateTensorWithShape(
            gpu_info, shape,
            TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER,
                             layout})
            .ok()) {
      *result = TensorStorageType::IMAGE_BUFFER;
      return absl::OkStatus();
    } else {
      *result = TensorStorageType::BUFFER;
      return CanCreateTensorWithShape(
          gpu_info, shape,
          TensorDescriptor{data_type, TensorStorageType::BUFFER, layout});
    }
  };
  auto GetBestTypeAfterTexture2D = [&]() {
    if (gpu_info.SupportsTextureArray() &&
        CanCreateTensorWithShape(
            gpu_info, shape,
            TensorDescriptor{data_type, TensorStorageType::TEXTURE_ARRAY,
                             layout})
            .ok()) {
      *result = TensorStorageType::IMAGE_BUFFER;
      return absl::OkStatus();
    } else {
      return GetBestTypeAfterTextureArray();
    }
  };
  auto GetBestTypeAfterTexture3D = [&]() {
    if (CanCreateTensorWithShape(
            gpu_info, shape,
            TensorDescriptor{data_type, TensorStorageType::TEXTURE_2D, layout})
            .ok()) {
      *result = TensorStorageType::TEXTURE_2D;
      return absl::OkStatus();
    } else {
      return GetBestTypeAfterTexture2D();
    }
  };
  switch (desired) {
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return GetBestTypeAfterTexture2D();
    case TensorStorageType::TEXTURE_ARRAY:
      return GetBestTypeAfterTextureArray();
    case TensorStorageType::TEXTURE_3D:
      return GetBestTypeAfterTexture3D();
    case TensorStorageType::IMAGE_BUFFER: {
      if (CanCreateTensorWithShape(
              gpu_info, shape,
              TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER,
                               layout})
              .ok()) {
        *result = TensorStorageType::IMAGE_BUFFER;
        return absl::OkStatus();
      } else {
        *result = TensorStorageType::BUFFER;
        return CanCreateTensorWithShape(
            gpu_info, shape,
            TensorDescriptor{data_type, TensorStorageType::BUFFER, layout});
      }
    }
    case TensorStorageType::BUFFER: {
      *result = TensorStorageType::BUFFER;
      return CanCreateTensorWithShape(
          gpu_info, shape,
          TensorDescriptor{data_type, TensorStorageType::BUFFER, layout});
    }
    default:
      return absl::UnimplementedError(absl::StrCat(
          "No support of this storage type - ", ToString(desired)));
  }
}

LinearStorageType DeduceLinearStorageType(
    TensorStorageType tensor_storage_type) {
  if (tensor_storage_type == TensorStorageType::BUFFER) {
    return LinearStorageType::BUFFER;
  } else {
    return LinearStorageType::TEXTURE_2D;
  }
}

}  // namespace gpu
}  // namespace tflite
