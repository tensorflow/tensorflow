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

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

absl::Status SelectBestStorageType(const GpuInfo& gpu_info, const BHWC& shape,
                                   TensorStorageType desired,
                                   DataType data_type, Layout layout,
                                   TensorStorageType* result) {
  if (TensorDescriptor{data_type, desired, layout}
          .CanCreateTensorWithShape(gpu_info, shape)
          .ok()) {
    *result = desired;
    return absl::OkStatus();
  }
  if (gpu_info.IsApiMetal()) {
    *result = TensorStorageType::BUFFER;
    return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
        .CanCreateTensorWithShape(gpu_info, shape);
  }
  auto GetBestTypeAfterTexture2D = [&]() {
    if (gpu_info.SupportsImageBuffer() &&
        TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok()) {
      *result = TensorStorageType::IMAGE_BUFFER;
      return absl::OkStatus();
    } else {
      *result = TensorStorageType::BUFFER;
      return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
          .CanCreateTensorWithShape(gpu_info, shape);
    }
  };
  auto GetBestTypeAfterTextureArray = [&]() {
    if (gpu_info.SupportsImageBuffer() &&
        TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok()) {
      *result = TensorStorageType::IMAGE_BUFFER;
      return absl::OkStatus();
    } else {
      *result = TensorStorageType::BUFFER;
      return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
          .CanCreateTensorWithShape(gpu_info, shape);
    }
  };
  auto GetBestTypeAfterTexture3D = [&]() {
    if (TensorDescriptor{data_type, TensorStorageType::TEXTURE_2D, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok()) {
      *result = TensorStorageType::TEXTURE_2D;
      return absl::OkStatus();
    } else {
      return GetBestTypeAfterTextureArray();
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
      if (TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER, layout}
              .CanCreateTensorWithShape(gpu_info, shape)
              .ok()) {
        *result = TensorStorageType::IMAGE_BUFFER;
        return absl::OkStatus();
      } else {
        *result = TensorStorageType::BUFFER;
        return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
            .CanCreateTensorWithShape(gpu_info, shape);
      }
    }
    case TensorStorageType::BUFFER: {
      *result = TensorStorageType::BUFFER;
      return TensorDescriptor{data_type, TensorStorageType::BUFFER, layout}
          .CanCreateTensorWithShape(gpu_info, shape);
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
