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

#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"

namespace tflite {
namespace gpu {
namespace cl {

ObjectType ToObjectType(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::BUFFER:
      return ObjectType::OPENCL_BUFFER;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return ObjectType::OPENCL_TEXTURE;
    default:
      return ObjectType::UNKNOWN;
  }
}

DataLayout ToDataLayout(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::BUFFER:
      return DataLayout::DHWC4;
    case TensorStorageType::IMAGE_BUFFER:
      return DataLayout::DHWC4;
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return DataLayout::BHWC;
    case TensorStorageType::TEXTURE_2D:
      return DataLayout::HDWC4;
    case TensorStorageType::TEXTURE_ARRAY:
      return DataLayout::DHWC4;
    default:
      return DataLayout::UNKNOWN;
  }
}

TensorStorageType ToTensorStorageType(ObjectType object_type,
                                      DataLayout data_layout) {
  switch (object_type) {
    case ObjectType::OPENCL_BUFFER:
      return TensorStorageType::BUFFER;
    case ObjectType::OPENCL_TEXTURE:
      switch (data_layout) {
        case DataLayout::BHWC:
          return TensorStorageType::SINGLE_TEXTURE_2D;
        case DataLayout::DHWC4:
          return TensorStorageType::TEXTURE_ARRAY;
        case DataLayout::HDWC4:
          return TensorStorageType::TEXTURE_2D;
        default:
          return TensorStorageType::UNKNOWN;
      }
    default:
      return TensorStorageType::UNKNOWN;
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
