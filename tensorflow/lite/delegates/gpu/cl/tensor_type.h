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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_TYPE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_TYPE_H_

#include <cstddef>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {
namespace cl {

enum class TensorStorageType {
  UNKNOWN,
  BUFFER,
  IMAGE_BUFFER,
  TEXTURE_2D,
  TEXTURE_3D,
  TEXTURE_ARRAY,
  SINGLE_TEXTURE_2D
};

struct TensorDescriptor {
  TensorDescriptor() = default;
  TensorDescriptor(DataType dt, TensorStorageType st, Layout l)
      : data_type(dt), storage_type(st), layout(l) {}

  bool operator==(const TensorDescriptor& d) const {
    return data_type == d.data_type && storage_type == d.storage_type &&
           layout == d.layout;
  }

  bool operator!=(const TensorDescriptor& d) const { return !(*this == d); }

  DataType data_type = DataType::UNKNOWN;
  TensorStorageType storage_type = TensorStorageType::UNKNOWN;
  // This field describes logical layout, actual(physical) GPU layout can be
  // totally different.
  Layout layout =
      Layout::UNKNOWN;  // Supported layouts is HWC, BHWC, HWDC, BHWDC
};

std::string ToString(TensorStorageType type);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_TYPE_H_
