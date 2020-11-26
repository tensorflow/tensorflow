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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_DATA_TYPE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_DATA_TYPE_H_

#include <stddef.h>
#include <string>

namespace tflite {
namespace gpu {

enum class DataType {
  UNKNOWN = 0,
  FLOAT16 = 1,
  FLOAT32 = 2,
  FLOAT64 = 3,
  UINT8 = 4,
  INT8 = 5,
  UINT16 = 6,
  INT16 = 7,
  UINT32 = 8,
  INT32 = 9,
  UINT64 = 10,
  INT64 = 11,
};

size_t SizeOf(DataType type);

std::string ToString(DataType t);

std::string ToCLDataType(DataType data_type, int vec_size = 1);

std::string ToMetalDataType(DataType data_type, int vec_size = 1);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_DATA_TYPE_H_
