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

#include "tensorflow/lite/delegates/gpu/gl/float16_conversions.h"

#include <cstdint>
#include <variant>
#include <vector>

#include "fp16.h"  // from @FP16
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Performs in-place conversion of float32 into float16
bool ToFloat16(std::vector<uint8_t>* values) {
  if (values->size() % sizeof(float) != 0) {
    return false;
  }

  uint16_t* store_f16 = reinterpret_cast<uint16_t*>(values->data());
  const float* load_f32 = reinterpret_cast<const float*>(values->data());
  const float* end_load_f32 =
      reinterpret_cast<const float*>(values->data() + values->size());

  while (load_f32 != end_load_f32) {
    *store_f16++ = fp16_ieee_from_fp32_value(*load_f32++);
  }

  values->resize(values->size() / 2);
  return true;
}

struct ConverterToFloat16 {
  bool operator()(ObjectData& data) const {  // NOLINT
    return ToFloat16(&data);
  }

  bool operator()(ObjectRef& buffer) const {  // NOLINT
    return true;
  }
};

}  // namespace

bool MaybeConvertToFloat16(Object* object) {
  if (object->data_type == DataType::FLOAT32 &&
      std::visit(ConverterToFloat16(), object->object)) {
    object->data_type = DataType::FLOAT16;
    return true;
  }
  return false;
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
