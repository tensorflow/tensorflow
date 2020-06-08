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

#include "tensorflow/lite/delegates/gpu/cl/kernels/flt_type.h"

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {
namespace cl {

FLT::FLT(CalculationsPrecision precision, float value)
    : f32_(precision == CalculationsPrecision::F32), active_(true) {
  if (f32_) {
    f_value_ = value;
  } else {
    h_value_ = half(value);
  }
}

const void* FLT::GetData() const {
  return f32_ ? static_cast<const void*>(&f_value_)
              : static_cast<const void*>(&h_value_);
}

std::string FLT::GetDeclaration() const {
  const std::string type = f32_ ? "float" : "half";
  return absl::StrCat(type, " ", name_);
}

FLT2::FLT2(CalculationsPrecision precision, const float2& value)
    : f32_(precision == CalculationsPrecision::F32), active_(true) {
  if (f32_) {
    f_value_ = value;
  } else {
    h_value_ = half2(value);
  }
}

const void* FLT2::GetData() const {
  return f32_ ? static_cast<const void*>(&f_value_)
              : static_cast<const void*>(&h_value_);
}

std::string FLT2::GetDeclaration() const {
  const std::string type = f32_ ? "float2" : "half2";
  return absl::StrCat(type, " ", name_);
}

FLT4::FLT4(CalculationsPrecision precision, const float4& value)
    : f32_(precision == CalculationsPrecision::F32), active_(true) {
  if (f32_) {
    f_value_ = value;
  } else {
    h_value_ = half4(value);
  }
}

const void* FLT4::GetData() const {
  return f32_ ? static_cast<const void*>(&f_value_)
              : static_cast<const void*>(&h_value_);
}

std::string FLT4::GetDeclaration() const {
  const std::string type = f32_ ? "float4" : "half4";
  return absl::StrCat(type, " ", name_);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
