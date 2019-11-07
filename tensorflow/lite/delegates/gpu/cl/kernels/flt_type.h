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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FLT_TYPE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FLT_TYPE_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class FLT {
 public:
  FLT() = default;
  FLT(CalculationsPrecision precision, float value);

  const void* GetData() const;
  size_t GetSize() const { return f32_ ? sizeof(float) : sizeof(half); }
  bool Active() const { return active_; }
  std::string GetDeclaration() const;
  std::string GetName() const { return name_; }
  void SetName(const std::string& name) { name_ = name; }

 private:
  float f_value_;
  half h_value_;
  bool f32_;
  bool active_ = false;
  std::string name_;
};

class FLT2 {
 public:
  FLT2() = default;
  FLT2(CalculationsPrecision precision, const float2& value);

  const void* GetData() const;
  size_t GetSize() const { return f32_ ? 8 : 4; }
  bool Active() const { return active_; }
  std::string GetDeclaration() const;
  std::string GetName() const { return name_; }
  void SetName(const std::string& name) { name_ = name; }

 private:
  float2 f_value_;
  half2 h_value_;
  bool f32_;
  bool active_ = false;
  std::string name_;
};

class FLT4 {
 public:
  FLT4() {}
  FLT4(CalculationsPrecision precision, const float4& value);

  const void* GetData() const;
  size_t GetSize() const { return f32_ ? sizeof(float4) : sizeof(half4); }
  bool Active() const { return active_; }
  std::string GetDeclaration() const;
  std::string GetName() const { return name_; }
  void SetName(const std::string& name) { name_ = name; }

 private:
  float4 f_value_;
  half4 h_value_;
  bool f32_;
  bool active_ = false;
  std::string name_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FLT_TYPE_H_
