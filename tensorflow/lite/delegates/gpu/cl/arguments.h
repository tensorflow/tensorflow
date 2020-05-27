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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_ARGUMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_ARGUMENTS_H_

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

class Arguments {
 public:
  Arguments() = default;
  void AddFloat(const std::string& name, float value = 0.0f);
  void AddInt(const std::string& name, int value = 0);

  absl::Status SetInt(const std::string& name, int value);
  absl::Status SetFloat(const std::string& name, float value);

  std::string GetListOfArgs();

  absl::Status Bind(cl_kernel kernel, int offset);

  void ResolveArgsPass(std::string* code);

  // Move only
  Arguments(Arguments&& args);
  Arguments& operator=(Arguments&& args);
  Arguments(const Arguments&) = delete;
  Arguments& operator=(const Arguments&) = delete;

 private:
  std::string AddActiveArgument(const std::string& arg_name);

  struct IntValue {
    int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, IntValue> int_values_;
  std::vector<int32_t> shared_int4s_data_;

  struct FloatValue {
    float value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<float> shared_float4s_data_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_ARGUMENTS_H_
