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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_ARGUMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_ARGUMENTS_H_

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base_generated.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
class CLArguments;
}

namespace metal {
class MetalArguments;
}

class ArgumentsBinder {
 public:
  virtual absl::Status SetInt(const std::string& name, int value) = 0;
  virtual absl::Status SetFloat(const std::string& name, float value) = 0;
  virtual absl::Status SetHalf(const std::string& name, half value) = 0;
  virtual ~ArgumentsBinder() = default;
};

class Arguments {
 public:
  Arguments() = default;
  ~Arguments() = default;

  // Move only
  Arguments(Arguments&& args) = default;
  Arguments& operator=(Arguments&& args) = default;
  Arguments(const Arguments&) = delete;
  Arguments& operator=(const Arguments&) = delete;

  void AddFloat(const std::string& name, float value = 0.0f);
  void AddHalf(const std::string& name, half value = half(0.0f));
  void AddInt(const std::string& name, int value = 0);
  void AddObjectRef(const std::string& name, AccessType access_type,
                    GPUObjectDescriptorPtr&& descriptor_ptr);
  void AddObject(const std::string& name,
                 GPUObjectDescriptorPtr&& descriptor_ptr);

  void RenameArgs(const std::string& postfix, std::string* code) const;
  absl::Status Merge(Arguments&& args, const std::string& postfix);

  void ReleaseCPURepresentation();

 private:
  friend flatbuffers::Offset<tflite::gpu::data::Arguments> Encode(
      const Arguments& args, flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const tflite::gpu::data::Arguments* fb_args,
                             Arguments* args);

  friend class cl::CLArguments;
  friend class metal::MetalArguments;
  void GetActiveArguments(const std::string& args_prefix,
                          const std::string& code);

  struct IntValue {
    int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  std::map<std::string, IntValue> int_values_;

  struct FloatValue {
    float value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  std::map<std::string, FloatValue> float_values_;

  struct HalfValue {
    half value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  std::map<std::string, HalfValue> half_values_;

  std::map<std::string, GPUObjectDescriptorPtr> object_refs_;
  std::map<std::string, GPUObjectDescriptorPtr> objects_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_ARGUMENTS_H_
