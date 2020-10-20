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
#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_ARGUMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_ARGUMENTS_H_

#import <Metal/Metal.h>

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/arguments.h"

namespace tflite {
namespace gpu {
namespace metal {

class MetalArguments : public ArgumentsSetter {
 public:
  MetalArguments() = default;

  absl::Status Init(int buffer_offset, Arguments* args, std::string* code);

  // Move only
  MetalArguments(MetalArguments&& args) = default;
  MetalArguments& operator=(MetalArguments&& args) = default;
  MetalArguments(const MetalArguments&) = delete;
  MetalArguments& operator=(const MetalArguments&) = delete;

  absl::Status SetInt(const std::string& name, int value) override;
  absl::Status SetFloat(const std::string& name, float value) override;

  void Encode(id<MTLComputeCommandEncoder> encoder, int buffer_offset) const;

 private:
  static constexpr char kArgsPrefix[] = "args.";
  struct IntValue {
    int value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t bytes_offset = -1;
  };
  std::map<std::string, IntValue> int_values_;

  struct FloatValue {
    float value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t bytes_offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<uint8_t> const_data_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_ARGUMENTS_H_
