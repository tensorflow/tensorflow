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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_CONVERTERS_BHWC_TO_PHWC4_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_CONVERTERS_BHWC_TO_PHWC4_H_

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"

namespace tflite {
namespace gpu {
namespace gl {

class ConverterBhwcToPhwc4 {
 public:
  // Creates invalid object.
  ConverterBhwcToPhwc4() : program_(), workgroup_size_() {}

  static absl::Status Create(ConverterBhwcToPhwc4* converter);

  absl::Status Convert(const BHWC& shape, const GlBuffer& source,
                       CommandQueue* command_queue /* optional */,
                       GlBuffer* destination);

 private:
  explicit ConverterBhwcToPhwc4(GlProgram program, const uint3& workgroup_size)
      : program_(std::move(program)), workgroup_size_(workgroup_size) {}

  GlProgram program_;
  uint3 workgroup_size_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_CONVERTERS_BHWC_TO_PHWC4_H_
