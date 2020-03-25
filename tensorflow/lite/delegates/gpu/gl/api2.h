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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_API2_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_API2_H_

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"

namespace tflite {
namespace gpu {
namespace gl {

struct InferenceOptions : public tflite::gpu::InferenceOptions {};

struct InferenceEnvironmentProperties {
  bool is_opengl_available = false;
};

// Manages all resources that need to stay around as long as any inference is
// running using the OpenGL backend.
class InferenceEnvironment {
 public:
  virtual ~InferenceEnvironment() = default;

  virtual absl::Status NewInferenceBuilder(
      GraphFloat32&& model, const InferenceOptions& options,
      std::unique_ptr<InferenceBuilder>* builder) = 0;
};

struct InferenceEnvironmentOptions {
  CommandQueue* queue = nullptr;
};

// Creates a new OpenGL environment that needs to stay around until all
// inference runners are destroyed.
absl::Status NewInferenceEnvironment(
    const InferenceEnvironmentOptions& options,
    std::unique_ptr<InferenceEnvironment>* environment,
    InferenceEnvironmentProperties* properties /* optional */);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_API2_H_
