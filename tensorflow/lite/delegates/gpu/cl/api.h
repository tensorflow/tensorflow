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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_API_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_API_H_

#include <cstdint>
#include <memory>

#include <EGL/egl.h>
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

// Usage example:
//
//   std::unique_ptr<InferenceEnvironment> env;
//   RETURN_IF_ERROR(NewInferenceEnvironment(option, &env));
//
//   InferenceOptions options;
//
//   std::unique_ptr<InferenceBuilder> builder;
//   RETURN_IF_ERROR(env->NewInferenceBuilder(options, model, &builder));
//   // now builder is ready to prepare inference runner.
//
// -----------------
// Supported formats
// -----------------
//
// OpenCL implementation uses 2D textures as the primary format.
// Tensor in HWDC4 layout is {TEXTURE_2D, RGBA, width := W*D, height := H}.
//

namespace tflite {
namespace gpu {
namespace cl {

struct InferenceOptions : public tflite::gpu::InferenceOptions {};

// Indicates environment
struct InferenceEnvironmentProperties {
  bool is_opencl_available = false;

  // GL objects (buffers and textures) could be shared with CL context.
  bool is_gl_sharing_supported = false;

  // Indicates whether fast GL->CL synchronization is supported.
  bool is_gl_to_cl_fast_sync_supported = false;

  // Indicates whether fast CL->GL synchronization is supported.
  bool is_cl_to_gl_fast_sync_supported = false;
};

// Environment manages all resources that need to stay until any inference is
// running using OpenCL backend.
class InferenceEnvironment {
 public:
  virtual ~InferenceEnvironment() {}

  virtual absl::Status NewInferenceBuilder(
      const InferenceOptions& options, GraphFloat32 model,
      std::unique_ptr<InferenceBuilder>* builder) = 0;

  // Returns opaque binary blob that contains a collection of already compiled
  // OpenCL kernels present in a cache. Returned data could be re-used later
  // to speed up compilation time when new environment is created for the same
  // set of models.
  // Returned data is valid only if used on the same device, otherwise it will
  // not be compatible and will be discarded.
  virtual std::vector<uint8_t> GetSerializedBinaryCache() const = 0;
};

struct InferenceEnvironmentOptions {
  // If any of these objects are set, created environment will use them instead
  // of creating/choosing own instances.
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue command_queue = nullptr;

  // Whenever input and/or output is GL object, EGL display and context must be
  // set to create GL aware OpenCL context. Do not set these variables whenever
  // GL interoperability is not needed.
  // It is the error to set egl_display, egl_context AND context at the same
  // time. If egl_display and egl_context are set, they will be used to create
  // GL-aware CL context.
  EGLDisplay egl_display = EGL_NO_DISPLAY;
  EGLContext egl_context = EGL_NO_CONTEXT;

  // Should contain data returned from
  // InferenceEnvironment::GetSerializedBinaryCache method.
  // Invalid or incompatible data will be discarded. Compiled binary may become
  // incompatible when GPU driver is updated.
  absl::Span<const uint8_t> serialized_binary_cache;

  bool IsGlAware() const {
    return egl_context != EGL_NO_CONTEXT && egl_display != EGL_NO_DISPLAY;
  }
};

// Creates new OpenCL environment that needs to stay around until all inference
// runners are destroyed.
absl::Status NewInferenceEnvironment(
    const InferenceEnvironmentOptions& options,
    std::unique_ptr<InferenceEnvironment>* environment,
    InferenceEnvironmentProperties* properties /* optional */);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_API_H_
