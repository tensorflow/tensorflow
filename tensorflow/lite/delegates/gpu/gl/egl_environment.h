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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_ENVIRONMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_ENVIRONMENT_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_context.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_surface.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_egl.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"

namespace tflite {
namespace gpu {
namespace gl {

// Class encapsulates creation of OpenGL objects needed before starting working
// with OpenGL: binds OpenGL ES API, creates new EGL context, binds it to EGL
// display and creates surfaces if needed.
//
// EGL environment needs to be created once per thread.
class EglEnvironment {
 public:
  static Status NewEglEnvironment(
      std::unique_ptr<EglEnvironment>* egl_environment);

  EglEnvironment() = default;
  ~EglEnvironment();

  const EglContext& context() const { return context_; }
  EGLDisplay display() const { return display_; }
  const GpuInfo& gpu_info() const { return gpu_info_; }

 private:
  Status Init();
  Status InitConfiglessContext();
  Status InitSurfacelessContext();
  Status InitPBufferContext();

  EGLDisplay display_ = EGL_NO_DISPLAY;
  EglSurface surface_draw_;
  EglSurface surface_read_;
  EglContext context_;
  GpuInfo gpu_info_;

  // Strange hack that helps on Mali GPUs
  // without it glFinish and glFenceSync don't work
  void ForceSyncTurning();
  GLuint dummy_framebuffer_ = GL_INVALID_INDEX;
  GLuint dummy_texture_ = GL_INVALID_INDEX;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_ENVIRONMENT_H_
