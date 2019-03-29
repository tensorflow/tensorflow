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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_SURFACE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_SURFACE_H_

#include <cstdint>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_egl.h"

namespace tflite {
namespace gpu {
namespace gl {

// An RAII wrapper for EGLSurface.
// See https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglIntro.xhtml for
// an introduction to the concepts.
//
// EglSurface is moveable but not copyable.
class EglSurface {
 public:
  // Creates an invalid EglSurface.
  EglSurface() : surface_(EGL_NO_SURFACE), display_(EGL_NO_DISPLAY) {}

  EglSurface(EGLSurface surface, EGLDisplay display)
      : surface_(surface), display_(display) {}

  // Move-only
  EglSurface(EglSurface&& other);
  EglSurface& operator=(EglSurface&& other);
  EglSurface(const EglSurface&) = delete;
  EglSurface& operator=(const EglSurface&) = delete;

  ~EglSurface() { Invalidate(); }

  EGLSurface surface() const { return surface_; }

 private:
  void Invalidate();

  EGLSurface surface_;
  EGLDisplay display_;
};

// Creates off-screen pbuffer-based surface of the given height and width.
Status CreatePbufferRGBSurface(EGLConfig config, EGLDisplay display,
                               uint32_t height, uint32_t width,
                               EglSurface* egl_surface);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_SURFACE_H_
