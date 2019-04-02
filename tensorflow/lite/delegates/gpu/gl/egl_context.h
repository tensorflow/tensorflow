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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_CONTEXT_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_egl.h"

namespace tflite {
namespace gpu {
namespace gl {

// EglContext is an RAII wrapper for an EGLContext.
//
// EglContext is moveable but not copyable.
//
// See https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglIntro.xhtml for
// more info.
class EglContext {
 public:
  // Creates an invalid EglContext.
  EglContext()
      : context_(EGL_NO_CONTEXT),
        display_(EGL_NO_DISPLAY),
        config_(EGL_NO_CONFIG_KHR) {}

  EglContext(EGLContext context, EGLDisplay display, EGLConfig config)
      : context_(context), display_(display), config_(config) {}

  // Move only
  EglContext(EglContext&& other);
  EglContext& operator=(EglContext&& other);
  EglContext(const EglContext&) = delete;
  EglContext& operator=(const EglContext&) = delete;

  ~EglContext() { Invalidate(); }

  EGLContext context() const { return context_; }

  EGLDisplay display() const { return display_; }

  EGLConfig config() const { return config_; }

  // Make this EglContext the current EGL context on this thread, replacing
  // the existing current.
  Status MakeCurrent(EGLSurface read, EGLSurface write);

  Status MakeCurrentSurfaceless() {
    return MakeCurrent(EGL_NO_SURFACE, EGL_NO_SURFACE);
  }

  // Returns true if this is the currently bound EGL context.
  bool IsCurrent() const;

 private:
  void Invalidate();

  EGLContext context_;
  EGLDisplay display_;
  EGLConfig config_;
};

// It uses the EGL_KHR_no_config_context extension to create a no config context
// since most modern hardware supports the extension.
Status CreateConfiglessContext(EGLDisplay display, EGLContext shared_context,
                               EglContext* egl_context);

Status CreateSurfacelessContext(EGLDisplay display, EGLContext shared_context,
                                EglContext* egl_context);

Status CreatePBufferContext(EGLDisplay display, EGLContext shared_context,
                            EglContext* egl_context);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_EGL_CONTEXT_H_
