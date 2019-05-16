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

#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_egl.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

const char* ErrorToString(GLenum error) {
  switch (error) {
    case GL_INVALID_ENUM:
      return "[GL_INVALID_ENUM]: An unacceptable value is specified for an "
             "enumerated argument.";
    case GL_INVALID_VALUE:
      return "[GL_INVALID_VALUE]: A numeric argument is out of range.";
    case GL_INVALID_OPERATION:
      return "[GL_INVALID_OPERATION]: The specified operation is not allowed "
             "in the current state.";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "[GL_INVALID_FRAMEBUFFER_OPERATION]: The framebuffer object is "
             "not complete.";
    case GL_OUT_OF_MEMORY:
      return "[GL_OUT_OF_MEMORY]: There is not enough memory left to execute "
             "the command.";
  }
  return "[UNKNOWN_GL_ERROR]";
}

struct ErrorFormatter {
  void operator()(std::string* out, GLenum error) const {
    absl::StrAppend(out, ErrorToString(error));
  }
};

}  // namespace

// TODO(akulik): create new error space for GL error.

Status GetOpenGlErrors() {
  auto error = glGetError();
  if (error == GL_NO_ERROR) {
    return OkStatus();
  }
  auto error2 = glGetError();
  if (error2 == GL_NO_ERROR) {
    return InternalError(ErrorToString(error));
  }
  std::vector<GLenum> errors = {error, error2};
  for (error = glGetError(); error != GL_NO_ERROR; error = glGetError()) {
    errors.push_back(error);
  }
  return InternalError(absl::StrJoin(errors, ",", ErrorFormatter()));
}

Status GetEglError() {
  EGLint error = eglGetError();
  switch (error) {
    case EGL_SUCCESS:
      return OkStatus();
    case EGL_NOT_INITIALIZED:
      return InternalError(
          "EGL is not initialized, or could not be initialized, for the "
          "specified EGL display connection.");
    case EGL_BAD_ACCESS:
      return InternalError(
          "EGL cannot access a requested resource (for example a context is "
          "bound in another thread).");
    case EGL_BAD_ALLOC:
      return InternalError(
          "EGL failed to allocate resources for the requested operation.");
    case EGL_BAD_ATTRIBUTE:
      return InternalError(
          "An unrecognized attribute or attribute value was passed in the "
          "attribute list.");
    case EGL_BAD_CONTEXT:
      return InternalError(
          "An EGLContext argument does not name a valid EGL rendering "
          "context.");
    case EGL_BAD_CONFIG:
      return InternalError(
          "An EGLConfig argument does not name a valid EGL frame buffer "
          "configuration.");
    case EGL_BAD_CURRENT_SURFACE:
      return InternalError(
          "The current surface of the calling thread is a window, pixel buffer "
          "or pixmap that is no longer valid.");
    case EGL_BAD_DISPLAY:
      return InternalError(
          "An EGLDisplay argument does not name a valid EGL display "
          "connection.");
    case EGL_BAD_SURFACE:
      return InternalError(
          "An EGLSurface argument does not name a valid surface (window, pixel "
          "buffer or pixmap) configured for GL rendering.");
    case EGL_BAD_MATCH:
      return InternalError(
          "Arguments are inconsistent (for example, a valid context requires "
          "buffers not supplied by a valid surface).");
    case EGL_BAD_PARAMETER:
      return InternalError("One or more argument values are invalid.");
    case EGL_BAD_NATIVE_PIXMAP:
      return InternalError(
          "A NativePixmapType argument does not refer to a valid native "
          "pixmap.");
    case EGL_BAD_NATIVE_WINDOW:
      return InternalError(
          "A NativeWindowType argument does not refer to a valid native "
          "window.");
    case EGL_CONTEXT_LOST:
      return InternalError(
          "A power management event has occurred. The application must destroy "
          "all contexts and reinitialise OpenGL ES state and objects to "
          "continue rendering.");
  }
  return UnknownError("EGL error: " + std::to_string(error));
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
