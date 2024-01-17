/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/async_buffers.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/android_hardware_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"

namespace {
PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID;

bool IsGlSupported() {
  static const bool extensions_allowed = [] {
    eglGetNativeClientBufferANDROID =
        reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    glBufferStorageExternalEXT =
        reinterpret_cast<PFNGLBUFFERSTORAGEEXTERNALEXTPROC>(
            eglGetProcAddress("glBufferStorageExternalEXT"));
    return eglGetNativeClientBufferANDROID && glBufferStorageExternalEXT;
  }();
  return extensions_allowed;
}
}  // namespace

namespace tflite {
namespace gpu {

// Where the AHWB<->SSBO mapping occurs
absl::Status AsyncBuffer::MapAHardwareBufferToGlBuffer() {
  if (!IsGlSupported()) {
    return absl::UnknownError(
        "No GL extension functions found to bind AHardwareBuffer and "
        "OpenGL buffer");
  }
  EGLClientBuffer native_buffer = eglGetNativeClientBufferANDROID(ahwb_);
  if (!native_buffer) {
    return absl::UnknownError("Can't get native buffer");
  }
  // If an error is traced back to below fcn, check your ahwb usage flags.
  // An example may be found in async_buffers_test.cc
  glBufferStorageExternalEXT(GL_SHADER_STORAGE_BUFFER, 0, bytes_, native_buffer,
                             GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                                 GL_MAP_COHERENT_BIT_EXT |
                                 GL_MAP_PERSISTENT_BIT_EXT);
  return gl::GetOpenGlErrors();
}

// Allocate SSBO, call the AHWB<->SSBO mapping, and fail gracefully if needed.
absl::Status AsyncBuffer::AllocateOpenGlBuffer() {
  if (opengl_buffer_ == GL_INVALID_INDEX) {
    // Generate and bind SSBO
    glGenBuffers(1, &opengl_buffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
    absl::Status status = MapAHardwareBufferToGlBuffer();
    if (!status.ok()) {
      // If we can't map to SSBO, clear AHWB & SSBO
      if (ahwb_ != nullptr) {
        if (OptionalAndroidHardwareBuffer::Instance().Supported()) {
          OptionalAndroidHardwareBuffer::Instance().Release(ahwb_);
        }
        ahwb_ = nullptr;
      }
      glBufferData(GL_SHADER_STORAGE_BUFFER, bytes_, nullptr, GL_STREAM_COPY);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  }
  return absl::OkStatus();
}

// Public function which will map the AHWB (from class constructor) to a SSBO
// and return the associated the id by reference
absl::Status AsyncBuffer::GetOpenGlBuffer(GLuint& buffer_ref) {
  if (!valid_) {
    absl::Status status = AllocateOpenGlBuffer();
    if (!status.ok()) {
      return status;
    }
  }
  valid_ = true;
  buffer_ref = opengl_buffer_;
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
