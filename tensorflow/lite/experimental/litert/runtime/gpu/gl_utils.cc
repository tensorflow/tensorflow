// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/gpu/gl_utils.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl32.h>

#include <cstddef>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"

namespace litert {

namespace {

PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID;
PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFDANDROID;
PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR;
PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR;
PFNEGLCLIENTWAITSYNCKHRPROC eglClientWaitSyncKHR;
PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;

bool IsGlSupported() {
  static const bool extensions_allowed = [] {
    eglGetNativeClientBufferANDROID =
        reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    glBufferStorageExternalEXT =
        reinterpret_cast<PFNGLBUFFERSTORAGEEXTERNALEXTPROC>(
            eglGetProcAddress("glBufferStorageExternalEXT"));
    eglDupNativeFenceFDANDROID =
        reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(
            eglGetProcAddress("eglDupNativeFenceFDANDROID"));
    eglCreateSyncKHR = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(
        eglGetProcAddress("eglCreateSyncKHR"));
    eglWaitSyncKHR = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(
        eglGetProcAddress("eglWaitSyncKHR"));
    eglClientWaitSyncKHR = reinterpret_cast<PFNEGLCLIENTWAITSYNCKHRPROC>(
        eglGetProcAddress("eglClientWaitSyncKHR"));
    eglDestroySyncKHR = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(
        eglGetProcAddress("eglDestroySyncKHR"));
    return eglClientWaitSyncKHR && eglWaitSyncKHR &&
           eglGetNativeClientBufferANDROID && glBufferStorageExternalEXT &&
           eglCreateSyncKHR && eglDupNativeFenceFDANDROID && eglDestroySyncKHR;
  }();
  return extensions_allowed;
}

}  // namespace

LiteRtStatus AllocateSsbo(GLuint* gl_buffer, size_t bytes) {
  glGenBuffers(1, gl_buffer);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gl_buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_STREAM_COPY);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  if (glGetError() != GL_NO_ERROR) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus MapAHardwareBufferToGlBuffer(AHardwareBuffer* handle,
                                          size_t bytes) {
  if (!IsGlSupported()) {
    LITERT_LOG(LITERT_ERROR,
               "No GL extension functions found to bind AHardwareBuffer and "
               "OpenGL buffer",
               "");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Create EGLClientBuffer from AHardwareBuffer.
  EGLClientBuffer native_buffer = eglGetNativeClientBufferANDROID(handle);

  if (native_buffer == nullptr) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to get native client buffer from AHardwareBuffer", "");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Create OpenGl buffer object backed by eternal memory, i.e. AHardwareBuffer.
  glBufferStorageExternalEXT(GL_SHADER_STORAGE_BUFFER, 0, bytes, native_buffer,
                             GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                                 GL_MAP_COHERENT_BIT_EXT |
                                 GL_MAP_PERSISTENT_BIT_EXT);
  GLenum gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Error in glBufferStorageExternalEXT", "");
    LITERT_LOG(LITERT_ERROR, "Error code: %d", gl_error);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus AllocateSsboBackedByAhwb(GLuint* gl_buffer, size_t bytes,
                                      AHardwareBuffer* ahwb) {
  glGenBuffers(1, gl_buffer);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gl_buffer);
  // Map the AHWB to the GL buffer.
  if (MapAHardwareBufferToGlBuffer(ahwb, bytes) != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to map AHWB to GL buffer", "");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  return kLiteRtStatusOk;
}

}  // namespace litert
