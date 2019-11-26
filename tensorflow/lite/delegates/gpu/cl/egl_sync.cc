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

#include "tensorflow/lite/delegates/gpu/cl/egl_sync.h"

#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"

namespace tflite {
namespace gpu {
namespace cl {

Status EglSync::NewFence(EGLDisplay display, EglSync* sync) {
  EGLSyncKHR egl_sync;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(eglCreateSyncKHR, &egl_sync, display,
                                      EGL_SYNC_FENCE_KHR, nullptr));
  if (egl_sync == EGL_NO_SYNC_KHR) {
    return InternalError("Returned empty KHR EGL sync");
  }
  *sync = EglSync(display, egl_sync);
  return OkStatus();
}

EglSync& EglSync::operator=(EglSync&& sync) {
  if (this != &sync) {
    Invalidate();
    std::swap(sync_, sync.sync_);
    display_ = sync.display_;
  }
  return *this;
}

void EglSync::Invalidate() {
  if (sync_ != EGL_NO_SYNC_KHR) {
    eglDestroySyncKHR(display_, sync_);
    sync_ = EGL_NO_SYNC_KHR;
  }
}

Status EglSync::ServerWait() {
  EGLint result;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(eglWaitSyncKHR, &result, display_, sync_, 0));
  return result == EGL_TRUE ? OkStatus() : InternalError("eglWaitSync failed");
}

Status EglSync::ClientWait() {
  EGLint result;
  // TODO(akulik): make it active wait for better performance
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(eglClientWaitSyncKHR, &result, display_,
                                      sync_, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR,
                                      EGL_FOREVER_KHR));
  return result == EGL_CONDITION_SATISFIED_KHR
             ? OkStatus()
             : InternalError("eglClientWaitSync failed");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
