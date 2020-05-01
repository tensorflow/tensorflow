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

namespace {

PFNEGLCREATESYNCKHRPROC g_eglCreateSyncKHR = nullptr;
PFNEGLDESTROYSYNCKHRPROC g_eglDestroySyncKHR = nullptr;
PFNEGLWAITSYNCKHRPROC g_eglWaitSyncKHR = nullptr;
PFNEGLCLIENTWAITSYNCKHRPROC g_eglClientWaitSyncKHR = nullptr;

absl::Status IsEglSyncSupported(EGLDisplay display) {
  static bool supported = [display]() -> bool {
    // EGL_KHR_fence_sync is apparently a display extension
    const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
    if (!extensions) {
      return false;
    }
    if (std::strstr(extensions, "EGL_KHR_fence_sync")) {
      g_eglCreateSyncKHR = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(
          eglGetProcAddress("eglCreateSyncKHR"));
      g_eglDestroySyncKHR = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(
          eglGetProcAddress("eglDestroySyncKHR"));
      g_eglWaitSyncKHR = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(
          eglGetProcAddress("eglWaitSyncKHR"));
      g_eglClientWaitSyncKHR = reinterpret_cast<PFNEGLCLIENTWAITSYNCKHRPROC>(
          eglGetProcAddress("eglClientWaitSyncKHR"));
    }
    return g_eglCreateSyncKHR && g_eglDestroySyncKHR && g_eglWaitSyncKHR &&
           g_eglClientWaitSyncKHR;
  }();
  if (!supported) {
    return absl::InternalError("EGL_KHR_fence_sync unsupported");
  }
  return absl::OkStatus();
}

}  // anonymous namespace

absl::Status EglSync::NewFence(EGLDisplay display, EglSync* sync) {
  EGLSyncKHR egl_sync;
  RETURN_IF_ERROR(IsEglSyncSupported(display));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(g_eglCreateSyncKHR, &egl_sync, display,
                                      EGL_SYNC_FENCE_KHR, nullptr));
  if (egl_sync == EGL_NO_SYNC_KHR) {
    return absl::InternalError("Returned empty KHR EGL sync");
  }
  *sync = EglSync(display, egl_sync);
  return absl::OkStatus();
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
  if (!IsEglSyncSupported(display_).ok()) return;
  if (sync_ != EGL_NO_SYNC_KHR) {
    g_eglDestroySyncKHR(display_, sync_);
    sync_ = EGL_NO_SYNC_KHR;
  }
}

absl::Status EglSync::ServerWait() {
  EGLint result;
  RETURN_IF_ERROR(IsEglSyncSupported(display_));
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(g_eglWaitSyncKHR, &result, display_, sync_, 0));
  return result == EGL_TRUE ? absl::OkStatus()
                            : absl::InternalError("eglWaitSync failed");
}

absl::Status EglSync::ClientWait() {
  EGLint result;
  RETURN_IF_ERROR(IsEglSyncSupported(display_));
  // TODO(akulik): make it active wait for better performance
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(g_eglClientWaitSyncKHR, &result, display_,
                                      sync_, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR,
                                      EGL_FOREVER_KHR));
  return result == EGL_CONDITION_SATISFIED_KHR
             ? absl::OkStatus()
             : absl::InternalError("eglClientWaitSync failed");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
