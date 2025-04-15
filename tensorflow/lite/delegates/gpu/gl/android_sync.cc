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
#include "tensorflow/lite/delegates/gpu/gl/android_sync.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>
#include <GLES2/gl2.h>
#include <unistd.h>

namespace {
PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFDANDROID;
PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR;
PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR;
PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;

bool IsGlSupported() {
  static const bool extensions_allowed = [] {
    eglDupNativeFenceFDANDROID =
        reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(
            eglGetProcAddress("eglDupNativeFenceFDANDROID"));
    eglCreateSyncKHR = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(
        eglGetProcAddress("eglCreateSyncKHR"));
    eglWaitSyncKHR = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(
        eglGetProcAddress("eglWaitSyncKHR"));
    eglDestroySyncKHR = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(
        eglGetProcAddress("eglDestroySyncKHR"));
    return eglWaitSyncKHR && eglCreateSyncKHR && eglDupNativeFenceFDANDROID &&
           eglDestroySyncKHR;
  }();
  return extensions_allowed;
}
}  // namespace

namespace tflite::gpu::gl {

// Insert a gpu wait sync to the queue; return true if successful.
bool WaitFdGpu(int fence_fd) {
  if (fence_fd == -1) {
    return false;
  }
  if (!IsGlSupported()) {
    return false;
  }

  // Server-side fence.
  EGLDisplay egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (egl_display == EGL_NO_DISPLAY) return false;

  // EGL will take ownership of the passed fd if eglCreateSyncKHR is
  // successful.
  int fd_for_egl = dup(fence_fd);

  EGLint sync_attribs[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, (EGLint)fd_for_egl,
                           EGL_NONE};
  EGLSync fence_sync = eglCreateSyncKHR(
      egl_display, EGL_SYNC_NATIVE_FENCE_ANDROID, sync_attribs);
  if (fence_sync != EGL_NO_SYNC_KHR) {
    eglWaitSyncKHR(egl_display, fence_sync, 0);
    return true;
  } else {
    close(fd_for_egl);
    return false;
  }
}

// Create a GL Fence object and return the associated fd
int CreateFdGpu() {
  if (IsGlSupported()) {
    EGLDisplay egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display != EGL_NO_DISPLAY) {
      EGLSync fence_sync =
          eglCreateSyncKHR(egl_display, EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
      if (fence_sync != EGL_NO_SYNC_KHR) {
        int fence_fd = eglDupNativeFenceFDANDROID(egl_display, fence_sync);
        if (fence_fd == -1) {
          eglDestroySyncKHR(egl_display, fence_sync);
        } else {
          return fence_fd;
        }
      }
    }
  }
  // Can't use Sync object. We use glFinish as CPU wait instead
  glFinish();
  return -1;
}

}  // namespace tflite::gpu::gl
