// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/runtime/gpu/egl_context.h"

#include <EGL/egl.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

#define EGL_OPENGL_ES3_BIT_KHR 0x00000040

litert::Expected<EGLDisplay> GetDisplay() {
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (display == EGL_NO_DISPLAY) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to get EGL display");
  }

  EGLint major_version, minor_version;
  EGLBoolean egl_initialized =
      eglInitialize(display, &major_version, &minor_version);
  if (!egl_initialized) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to initialize EGL");
  }
  LITERT_LOG(LITERT_INFO,
             "Successfully initialized EGL. Major: ", major_version,
             " Minor: ", minor_version);

  return display;
}

litert::Expected<EGLContext> CreateContext(EGLDisplay display,
                                           EGLContext share_context) {
  int gl_version = 2;
#define OMIT_EGL_WINDOW_BIT
  const EGLint config_attr[] = {
      // clang-format off
      EGL_RENDERABLE_TYPE, gl_version == 3 ? EGL_OPENGL_ES3_BIT_KHR
                                           : EGL_OPENGL_ES2_BIT,
      // Allow rendering to pixel buffers or directly to windows.
      EGL_SURFACE_TYPE,
#ifdef OMIT_EGL_WINDOW_BIT
      EGL_PBUFFER_BIT,
#else
      EGL_PBUFFER_BIT | EGL_WINDOW_BIT,
#endif
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,  // if you need the alpha channel
      EGL_DEPTH_SIZE, 16,  // if you need the depth buffer
      EGL_NONE
  };

  EGLint num_configs;
  EGLConfig config;
  EGLBoolean success =
      eglChooseConfig(display, config_attr, &config, 1, &num_configs);
  if (!success){
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to choose EGL config");
  }
  if (!num_configs){
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "No EGL config found");
  }

  const EGLint context_attr[] = {
      EGL_CONTEXT_CLIENT_VERSION, gl_version,
      EGL_NONE
  };
  EGLContext context =
      eglCreateContext(display, config, share_context, context_attr);
  // int error = eglGetError();
  if (context == EGL_NO_CONTEXT) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create EGL context, error:");
  }
  return context;
}
