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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GPU_EGL_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GPU_EGL_CONTEXT_H_

#include <EGL/egl.h>

#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

litert::Expected<EGLDisplay> GetDisplay();

litert::Expected<EGLContext> CreateContext(EGLDisplay display,
                                           EGLContext share_context);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GPU_EGL_CONTEXT_H_
