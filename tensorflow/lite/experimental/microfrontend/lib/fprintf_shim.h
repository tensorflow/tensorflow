/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FPRINTF_SHIM_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FPRINTF_SHIM_H_

// This code is shared between the TensorFlow training environment and
// the embedded Micro codebase. On Micro, many platforms don't support
// stdio, so we stub out the fprintf call so it does nothing. In the
// Bazel build files for the training ops, we enable this macro so that
// useful debug logging will still be output.
#ifdef MICROFRONTEND_USE_FPRINTF

#include <stdio.h>

// Redirect to the real fprintf.
#define MICROFRONTEND_FPRINTF fprintf

#else  // MICROFRONTEND_USE_FPRINTF

// Stub out calls to fprintf so they do nothing.
#define MICROFRONTEND_FPRINTF(stream, format, ...)

#endif  // MICROFRONTEND_USE_FPRINTF

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_MEMORY_UTIL_H_
