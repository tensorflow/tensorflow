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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_SCOPED_PROFILING_LABEL_WRAPPER_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_SCOPED_PROFILING_LABEL_WRAPPER_H_

// We are using TF_LITE_STATIC_MEMORY to inform if we are building for micro or
// not.  This is set for all micro builds (host and target) via the Makefile but
// not so for a bazel build.
//
// TODO(b/142948705): Ideally we would have a micro-specific bazel build too.
//
// We need to additionally check for ARDUINO because library specific defines
// are not supported by the Aruino IDE. See b/145161069 for more details.

#if defined(TF_LITE_STATIC_MEMORY) || defined(ARDUINO)

namespace tflite {
class ScopedProfilingLabelWrapper {
 public:
  explicit ScopedProfilingLabelWrapper(const char* label) {}
};
}  // namespace tflite

#else

#include "profiling/instrumentation.h"

namespace tflite {
class ScopedProfilingLabelWrapper {
 public:
  explicit ScopedProfilingLabelWrapper(const char* label)
      : scoped_profiling_label_(label) {}

 private:
  gemmlowp::ScopedProfilingLabel scoped_profiling_label_;
};
}  // namespace tflite

#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_SCOPED_PROFILING_LABEL_WRAPPER_H_
