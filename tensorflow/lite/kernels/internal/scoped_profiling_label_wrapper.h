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

// gemmlowp itself defines an empty class for ScopedProfilingLabel when
// GEMMLOWP_PROFILING is not defined. However, that does not work for embedded
// builds because instrumentation.h depends on pthread and defines a few Mutex
// classes independent of GEMMLOWP_PROFILING.
//
// As a result, we are using GEMMLOWP_PROFILING to either pull in the
// gemmlowp implementation or use our own empty class.
//
// The downside with this approach is that we are using a gemmlowp macro from
// the TFLite codebase. The upside is that it is much simpler than the
// alternatives (see history of this file).

#ifdef GEMMLOWP_PROFILING

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

#else  // GEMMLOWP_PROFILING

namespace tflite {
class ScopedProfilingLabelWrapper {
 public:
  explicit ScopedProfilingLabelWrapper(const char* label) {}
};
}  // namespace tflite

#endif  // GEMMLOWP_PROFILING

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_SCOPED_PROFILING_LABEL_WRAPPER_H_
