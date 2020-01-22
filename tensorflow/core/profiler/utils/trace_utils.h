/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TRACE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TRACE_UTILS_H_

namespace tensorflow {
namespace profiler {

// The thread id used for step information in GPU trace viewer.
// First derived stream/thread id.
constexpr int kThreadIdDerivedMin = 0xdeadbeef;
constexpr int kThreadIdStepInfo = kThreadIdDerivedMin;
constexpr int kThreadIdTfOp = kThreadIdDerivedMin + 1;
constexpr int kThreadIdHloOp = kThreadIdDerivedMin + 2;
constexpr int kThreadIdOverhead = kThreadIdDerivedMin + 3;
constexpr int kThreadIdHloModule = kThreadIdDerivedMin + 4;
// Last derived stream/thread id.
constexpr int kThreadIdDerivedMax = kThreadIdHloModule;

static inline bool IsDerivedThreadId(int thread_id) {
  return thread_id >= kThreadIdDerivedMin && thread_id <= kThreadIdDerivedMax;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TRACE_UTILS_H_
