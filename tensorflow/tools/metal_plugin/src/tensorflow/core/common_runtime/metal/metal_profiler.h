/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_PROFILER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_PROFILER_H_

#include <cstdint>
#include <string>

#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace metal {

// The name of the op whose kernel is running on this thread, for labelling the
// command buffers it submits.
//
// A kernel does not otherwise know what it is called: the label has to come
// from the op kernel context, and the only place every kernel passes one is
// where it asks for its stream. Setting it there covers the whole backend
// without touching seventy kernel files.
//
// Empty when nothing is running a kernel, which is the case for the copies and
// fills the runtime issues on its own.
void SetCurrentOpName(const char* name, size_t length);
const std::string& CurrentOpName();

// Whether a profiling session is collecting. Checked before every command
// buffer completion does any work, so that the cost when no one is profiling
// is one relaxed atomic load.
bool ProfilingActive();

// Records one finished command buffer. `start` and `end` are Metal's GPU
// timestamps, in the seconds of CACurrentMediaTime's timebase; a buffer that
// reports zero for either never ran on the GPU and is dropped.
void RecordCommandBuffer(const std::string& label, double start, double end);

// Fills in the pluggable profiler C API. Exported as TF_InitProfiler.
void MetalInitProfiler(TF_ProfilerRegistrationParams* params,
                       TF_Status* status);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_PROFILER_H_
