/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_CPU_HOST_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_CPU_HOST_TRACER_H_

#include <memory>

#include "tensorflow/compiler/xla/backends/profiler/cpu/host_tracer.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"

namespace tensorflow {
namespace profiler {

using xla::profiler::HostTracerOptions;  // NOLINT

using xla::profiler::CreateHostTracer;  // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_CPU_HOST_TRACER_H_
