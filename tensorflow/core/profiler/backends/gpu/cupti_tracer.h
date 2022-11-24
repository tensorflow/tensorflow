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

#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_TRACER_H_

#include "absl/types/optional.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "tensorflow/compiler/xla/backends/profiler/gpu/cupti_tracer.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/backends/gpu/cupti_collector.h"
#include "tensorflow/core/profiler/backends/gpu/cupti_interface.h"
#include "tensorflow/tsl/profiler/utils/buffer_pool.h"

namespace tensorflow {
namespace profiler {

using xla::profiler::CuptiTracer;         // NOLINT
using xla::profiler::CuptiTracerOptions;  // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_TRACER_H_
