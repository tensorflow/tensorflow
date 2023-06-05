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

#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_ROCM_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_ROCM_TRACER_H_

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/backends/profiler/gpu/rocm_tracer.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/roctracer_wrapper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

using xla::profiler::AnnotationMap;                 // NOLINT
using xla::profiler::DumpRocmTracerEvent;           // NOLINT
using xla::profiler::GetRocmTracerEventDomainName;  // NOLINT
using xla::profiler::GetRocmTracerEventSourceName;  // NOLINT
using xla::profiler::GetRocmTracerEventTypeName;    // NOLINT
using xla::profiler::KernelDetails;                 // NOLINT
using xla::profiler::MemAllocDetails;               // NOLINT
using xla::profiler::MemcpyDetails;                 // NOLINT
using xla::profiler::MemsetDetails;                 // NOLINT
using xla::profiler::RocmActivityCallbackImpl;      // NOLINT
using xla::profiler::RocmApiCallbackImpl;           // NOLINT
using xla::profiler::RocmTraceCollector;            // NOLINT
using xla::profiler::RocmTraceCollectorOptions;     // NOLINT
using xla::profiler::RocmTracer;                    // NOLINT
using xla::profiler::RocmTracerEvent;               // NOLINT
using xla::profiler::RocmTracerEventDomain;         // NOLINT
using xla::profiler::RocmTracerEventSource;         // NOLINT
using xla::profiler::RocmTracerEventType;           // NOLINT
using xla::profiler::RocmTracerOptions;             // NOLINT
using xla::profiler::RocmTracerSyncTypes;           // NOLINT
using xla::profiler::SynchronizationDetails;        // NOLINT

}  // namespace profiler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_ROCM_TRACER_H_
