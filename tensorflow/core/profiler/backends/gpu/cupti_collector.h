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

#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_COLLECTOR_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_COLLECTOR_H_

#include <memory>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/backends/profiler/gpu/cupti_collector.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

using xla::profiler::AnnotationMap;                // NOLINT
using xla::profiler::CreateCuptiCollector;         // NOLINT
using xla::profiler::CuptiTraceCollector;          // NOLINT
using xla::profiler::CuptiTracerCollectorOptions;  // NOLINT
using xla::profiler::CuptiTracerEvent;             // NOLINT
using xla::profiler::CuptiTracerEventSource;       // NOLINT
using xla::profiler::CuptiTracerEventType;         // NOLINT
using xla::profiler::GetMemoryKindName;            // NOLINT
using xla::profiler::GetTraceEventTypeName;        // NOLINT
using xla::profiler::KernelDetails;                // NOLINT
using xla::profiler::MemAllocDetails;              // NOLINT
using xla::profiler::MemcpyDetails;                // NOLINT
using xla::profiler::MemsetDetails;                // NOLINT
using xla::profiler::ToXStat;                      // NOLINT

using MemFreeDetails = MemAllocDetails;
using MemoryResidencyDetails = MemAllocDetails;

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_COLLECTOR_H_
