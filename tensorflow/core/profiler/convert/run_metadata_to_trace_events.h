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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_RUN_METADATA_TO_TRACE_EVENTS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_RUN_METADATA_TO_TRACE_EVENTS_H_

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

void ConvertRunMetadataToTraceEvents(uint64 profile_start_time_ns,
                                     uint64 profile_end_time_ns,
                                     RunMetadata* run_metadata, Trace* trace);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_RUN_METADATA_TO_TRACE_EVENTS_H_
