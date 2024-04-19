/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TRACE_CONTAINER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TRACE_CONTAINER_H_

#include "tensorflow/core/profiler/convert/trace_viewer/trace_events.h"
#include "tensorflow/core/profiler/protobuf/trace_events_raw.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

using TraceEventsContainer = TraceEventsContainerBase<EventFactory, RawData>;

// Converts XEvents within the XSpace into trace_viewer events container.
void ConvertXSpaceToTraceEventsContainer(absl::string_view hostname,
                                         const XSpace& xspace,
                                         TraceEventsContainer* container);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TRACE_CONTAINER_H_
