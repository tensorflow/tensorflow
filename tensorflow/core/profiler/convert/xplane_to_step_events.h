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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_STEP_EVENTS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_STEP_EVENTS_H_

#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// Convert the host threads in XLine format to StepEvents format. If
// use_device_step_events is true, we will filter out events that only happens
// on CPU.
StepEvents ConvertHostThreadsXLineToStepEvents(
    const XLineVisitor& line, bool use_device_step_events,
    const StepEvents& device_step_events);

// Convert the host threads in XPlane format to StepEvents format. If
// use_device_step_events is true, we will filter out events that only happens
// on CPU.
StepEvents ConvertHostThreadsXPlaneToStepEvents(
    const XPlane& host_trace, bool use_device_step_events,
    const StepEvents& device_step_events);

// Convert the device trace in XLine format to StepEvents.
StepEvents ConvertDeviceTraceXLineToStepEvents(const XLineVisitor& line);

// Convert the device trace in XPlane format to StepEvents.
StepEvents ConvertDeviceTraceXPlaneToStepEvents(const XPlane& device_trace);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_STEP_EVENTS_H_
