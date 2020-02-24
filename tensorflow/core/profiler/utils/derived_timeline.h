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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/group_events.h"

namespace tensorflow {
namespace profiler {

typedef std::function<absl::string_view(absl::string_view hlo_module_name,
                                        absl::string_view hlo_op)>
    SymbolResolver;

// Derives "Step Info", "Tensorflow Ops", "XLA Ops" and "XLA Module" lines in
// an NVIDIA_GPU device trace from data passed as ScopedAnnotations and stored
// as XStats in XEvents corresponding to GPU Kernels. Consecutive annotations
// with the same value are merged into a single event except for XLA modules.
// The device_trace is both input and output.
void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const EventGroupNameMap& event_group_name_map,
                                 XPlane* device_trace,
                                 bool step_info_only = false);

// Derives "Launch Activities Summary" line from host trace.
void DeriveEventsFromHostTrace(const XPlane* host_trace,
                               const EventGroupNameMap& event_group_name_map,
                               std::vector<XPlane*> device_traces);

// Loops through XPlanes of input XSpace, if it is "device" XPlane, generating
// derived timelines for the plane by calling DeriveEventsFromAnnotations.
void GenerateDerivedTimeLines(const EventGroupNameMap& event_group_name_map,
                              XSpace* space, bool step_info_only = false);
void GenerateDerivedTimeLines(const EventGroupNameMap& event_group_name_map,
                              const std::vector<XPlane*>& device_traces,
                              bool step_info_only = false);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
