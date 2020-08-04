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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_ERRORS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_ERRORS_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Error message that the visualization is based on incomplete step.
ABSL_CONST_INIT extern const absl::string_view kErrorIncompleteStep;

// Error message that no step marker is seen and visualization contains no
// step info.
ABSL_CONST_INIT extern const absl::string_view kErrorNoStepMarker;

ABSL_CONST_INIT extern const absl::string_view kNoDeviceTraceCollected;

ABSL_CONST_INIT extern const absl::string_view kStepsDropped;

void PopulateStepDiagnostics(const OpStats& op_stats, Diagnostics* diag);

void PopulateOverviewDiagnostics(const OpStats& op_stats, Diagnostics* diag);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_ERRORS_H_
