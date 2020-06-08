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

#include "tensorflow/core/profiler/utils/diagnostics.h"

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {

const absl::string_view kErrorIncompleteStep =
    "Incomplete step observed and hence the step time is unknown."
    "Instead, we use the trace duration as the step time. This may happen"
    " if your profiling duration is shorter than the step time. In this"
    " case, you may try to profile longer.";

const absl::string_view kErrorNoStepMarker =
    "No step marker observed and hence the step time is unknown."
    " This may happen if (1) training steps are not instrumented (e.g., if"
    " you are not using Keras) or (2) the profiling duration is shorter"
    " than the step time. For (1), you need to add step instrumentation;"
    " for (2), you may try to profile longer.";

const absl::string_view kNoDeviceTraceCollected =
    "No device trace was collected. This might happen if your job hadn't been "
    "run on the device when sampling was turned on. You could try the sampling"
    " again later.";

void PopulateStepDiagnostics(const OpStats& op_stats, Diagnostics* diag) {
  if (op_stats.step_db().use_incomplete_step()) {
    *diag->add_warnings() = std::string(kErrorIncompleteStep);
  } else if (op_stats.step_db().step_sequence().empty()) {
    *diag->add_warnings() = std::string(kErrorNoStepMarker);
  }
}

void PopulateOverviewDiagnostics(const OpStats& op_stats, Diagnostics* diag) {
  *diag->mutable_errors() = op_stats.errors();
  absl::c_sort(*diag->mutable_errors());
  if (diag->errors().empty()) {
    // Shows run-environment error only if there is no other existing error.
    if (op_stats.run_environment().device_type() != "CPU" &&
        op_stats.run_environment().device_core_count() <= 0) {
      *diag->add_errors() = std::string(kNoDeviceTraceCollected);
    }
  }
  PopulateStepDiagnostics(op_stats, diag);
}

}  // namespace profiler
}  // namespace tensorflow
