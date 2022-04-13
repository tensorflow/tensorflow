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

#include "tensorflow/core/profiler/convert/xplane_to_kernel_stats_db.h"

#include <functional>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

void ConvertDeviceTraceXPlaneToKernelReports(
    const XPlane& device_trace,
    const std::function<void(const XEventVisitor&, KernelReport*)>&
        on_kernel_fn,
    KernelReportMap* reports) {
  XPlaneVisitor plane = CreateTfXPlaneVisitor(&device_trace);
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) {
      return;
    }
    line.ForEachEvent([&](const XEventVisitor& event) {
      absl::string_view tf_op_fullname;
      KernelReport kernel;

      absl::string_view equation;
      event.ForEachStat([&](const tensorflow::profiler::XStatVisitor& stat) {
        if (!stat.Type().has_value()) return;
        switch (stat.Type().value()) {
          case StatType::kTfOp:
            tf_op_fullname = stat.StrOrRefValue();
            break;
          case StatType::kKernelDetails: {
            kernel.set_name(event.Name().data(), event.Name().size());
            kernel.set_is_kernel_using_tensor_core(
                IsKernelUsingTensorCore(event.Name()));
            kernel.set_total_duration_ns(event.DurationNs());
            kernel.set_min_duration_ns(event.DurationNs());
            kernel.set_max_duration_ns(event.DurationNs());
            absl::string_view launch_params = stat.StrOrRefValue();
            ParseKernelLaunchParams(launch_params, &kernel);
            break;
          }
          case StatType::kEquation:
            equation = stat.StrOrRefValue();
            break;
        }
      });

      if (!tf_op_fullname.empty()) {
        tensorflow::profiler::TfOp tf_op = ParseTfOpFullname(tf_op_fullname);

        if (kernel.total_duration_ns()) {
          kernel.set_op_name(tf_op.name.data(), tf_op.name.size());
          bool tensor_core_eligible = IsEinsumTensorCoreEligible(equation) ||
                                      IsOpTensorCoreEligible(kernel.op_name());

          if (!tensor_core_eligible && kernel.is_kernel_using_tensor_core()) {
            VLOG(1) << "Detected new Op using TensorCores: " << kernel.op_name()
                    << std::endl;
            tensor_core_eligible = true;
          }

          kernel.set_is_op_tensor_core_eligible(tensor_core_eligible);
        }
      }

      if (on_kernel_fn) {
        on_kernel_fn(event, &kernel);
      }

      if (kernel.total_duration_ns()) {
        KernelReportValue value;
        value.total_duration_ns = event.DurationNs();
        value.min_duration_ns = event.DurationNs();
        value.max_duration_ns = event.DurationNs();
        value.occurrences = 1;
        InsertOrUpdateKernelReport(kernel, value, reports);
      }
    });
  });
}

}  // namespace profiler
}  // namespace tensorflow
