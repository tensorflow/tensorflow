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
#include <ostream>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/gpu_event_stats.h"  // from @org_xprof
#include "xprof/utils/kernel_stats_utils.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {

void ConvertDeviceTraceXPlaneToKernelReports(
    const XPlane& device_trace,
    const std::function<void(const GpuEventStats&, KernelReport*)>&
        on_kernel_fn,
    KernelReportMap* reports) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) {
      return;
    }
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.DurationNs() == 0) return;
      KernelReport kernel;
      GpuEventStats stats(&event);
      if (!stats.IsKernel()) return;

      kernel.set_name(std::string(event.Name()));
      kernel.set_is_kernel_using_tensor_core(
          IsKernelUsingTensorCore(event.Name()));
      kernel.set_total_duration_ns(event.DurationNs());
      kernel.set_min_duration_ns(event.DurationNs());
      kernel.set_max_duration_ns(event.DurationNs());
      ParseKernelLaunchParams(stats.kernel_details, &kernel);

      if (stats.IsTfOp()) {
        tsl::profiler::TfOp tf_op =
            tsl::profiler::ParseTfOpFullname(stats.tf_op_fullname);
        kernel.set_op_name(std::string(tf_op.name));
        bool tensor_core_eligible =
            IsEinsumTensorCoreEligible(stats.equation) ||
            IsOpTensorCoreEligible(kernel.op_name());
        if (!tensor_core_eligible && kernel.is_kernel_using_tensor_core()) {
          VLOG(1) << "Detected new Op using TensorCores: " << kernel.op_name()
                  << std::endl;
          tensor_core_eligible = true;
        }
        kernel.set_is_op_tensor_core_eligible(tensor_core_eligible);
      }

      if (on_kernel_fn) {
        on_kernel_fn(stats, &kernel);
      }

      KernelReportValue value;
      value.total_duration_ns = event.DurationNs();
      value.min_duration_ns = event.DurationNs();
      value.max_duration_ns = event.DurationNs();
      value.occurrences = 1;
      InsertOrUpdateKernelReport(kernel, value, reports);
    });
  });
}

}  // namespace profiler
}  // namespace tensorflow
