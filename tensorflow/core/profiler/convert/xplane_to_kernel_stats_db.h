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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_KERNEL_STATS_DB_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_KERNEL_STATS_DB_H_

#include <functional>
#include <ostream>

#include "absl/log/log.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/tensorboard_plugin_profile/protobuf/kernel_stats.pb.h"  // from @org_xprof
#include "xprof/utils/gpu_event_stats.h"  // from @org_xprof
#include "xprof/utils/hlo_module_map.h"  // from @org_xprof
#include "xprof/utils/kernel_stats_utils.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {

void ConvertDeviceTraceXPlaneToKernelReports(
    const XPlane& device_trace,
    const std::function<void(const GpuEventStats&, KernelReport*)>&
        on_kernel_fn,
    KernelReportMap* reports);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_KERNEL_STATS_DB_H_
