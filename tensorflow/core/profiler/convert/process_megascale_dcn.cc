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
#include "tensorflow/core/profiler/convert/process_megascale_dcn.h"

#include <vector>

#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "tensorflow/core/profiler/convert/dcn_analysis.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::CreateTfXPlaneVisitor;
using tsl::profiler::FindMutableTensorCorePlanes;

void ProcessMegascaleDcn(XSpace* space) {
  std::vector<XPlane*> device_xplanes = FindMutableTensorCorePlanes(space);
  int num_tpu_cores = device_xplanes.size();
  // DCN TraceMe's are in the Host XPlane
  XPlane* host_plane =
      FindMutablePlaneWithName(space, tsl::profiler::kHostThreadsPlaneName);
  const XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(host_plane);
  // TODO(yashjs): Update parameter value for `is_megacore`.
  DcnEventsProcessor dcn_events_processor(num_tpu_cores, false);
  dcn_events_processor.SetupMessageInfo(plane_visitor);
  if (dcn_events_processor.HasDcnMessages(
          tsl::profiler::kMegaScaleDcnReceive)) {
    dcn_events_processor.ProcessReceiveMessages(plane_visitor);
  }
  // Update host XPlane with DCN traffic
  dcn_events_processor.AddHostDcnTrafficToXPlane(host_plane);
  // Update device XPlanes with per collective TPU traffic.
  for (XPlane* device_xplane : device_xplanes) {
    dcn_events_processor.AddTpuCollectiveDcnTrafficToXPlane(device_xplane);
  }

  SortXSpace(space);
}
}  // namespace profiler
}  // namespace tensorflow
