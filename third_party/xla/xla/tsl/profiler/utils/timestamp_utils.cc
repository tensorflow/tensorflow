/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/utils/timestamp_utils.h"

#include <cstdint>

#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

void SetSessionTimestamps(uint64_t start_walltime_ns, uint64_t stop_walltime_ns,
                          tensorflow::profiler::XSpace& space) {
  if (start_walltime_ns != 0 && stop_walltime_ns != 0) {
    tsl::profiler::XPlaneBuilder plane(
        tsl::profiler::FindOrAddMutablePlaneWithName(
            &space, tsl::profiler::kTaskEnvPlaneName));
    plane.AddStatValue(*plane.GetOrCreateStatMetadata(
                           GetTaskEnvStatTypeStr(kEnvProfileStartTime)),
                       start_walltime_ns);
    plane.AddStatValue(*plane.GetOrCreateStatMetadata(
                           GetTaskEnvStatTypeStr(kEnvProfileStopTime)),
                       stop_walltime_ns);
  } else {
    LOG(WARNING) << "Not Setting Session Timestamps, (start_walltime_ns, "
                    "stop_walltime_ns) : "
                 << start_walltime_ns << ", " << stop_walltime_ns;
  }
}

}  // namespace profiler
}  // namespace tsl
