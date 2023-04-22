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

#include "tensorflow/core/profiler/utils/device_caps_utils.h"

#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

void SetDeviceCaps(const DeviceCapabilities& caps, XPlane* plane) {
  XPlaneBuilder xplane(plane);
  int clock_rate_in_khz =
      static_cast<int>(caps.clock_rate_in_ghz() * 1000000.0);
  xplane.AddStatValue(*xplane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kDevCapClockRateKHz)),
                      clock_rate_in_khz);
  xplane.AddStatValue(*xplane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kDevCapCoreCount)),
                      caps.num_cores());
  xplane.AddStatValue(*xplane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kDevCapMemoryBandwidth)),
                      caps.memory_bandwidth());
  xplane.AddStatValue(*xplane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kDevCapMemorySize)),
                      caps.memory_size_in_bytes());
  if (caps.has_compute_capability()) {
    xplane.AddStatValue(*xplane.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDevCapComputeCapMajor)),
                        caps.compute_capability().major());
    xplane.AddStatValue(*xplane.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDevCapComputeCapMinor)),
                        caps.compute_capability().minor());
  }
}

DeviceCapabilities GetDeviceCaps(const XPlane& plane) {
  DeviceCapabilities caps;
  XPlaneVisitor xplane(&plane);
  xplane.ForEachStat([&](const tensorflow::profiler::XStatVisitor& stat) {
    if (!stat.Type().has_value()) return;
    switch (stat.Type().value()) {
      case StatType::kDevCapClockRateKHz:
        caps.set_clock_rate_in_ghz(stat.IntOrUintValue() * 1000000.0);
        break;
      case StatType::kDevCapCoreCount:
        caps.set_num_cores(stat.IntOrUintValue());
        break;
      case StatType::kDevCapMemoryBandwidth:
        caps.set_memory_bandwidth(stat.IntOrUintValue());
        break;
      case StatType::kDevCapMemorySize:
        caps.set_memory_size_in_bytes(stat.IntOrUintValue());
        break;
      case StatType::kDevCapComputeCapMajor:
        caps.mutable_compute_capability()->set_major(stat.IntOrUintValue());
        break;
      case StatType::kDevCapComputeCapMinor:
        caps.mutable_compute_capability()->set_minor(stat.IntOrUintValue());
        break;
    }
  });

  return caps;
}

}  // namespace profiler
}  // namespace tensorflow
