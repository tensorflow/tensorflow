/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/tsl/profiler/convert/xplane_to_profile_instructions.h"

#include <numeric>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/file_system_utils.h"
#include "tensorflow/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/tsl/profiler/utils/xplane_utils.h"

namespace tsl {
namespace profiler {
namespace {

constexpr char kXPlanePb[] = "xplane.pb";

void GetXPlaneLatencyInfo(
    const XPlaneVisitor& xplane,
    absl::flat_hash_map<std::string, HloLatencyInfo>* hlo_latency_info) {
  // Iterate events.
  xplane.ForEachLine([hlo_latency_info](const XLineVisitor& xline) {
    if (xline.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    xline.ForEachEvent([hlo_latency_info](const XEventVisitor& xevent) {
      int64_t event_type =
          xevent.Type().value_or(HostEventType::kUnknownHostEventType);
      if (IsInternalEvent(event_type)) return;
      auto for_each_stat = [&](const XStatVisitor& stat) {
        if (stat.ValueCase() == XStat::VALUE_NOT_SET) return;
        if (IsInternalStat(stat.Type())) return;
        // Store latency information for HLOs.
        if (stat.Name() == GetStatTypeStr(StatType::kHloOp)) {
          std::string hlo_name = stat.ToString();
          double latency = static_cast<double>(xevent.DurationNs()) / 1e3;
          (*hlo_latency_info)[hlo_name].durations.emplace_back(latency);
        }
      };
      xevent.Metadata().ForEachStat(for_each_stat);
      xevent.ForEachStat(for_each_stat);
    });
  });
}

}  // namespace

Status ConvertXplaneToProfiledInstructionsProto(
    const std::string& logdir, tensorflow::profiler::ProfiledInstructionsProto*
                                   profiled_instructions_proto) {
  // Find the xplane files for each host under logdir.
  std::vector<string> children_path;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(logdir, &children_path));
  if (children_path.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find file under: ", logdir));
  }
  std::vector<tensorflow::profiler::XSpace> xspaces;
  for (const string& child_path : children_path) {
    if (absl::StrContains(child_path, kXPlanePb)) {
      std::string xspace_path = ProfilerJoinPath(logdir, child_path);
      tensorflow::profiler::XSpace xspace;
      TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), xspace_path, &xspace));
      xspaces.emplace_back(xspace);
    }
  }

  // Gets the duration information for each hlo.
  absl::flat_hash_map<std::string, HloLatencyInfo> hlo_latency_info;
  // Iterate through each host.
  for (const XSpace& xspace : xspaces) {
    std::vector<const XPlane*> device_planes =
        FindPlanesWithPrefix(xspace, kGpuPlanePrefix);
    // We don't expect GPU and TPU planes and custom devices to be present in
    // the same XSpace.
    if (device_planes.empty()) {
      device_planes = FindPlanesWithPrefix(xspace, kTpuPlanePrefix);
    }
    if (device_planes.empty()) {
      device_planes = FindPlanesWithPrefix(xspace, kCustomPlanePrefix);
    }
    // Go over each device plane.
    for (const XPlane* device_plane : device_planes) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(device_plane);
      GetXPlaneLatencyInfo(xplane, &hlo_latency_info);
    }
  }

  // Get the mean duration for each hlo and store into the proto.
  for (const auto& iter : hlo_latency_info) {
    auto* cost = profiled_instructions_proto->add_costs();
    std::vector<double> durations = iter.second.durations;
    double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
    cost->set_cost_us(sum / durations.size());
    cost->set_name(iter.first);
  }

  return OkStatus();
}

}  // namespace profiler
}  // namespace tsl
