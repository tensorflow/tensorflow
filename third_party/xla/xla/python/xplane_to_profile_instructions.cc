/* Copyright 2017 The OpenXLA Authors.

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
#include "xla/python/xplane_to_profile_instructions.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xla/xla.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

constexpr char kXPlanePb[] = "xplane.pb";
constexpr char kCostNameSep[] = "::";

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tsl::profiler::CreateTfXPlaneVisitor;
using tsl::profiler::FindPlanesWithPrefix;
using tsl::profiler::FindPlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::HostEventType;
using tsl::profiler::IsInternalEvent;
using tsl::profiler::ProfilerJoinPath;
using tsl::profiler::StatType;
using tsl::profiler::XEventMetadataVisitor;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;

struct HloLatencySpan {
  int64_t start_ps, end_ps;
};

struct HloOpInstance {
  std::string name_and_fingerprint;
  int64_t instance_id;

  bool operator==(const HloOpInstance& other) const {
    return name_and_fingerprint == other.name_and_fingerprint &&
           instance_id == other.instance_id;
  }

  template <typename H>
  friend H AbslHashValue(H h, const HloOpInstance& op_instance) {
    return H::combine(std::move(h), op_instance.name_and_fingerprint,
                      op_instance.instance_id);
  }
};

struct SumAndTotal {
  double sum = 0.0;
  int64_t total = 0;
};

void GetXPlaneLatencyInfo(
    const XPlaneVisitor& xplane,
    const absl::flat_hash_map<std::string, std::string>& hlo_module_info,
    absl::flat_hash_map<HloOpInstance, HloLatencySpan>* hlo_latency_info) {
  // Iterate events.
  xplane.ForEachLine([hlo_latency_info,
                      &hlo_module_info](const XLineVisitor& xline) {
    if (xline.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    xline.ForEachEvent([hlo_latency_info,
                        &hlo_module_info](const XEventVisitor& xevent) {
      int64_t event_type =
          xevent.Type().value_or(HostEventType::kUnknownHostEventType);
      if (IsInternalEvent(event_type)) return;
      std::optional<std::string> hlo_name = std::nullopt;
      std::optional<std::string> hlo_module_name = std::nullopt;
      std::optional<std::string> fingerprint = std::nullopt;
      std::optional<int64_t> program_id = std::nullopt;
      std::optional<int64_t> scope_range_id = std::nullopt;

      auto for_each_stat = [&](const XStatVisitor& stat) {
        if (stat.ValueCase() == tsl::profiler::XStat::VALUE_NOT_SET) return;
        // Store latency information for HLOs.
        if (stat.Name() == GetStatTypeStr(StatType::kHloOp)) {
          hlo_name = stat.ToString();
        }
        if (stat.Name() == GetStatTypeStr(StatType::kProgramId)) {
          program_id = stat.IntValue();
        }
        if (stat.Name() == GetStatTypeStr(StatType::kHloModule)) {
          hlo_module_name = stat.ToString();
        }
        if (stat.Name() == GetStatTypeStr(StatType::kScopeRangeId)) {
          scope_range_id = stat.IntValue();
        }
      };
      xevent.Metadata().ForEachStat(for_each_stat);
      xevent.ForEachStat(for_each_stat);
      if (!hlo_name.has_value() || !hlo_module_name.has_value()) {
        return;
      }

      if (hlo_module_name.has_value()) {
        std::string fingerprint_key = hlo_module_name.value();
        if (program_id.has_value()) {
          fingerprint_key = tsl::profiler::HloModuleNameWithProgramId(
              hlo_module_name.value(), program_id.value());
        }
        if (hlo_module_info.contains(fingerprint_key)) {
          fingerprint = hlo_module_info.at(fingerprint_key);
        }
      }
      std::string key = hlo_name.value();
      if (fingerprint.has_value()) {
        key = absl::StrCat(fingerprint.value(), kCostNameSep, hlo_name.value());
      }
      // Store the enclosing [start, end] range for device activity with this
      // fingerprint + name + scope_id. This ensures that ops that produce >1
      // activity fragment are accounted for properly, as the fragments will
      // share a scope_id but separate instances of the op will not. If there
      // is no scope_range_id, use the timestamp as a ~unique key to disable
      // merging spans.
      auto [it, inserted] = hlo_latency_info->try_emplace(
          HloOpInstance{key, scope_range_id.value_or(xevent.TimestampPs())},
          HloLatencySpan{xevent.TimestampPs(), xevent.EndTimestampPs()});
      if (!inserted) {
        it->second.start_ps =
            std::min(it->second.start_ps, xevent.TimestampPs());
        it->second.end_ps =
            std::max(it->second.end_ps, xevent.EndTimestampPs());
      }
    });
  });
}

std::unique_ptr<xla::HloModule> CreateModuleFromProto(
    const xla::HloModuleProto& proto) {
  auto config = xla::HloModule::CreateModuleConfigFromProto(proto, {});
  if (config.ok()) {
    auto module = xla::HloModule::CreateFromProto(proto, config.value());
    if (module.ok()) {
      return std::move(*module);
    }
  }
  return nullptr;
}

std::optional<std::string> GetHloModuleFingerprint(
    const xla::HloModuleProto& hlo_module_proto) {
  std::unique_ptr<xla::HloModule> hlo_module =
      CreateModuleFromProto(hlo_module_proto);
  if (hlo_module == nullptr) {
    return std::nullopt;
  }
  const auto& map = hlo_module->entry_computation()
                        ->root_instruction()
                        ->frontend_attributes()
                        .map();
  auto it = map.find("fingerprint_before_lhs");
  if (it != map.end()) {
    return it->second;
  }
  return std::nullopt;
}

void GetXPlaneHloModuleInfo(
    const XPlaneVisitor& xplane,
    absl::flat_hash_map<std::string, std::string>* hlo_module_info) {
  // Iterate events.
  xplane.ForEachEventMetadata([&](const XEventMetadataVisitor& event_metadata) {
    event_metadata.ForEachStat([&](const XStatVisitor& stat) {
      xla::HloProto hlo_proto;
      if (tsl::ParseProtoUnlimited(&hlo_proto, stat.BytesValue().data(),
                                   stat.BytesValue().size())) {
        const xla::HloModuleProto& hlo_module_proto = hlo_proto.hlo_module();

        std::optional<std::string> fingerprint =
            GetHloModuleFingerprint(hlo_module_proto);
        if (fingerprint.has_value()) {
          std::string key_with_id = tsl::profiler::HloModuleNameWithProgramId(
              hlo_module_proto.name(), hlo_module_proto.id());
          (*hlo_module_info)[key_with_id] = fingerprint.value();
        }
      }
    });
  });
}

}  // namespace

absl::Status ConvertXplaneUnderLogdirToProfiledInstructionsProto(
    const std::string& logdir, tensorflow::profiler::ProfiledInstructionsProto*
                                   profiled_instructions_proto) {
  // Find the xplane files for each host under logdir.
  std::vector<std::string> children_path;
  ABSL_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(logdir, &children_path));
  if (children_path.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find file under: ", logdir));
  }
  std::vector<tensorflow::profiler::XSpace> xspaces;
  for (const std::string& child_path : children_path) {
    if (absl::StrContains(child_path, kXPlanePb)) {
      std::string xspace_path = ProfilerJoinPath(logdir, child_path);
      tensorflow::profiler::XSpace xspace;
      ABSL_RETURN_IF_ERROR(
          ReadBinaryProto(tsl::Env::Default(), xspace_path, &xspace));
      xspaces.push_back(xspace);
    }
  }

  return ConvertXplaneToProfiledInstructionsProto(std::move(xspaces),
                                                  profiled_instructions_proto);
}

absl::Status ConvertXplaneToProfiledInstructionsProto(
    std::vector<tensorflow::profiler::XSpace> xspaces,
    tensorflow::profiler::ProfiledInstructionsProto*
        profiled_instructions_proto) {
  absl::flat_hash_map<std::string, std::string> hlo_module_info;
  absl::flat_hash_map<std::string, SumAndTotal> duration_stats;
  // Iterate through each host.
  for (const XSpace& xspace : xspaces) {
    const XPlane* metadata_plane =
        FindPlaneWithName(xspace, tsl::profiler::kMetadataPlaneName);
    if (metadata_plane != nullptr) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(metadata_plane);
      GetXPlaneHloModuleInfo(xplane, &hlo_module_info);
    }
    std::vector<const XPlane*> device_planes =
        FindPlanesWithPrefix(xspace, tsl::profiler::kGpuPlanePrefix);
    // We don't expect GPU and TPU planes and custom devices to be present in
    // the same XSpace.
    if (device_planes.empty()) {
      device_planes =
          FindPlanesWithPrefix(xspace, tsl::profiler::kTpuPlanePrefix);
    }
    if (device_planes.empty()) {
      device_planes =
          FindPlanesWithPrefix(xspace, tsl::profiler::kCustomPlanePrefix);
    }
    // Go over each device plane; hlo_latency_info is per-host to avoid
    // assuming scope_range_id is unique across hosts.
    absl::flat_hash_map<HloOpInstance, HloLatencySpan> hlo_latency_info;
    for (const XPlane* device_plane : device_planes) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(device_plane);
      GetXPlaneLatencyInfo(xplane, hlo_module_info, &hlo_latency_info);
    }
    for (const auto& [key, span] : hlo_latency_info) {
      SumAndTotal& stats = duration_stats[key.name_and_fingerprint];
      stats.sum += static_cast<double>(span.end_ps - span.start_ps) / 1e6;
      ++stats.total;
    }
  }

  // Get the mean duration for each hlo and store into the proto.
  for (const auto& [name, stats] : duration_stats) {
    auto* cost = profiled_instructions_proto->add_costs();
    cost->set_name(name);
    cost->set_cost_us(stats.sum / stats.total);
  }

  return absl::OkStatus();
}

}  // namespace xla
