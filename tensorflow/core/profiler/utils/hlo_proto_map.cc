/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/hlo_proto_map.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/profiler/convert/xla_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

int NumHeapSimulatorTraceEvents(const xla::HloProto* hlo) {
  int result = 0;
  for (const auto& trace : hlo->buffer_assignment().heap_simulator_traces()) {
    result += trace.events_size();
  }
  return result;
}

}  // namespace

std::vector<std::pair<uint64_t, std::unique_ptr<xla::HloProto>>>
ParseHloProtosFromXSpace(const XSpace& space) {
  std::vector<std::pair<uint64_t, std::unique_ptr<xla::HloProto>>> hlo_protos;
  const XPlane* raw_plane = FindPlaneWithName(space, kMetadataPlaneName);
  if (raw_plane != nullptr) {
    XPlaneVisitor plane = CreateTfXPlaneVisitor(raw_plane);
    if (raw_plane->stats_size() > 0) {
      // Fallback for legacy aggregated XPlane.
      // TODO(b/235990417): Remove after 06/14/2023.
      plane.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.ValueCase() != XStat::kBytesValue) return;
        auto hlo_proto = std::make_unique<xla::HloProto>();
        if (hlo_proto->ParseFromString(stat.BytesValue())) {
          hlo_protos.emplace_back(stat.Id(), std::move(hlo_proto));
        }
      });
    } else {
      const XStatMetadata* hlo_proto_stat_metadata =
          plane.GetStatMetadataByType(StatType::kHloProto);
      if (hlo_proto_stat_metadata == nullptr) {
        // Fallback for legacy XPlane.
        // TODO(b/235990417): Remove after 06/14/2023.
        hlo_proto_stat_metadata = plane.GetStatMetadata(StatType::kHloProto);
      }
      if (hlo_proto_stat_metadata != nullptr) {
        plane.ForEachEventMetadata(
            [&](const XEventMetadataVisitor& event_metadata) {
              auto hlo_proto_stat = event_metadata.GetStat(
                  StatType::kHloProto, *hlo_proto_stat_metadata);
              if (!hlo_proto_stat) return;
              if (hlo_proto_stat->ValueCase() != XStat::kBytesValue) return;
              auto hlo_proto = std::make_unique<xla::HloProto>();
              if (hlo_proto->ParseFromString(hlo_proto_stat->BytesValue())) {
                hlo_protos.emplace_back(event_metadata.Id(),
                                        std::move(hlo_proto));
              }
            });
      }
    }
  }
  return hlo_protos;
}

void HloProtoMap::AddHloProto(uint64_t program_id,
                              const xla::HloProto* hlo_proto) {
  hlo_protos_by_program_id_.try_emplace(program_id, hlo_proto);
  absl::string_view hlo_module_name = hlo_proto->hlo_module().name();
  hlo_protos_by_name_.try_emplace(
      HloModuleNameWithProgramId(hlo_module_name, program_id), hlo_proto);
}

void HloProtoMap::AddHloProto(uint64_t program_id,
                              std::unique_ptr<const xla::HloProto> hlo_proto) {
  AddHloProto(program_id, hlo_proto.get());
  owned_hlo_protos_.push_back(std::move(hlo_proto));
}

void HloProtoMap::AddHloProtosFromXSpace(const XSpace& space) {
  for (auto& [program_id, hlo_proto] : ParseHloProtosFromXSpace(space)) {
    AddHloProto(program_id, std::move(hlo_proto));
  }
}

std::vector<absl::string_view> HloProtoMap::GetModuleList() const {
  std::vector<absl::string_view> module_list;
  module_list.reserve(hlo_protos_by_name_.size());
  for (const auto& [name, hlo_proto] : hlo_protos_by_name_) {
    module_list.push_back(name);
  }
  return module_list;
}

std::vector<absl::string_view> HloProtoMap::GetSortedModuleList() const {
  std::vector<absl::string_view> module_list = GetModuleList();
  absl::c_sort(module_list);
  return module_list;
}

std::vector<absl::string_view> HloProtoMap::GetSortedModuleListByHeapTraceSize()
    const {
  std::vector<std::pair<absl::string_view, const xla::HloProto*>> hlo_protos(
      hlo_protos_by_name_.begin(), hlo_protos_by_name_.end());

  // Sort the hlo protos by heap trace size and then by hlo module name.
  // This way trivial computations will be on the bottom of the list.
  absl::c_stable_sort(hlo_protos, [](const auto& a, const auto& b) {
    int num_a = NumHeapSimulatorTraceEvents(a.second);
    int num_b = NumHeapSimulatorTraceEvents(b.second);
    return std::tie(num_a, b.first) > std::tie(num_b, a.first);
  });

  std::vector<absl::string_view> module_list;
  module_list.reserve(hlo_protos.size());
  for (const auto& [name, hlo_proto] : hlo_protos) {
    module_list.push_back(name);
  }
  return module_list;
}

absl::StatusOr<const xla::HloProto*> HloProtoMap::GetHloProtoByModuleName(
    absl::string_view module_name) const {
  auto iter = hlo_protos_by_name_.find(module_name);
  if (iter != hlo_protos_by_name_.end()) {
    return iter->second;
  }
  return absl::NotFoundError(
      absl::StrCat("Module name: ", module_name, " is not found."));
}

}  // namespace profiler
}  // namespace tensorflow
