/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_CPU_METADATA_UTILS_H_
#define XLA_BACKENDS_PROFILER_CPU_METADATA_UTILS_H_

#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

class MetadataXPlaneBuilder {
 public:
  explicit MetadataXPlaneBuilder(tsl::profiler::XPlane* raw_plane)
      : plane_(raw_plane),
        hlo_proto_stat_(plane_.GetOrCreateStatMetadata(
            GetStatTypeStr(tsl::profiler::StatType::kHloProto))),
        program_id_stat_(plane_.GetOrCreateStatMetadata(
            GetStatTypeStr(tsl::profiler::StatType::kProgramId))),
        original_hlo_proto_stat_(plane_.GetOrCreateStatMetadata(
            GetStatTypeStr(tsl::profiler::StatType::kOriginalHloProto))) {}

  void AddHloProto(uint64_t program_id, const xla::HloProto& hlo_proto) {
    auto name = tsl::profiler::HloModuleNameWithProgramId(
        hlo_proto.hlo_module().name(), program_id);
    tsl::profiler::XEventMetadata* event_metadata =
        plane_.GetOrCreateEventMetadata(name);
    if (event_metadata->display_name().empty()) {
      event_metadata->set_display_name(name);
    }
    bool has_hlo_proto = false;
    bool has_program_id = false;
    for (const auto& stat : event_metadata->stats()) {
      if (stat.metadata_id() == hlo_proto_stat_->id()) has_hlo_proto = true;
      if (stat.metadata_id() == program_id_stat_->id()) has_program_id = true;
    }
    tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> event_stats(
        event_metadata, &plane_);
    if (!has_hlo_proto) {
      event_stats.AddStatValue(*hlo_proto_stat_, hlo_proto);
    }
    if (!has_program_id) {
      event_stats.AddStatValue(*program_id_stat_, program_id);
    }
  }

  void AddOriginalHloProto(uint64_t program_id,
                           const xla::HloProto& hlo_proto) {
    auto name = tsl::profiler::HloModuleNameWithProgramId(
        hlo_proto.hlo_module().name(), program_id);
    tsl::profiler::XEventMetadata* event_metadata =
        plane_.GetOrCreateEventMetadata(name);
    if (event_metadata->display_name().empty()) {
      event_metadata->set_display_name(name);
    }
    bool has_original_hlo_proto = false;
    bool has_program_id = false;
    for (const auto& stat : event_metadata->stats()) {
      if (stat.metadata_id() == original_hlo_proto_stat_->id())
        has_original_hlo_proto = true;
      if (stat.metadata_id() == program_id_stat_->id()) has_program_id = true;
    }
    tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> event_stats(
        event_metadata, &plane_);
    if (!has_original_hlo_proto) {
      event_stats.AddStatValue(*original_hlo_proto_stat_, hlo_proto);
    }
    if (!has_program_id) {
      event_stats.AddStatValue(*program_id_stat_, program_id);
    }
  }

 private:
  tsl::profiler::XPlaneBuilder plane_;
  const tsl::profiler::XStatMetadata* hlo_proto_stat_ = nullptr;
  const tsl::profiler::XStatMetadata* program_id_stat_ = nullptr;
  const tsl::profiler::XStatMetadata* original_hlo_proto_stat_ = nullptr;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_CPU_METADATA_UTILS_H_
