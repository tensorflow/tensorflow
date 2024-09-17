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
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/utils/xplane_schema.h"

namespace xla {
namespace profiler {

class MetadataXPlaneBuilder {
 public:
  explicit MetadataXPlaneBuilder(tsl::profiler::XPlane* raw_plane)
      : plane_(raw_plane),
        hlo_proto_stat_(plane_.GetOrCreateStatMetadata(
            GetStatTypeStr(tsl::profiler::StatType::kHloProto))) {}

  void AddHloProto(uint64_t program_id, const xla::HloProto& hlo_proto) {
    tsl::profiler::XEventMetadata* event_metadata =
        plane_.GetOrCreateEventMetadata(program_id);
    if (event_metadata->name().empty()) {
      event_metadata->set_name(tsl::profiler::HloModuleNameWithProgramId(
          hlo_proto.hlo_module().name(), program_id));
      tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> event_stats(
          event_metadata, &plane_);
      event_stats.AddStatValue(*hlo_proto_stat_, hlo_proto);
    }
  }

 private:
  tsl::profiler::XPlaneBuilder plane_;
  const tsl::profiler::XStatMetadata* hlo_proto_stat_ = nullptr;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_CPU_METADATA_UTILS_H_
