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
#include "xla/tsl/profiler/utils/xplane_test_utils.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

template <typename T>
class XStatValueVisitor {
 public:
  XStatValueVisitor(XStatsBuilder<T>& stats_owner,
                    const XStatMetadata* stat_metadata)
      : stats_owner_(stats_owner), stat_metadata_(stat_metadata) {}

  template <typename V>
  void operator()(const V& value) {
    stats_owner_.SetOrAddStatValue(*stat_metadata_, value);
  }

 private:
  XStatsBuilder<T>& stats_owner_;
  const XStatMetadata* stat_metadata_;
};

}  // namespace

XPlane* GetOrCreateHostXPlane(XSpace* space) {
  return FindOrAddMutablePlaneWithName(space, kHostThreadsPlaneName);
}

XPlane* GetOrCreateTpuXPlane(XSpace* space, int32_t device_ordinal,
                             absl::string_view device_type,
                             double peak_tera_flops_per_second,
                             double peak_hbm_bw_gigabytes_per_second,
                             std::optional<int32_t> sparsecore_core_id) {
  std::string name = TpuPlaneName(device_ordinal);
  if (sparsecore_core_id.has_value()) {
    name = std::string(
        absl::StrCat(name, " SparseCore ", sparsecore_core_id.value()));
  }
  XPlane* xplane = FindOrAddMutablePlaneWithName(space, name);
  XPlaneBuilder builder(xplane);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       device_type);
  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata("peak_teraflops_per_second"),
      peak_tera_flops_per_second);
  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata("peak_hbm_bw_gigabytes_per_second"),
      peak_hbm_bw_gigabytes_per_second);
  return xplane;
}

XPlane* GetOrCreateGpuXPlane(XSpace* space, int32_t device_ordinal) {
  std::string name = GpuPlaneName(device_ordinal);
  return FindOrAddMutablePlaneWithName(space, name);
}

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64_t offset_ps, int64_t duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats) {
  auto event_builder = line_builder->AddEvent(
      *plane_builder->GetOrCreateEventMetadata(event_name));
  event_builder.SetOffsetPs(offset_ps);
  event_builder.SetDurationPs(duration_ps);
  for (const auto& stat_type_and_value : stats) {
    StatType stat_type = stat_type_and_value.first;
    const XStatValue& stat_value = stat_type_and_value.second;
    XStatValueVisitor<XEvent> event_stat_visitor(
        event_builder,
        plane_builder->GetOrCreateStatMetadata(GetStatTypeStr(stat_type)));
    std::visit(event_stat_visitor, stat_value);
  }
}

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64_t offset_ps, int64_t duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats) {
  CreateXEvent(plane_builder, line_builder, GetHostEventTypeStr(event_type),
               offset_ps, duration_ps, stats);
}

void CreateXEventMetadata(
    XPlaneBuilder* plane_builder, absl::string_view event_name,
    std::initializer_list<std::pair<StatType, XStatValue>> stats) {
  XEventMetadata* event_metadata =
      plane_builder->GetOrCreateEventMetadata(event_name);
  XStatsBuilder<XEventMetadata> event_metadata_stats(event_metadata,
                                                     plane_builder);
  for (const auto& [stat_type, stat_value] : stats) {
    XStatValueVisitor<XEventMetadata> event_metadata_stat_visitor(
        event_metadata_stats,
        plane_builder->GetOrCreateStatMetadata(GetStatTypeStr(stat_type)));
    std::visit(event_metadata_stat_visitor, stat_value);
  }
}

void CreateTfFunctionCallEvent(XPlaneBuilder* plane_builder,
                               XLineBuilder* line_builder,
                               absl::string_view function_name,
                               int64_t offset_ps, int64_t duration_ps,
                               absl::string_view execution_mode,
                               int64_t tracing_count) {
  if (tracing_count >= 0) {
    // Adds the tracing_count stats only if tracing_count is valid.
    CreateXEvent(plane_builder, line_builder, function_name, offset_ps,
                 duration_ps,
                 {{StatType::kTfFunctionCall, execution_mode},
                  {StatType::kTfFunctionTracingCount, tracing_count}});
  } else {
    CreateXEvent(plane_builder, line_builder, function_name, offset_ps,
                 duration_ps, {{StatType::kTfFunctionCall, execution_mode}});
  }
}

}  // namespace profiler
}  // namespace tsl
