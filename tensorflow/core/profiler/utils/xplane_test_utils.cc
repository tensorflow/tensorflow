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
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

class XStatValueVisitor {
 public:
  XStatValueVisitor(XEventBuilder* event, const XStatMetadata* stat_metadata)
      : event_(event), stat_metadata_(stat_metadata) {}

  template <typename T>
  void operator()(const T& value) {
    event_->AddStatValue(*stat_metadata_, value);
  }

 private:
  XEventBuilder* event_;
  const XStatMetadata* stat_metadata_;
};

}  // namespace

XPlane* GetOrCreateHostXPlane(XSpace* space) {
  return FindOrAddMutablePlaneWithName(space, kHostThreadsPlaneName);
}

XPlane* GetOrCreateTpuXPlane(XSpace* space, int32_t device_ordinal,
                             absl::string_view device_type,
                             double peak_tera_flops_per_second,
                             double peak_hbm_bw_gigabytes_per_second) {
  std::string name = TpuPlaneName(device_ordinal);
  XPlane* xplane = FindOrAddMutablePlaneWithName(space, name);
  XPlaneBuilder builder(xplane);
  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(kDeviceTypeString)),
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
    XStatValueVisitor stat_value_visitor(
        &event_builder,
        plane_builder->GetOrCreateStatMetadata(GetStatTypeStr(stat_type)));
    absl::visit(stat_value_visitor, stat_value);
  }
}

void CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64_t offset_ps, int64_t duration_ps,
    std::initializer_list<std::pair<StatType, XStatValue>> stats) {
  CreateXEvent(plane_builder, line_builder, GetHostEventTypeStr(event_type),
               offset_ps, duration_ps, stats);
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
}  // namespace tensorflow
