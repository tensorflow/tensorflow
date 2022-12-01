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
#include "tensorflow/core/profiler/utils/tfstreamz_utils.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/tfstreamz.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"

namespace tensorflow {
namespace profiler {

namespace {

std::string ConstructXStatName(absl::string_view name,
                               const monitoring::Point& point) {
  if (point.labels.empty()) {
    return std::string(name);
  }
  return absl::Substitute(
      "$0{$1}", name,
      absl::StrJoin(
          point.labels, ", ",
          [](std::string* out, const monitoring::Point::Label& label) {
            absl::StrAppend(out, label.name, "=", label.value);
          }));
}

tfstreamz::Percentiles ToProto(const monitoring::Percentiles& percentiles) {
  tfstreamz::Percentiles output;
  output.set_unit_of_measure(
      static_cast<tfstreamz::UnitOfMeasure>(percentiles.unit_of_measure));
  output.set_start_nstime(percentiles.start_nstime);
  output.set_end_nstime(percentiles.end_nstime);
  output.set_min_value(percentiles.min_value);
  output.set_max_value(percentiles.max_value);
  output.set_mean(percentiles.mean);
  output.set_stddev(percentiles.stddev);
  output.set_num_samples(percentiles.num_samples);
  output.set_total_samples(percentiles.total_samples);
  output.set_accumulator(percentiles.accumulator);
  for (const auto& pp : percentiles.points) {
    auto* percentile_point = output.add_points();
    percentile_point->set_percentile(pp.percentile);
    percentile_point->set_value(pp.value);
  }
  return output;
}

}  // namespace

Status SerializeToXPlane(const std::vector<TfStreamzSnapshot>& snapshots,
                         XPlane* plane, uint64 line_start_time_ns) {
  XPlaneBuilder xplane(plane);
  XLineBuilder line = xplane.GetOrCreateLine(0);  // This plane has single line.
  line.SetTimestampNs(line_start_time_ns);

  // For each snapshot, create a virtual event.
  for (const auto& snapshot : snapshots) {
    XEventMetadata* event_metadata =
        xplane.GetOrCreateEventMetadata("TFStreamz Snapshot");
    XEventBuilder xevent = line.AddEvent(*event_metadata);
    xevent.SetTimestampNs(snapshot.start_time_ns);
    xevent.SetEndTimestampNs(snapshot.end_time_ns);
    auto& metric_descriptor_map = snapshot.metrics->metric_descriptor_map;
    for (const auto& point_set : snapshot.metrics->point_set_map) {
      const std::string& metric_name = point_set.first;
      // Each metrics have multiple points corresponding to different labels.
      for (const auto& point : point_set.second->points) {
        // Generates one KPI metric for each point.
        std::string stat_name = ConstructXStatName(metric_name, *point);
        auto* metadata = xplane.GetOrCreateStatMetadata(stat_name);
        auto it = metric_descriptor_map.find(metric_name);
        if (it != metric_descriptor_map.end()) {
          metadata->set_description(it->second->description);
        }
        switch (point->value_type) {
          case monitoring::ValueType::kInt64:
            xevent.AddStatValue(*metadata, point->int64_value);
            break;
          case monitoring::ValueType::kBool:
            xevent.AddStatValue(*metadata, point->bool_value);
            break;
          case monitoring::ValueType::kString:
            xevent.AddStatValue(*metadata, *xplane.GetOrCreateStatMetadata(
                                               point->string_value));
            break;
          case monitoring::ValueType::kHistogram:
            xevent.AddStatValue(*metadata, point->histogram_value);
            break;
          case monitoring::ValueType::kPercentiles:
            xevent.AddStatValue(*metadata, ToProto(point->percentiles_value));
            break;
        }
      }
    }
  }
  return OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
