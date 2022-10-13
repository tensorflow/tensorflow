/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Standard format in which the metrics are collected, before being exported.
// These are to be used only by the CollectionRegistry and exporters which
// collect metrics using the CollectionRegistry.

#ifndef TENSORFLOW_TSL_LIB_MONITORING_COLLECTED_METRICS_H_
#define TENSORFLOW_TSL_LIB_MONITORING_COLLECTED_METRICS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/tsl/lib/monitoring/metric_def.h"
#include "tensorflow/tsl/lib/monitoring/types.h"
#include "tensorflow/tsl/protobuf/histogram.pb.h"

namespace tsl {
namespace monitoring {

// A metric is a statistic about a monitorable entity.
//
// Metrics are named with path-like strings, which must conform to the regular
// expression (/[a-zA-Z0-9_-]+)+.  For example:
//
//     /proc/cpu_usage
//     /rpc/client/count
//
// Metrics may optionally have labels, which are additional dimensions used to
// identify the metric's values.  For example, the metric /rpc/client/count
// might have two labels named "rpc_service" and "rpc_method".
//
// A label name must be an identifier, which conform to the regular expression
// [a-zA-Z_][a-zA-Z_0-9]*, and is only unique within the context of the metric
// it is a label for.
//
// MetricDescriptor defines the structure of the metric (e.g. the fact that it's
// a counter and that it has two labels named "rpc_service" and "rpc_method").
// Individual points will provide a value for the metric (e.g. the counter
// value) and specific values for each of the labels.
//
// There's no scoping relationship between metrics and monitorable entities: the
// metric /rpc/client/count should be defined the same way no matter which
// monitorable entity is exporting it.
struct MetricDescriptor {
  // Metric names are path-like.  E.g., "/mycomponent/mymetric".
  string name;

  // A human-readable description of what this metric measures.
  string description;

  // Label names for the metric.
  // See the example in the top level comment for MetricDescriptor.
  std::vector<string> label_names;

  MetricKind metric_kind;

  ValueType value_type;
};

struct Point {
  // Usually a Point should provide a |label| field for each of the labels
  // defined in the corresponding MetricDescriptor.  During transitions in
  // metric definitions, however, there may be times when a Point provides more
  // or fewer labels than those that appear in the MetricDescriptor.
  struct Label {
    // The |name| field must match the |label_name| field in the
    // MetricDescriptor for this Point.
    string name;
    string value;
  };
  std::vector<Label> labels;

  // The actual metric value, dependent on the value_type enum.
  ValueType value_type;
  int64_t int64_value;
  string string_value;
  bool bool_value;
  HistogramProto histogram_value;
  Percentiles percentiles_value;

  // start_timestamp and end_timestamp indicate the time period over which this
  // point's value measurement applies.
  //
  // A cumulative metric like /rpc/client/count typically has runs of
  // consecutive points that share a common start_timestamp, which is often
  // the time at which the exporting process started.  For example:
  //
  //   value:  3  start_timestamp: 1000  end_timestamp: 1234
  //   value:  7  start_timestamp: 1000  end_timestamp: 1245
  //   value: 10  start_timestamp: 1000  end_timestamp: 1256
  //   value: 15  start_timestamp: 1000  end_timestamp: 1267
  //   value: 21  start_timestamp: 1000  end_timestamp: 1278
  //   value:  4  start_timestamp: 1300  end_timestamp: 1400
  //
  // The meaning of each point is: "Over the time period from
  // 'start_timestamp' to 'end_timestamp', 'value' client RPCs finished."
  //
  // Note the changed start_timestamp and the decrease in 'value' in the
  // last line; those are the effects of the process restarting.
  //
  // Delta metrics have the same interpretation of the timestamps and values,
  // but the time ranges of two points do not overlap.  The delta form of the
  // above sequence would be:
  //
  //   value:  3  start_timestamp: 1000  end_timestamp: 1234
  //   value:  4  start_timestamp: 1235  end_timestamp: 1245
  //   value:  3  start_timestamp: 1246  end_timestamp: 1256
  //   value:  5  start_timestamp: 1257  end_timestamp: 1267
  //   value:  6  start_timestamp: 1268  end_timestamp: 1278
  //   value:  4  start_timestamp: 1300  end_timestamp: 1400
  //
  // For gauge metrics whose values are instantaneous measurements,
  // start_timestamp and end_timestamp may be identical.  I.e., there is no need
  // to strictly measure the time period during which the value measurement was
  // made.
  //
  // start_timestamp must not be younger than end_timestamp.
  uint64 start_timestamp_millis;
  uint64 end_timestamp_millis;
};

// A set of points belonging to a metric.
struct PointSet {
  // This must match a name defined by a MetricDescriptor message.
  string metric_name;

  // No two Points in the same PointSet should have the same set of labels.
  std::vector<std::unique_ptr<Point>> points;
};

// Standard format in which the metrics are collected, before being exported.
struct CollectedMetrics {
  // The keys are the metric-names.
  std::map<string, std::unique_ptr<MetricDescriptor>> metric_descriptor_map;
  std::map<string, std::unique_ptr<PointSet>> point_set_map;
};

}  // namespace monitoring
}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_MONITORING_COLLECTED_METRICS_H_
