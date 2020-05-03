/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

TfStatsRecord ConvertOpMetricsToTfStatsRecord(
    bool on_device, const OpMetrics& metrics,
    double ridge_point_operational_intensity) {
  TfStatsRecord record;
  record.set_host_or_device(on_device ? "Device" : "Host");
  record.set_is_eager(metrics.is_eager());
  record.set_op_type(metrics.category());
  record.set_op_name(metrics.name());
  SetExecutionTimes(metrics, &record);
  SetRooflineMetrics(metrics, ridge_point_operational_intensity, &record);
  return record;
}

TfStatsTable GenerateTfStatsTable(const OpMetricsDb& host_tf_metrics_db,
                                  const OpMetricsDb& device_tf_metrics_db,
                                  double ridge_point, bool exclude_idle) {
  TfStatsTable tf_stats_table;
  TfStatsRecord sentinel;
  sentinel.set_rank(0);
  sentinel.set_device_cumulative_total_self_time_as_fraction(0.0);
  sentinel.set_host_cumulative_total_self_time_as_fraction(0.0);
  const TfStatsRecord* prev_record = &sentinel;

  // Sets device-side TF stats.
  uint64 total_device_time_ps = device_tf_metrics_db.total_time_ps();
  if (exclude_idle) {
    total_device_time_ps -= IdleTimePs(device_tf_metrics_db);
  }
  double total_device_time_us = PicosToMicros(total_device_time_ps);
  for (const OpMetrics* metrics : SortedOpMetricsDb(device_tf_metrics_db)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        /*on_device=*/true, *metrics, ridge_point);
    SetRankAndDeviceTimeFractions(total_device_time_us, *prev_record, record);
    prev_record = record;
  }

  // Sets host-side TF stats.
  uint64 total_host_time_ps = host_tf_metrics_db.total_time_ps();
  if (exclude_idle) {
    total_host_time_ps -= IdleTimePs(host_tf_metrics_db);
  }
  double total_host_time_us = PicosToMicros(total_host_time_ps);
  for (const OpMetrics* metrics :
       tensorflow::profiler::SortedOpMetricsDb(host_tf_metrics_db)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        /*on_device=*/false, *metrics, ridge_point);
    SetRankAndHostTimeFractions(total_host_time_us, *prev_record, record);
    prev_record = record;
  }
  return tf_stats_table;
}

}  // namespace

TfStatsDatabase ConvertOpStatsToTfStats(const OpStats& op_stats) {
  const OpMetricsDb& host_tf_metrics_db = op_stats.host_op_metrics_db();
  OpMetricsDb device_tf_metrics_db =
      CreateTfMetricsDbFromDeviceOpMetricsDb(op_stats.device_op_metrics_db());
  double ridge_point = op_stats.perf_env().ridge_point();
  TfStatsDatabase tf_stats_db;
  *tf_stats_db.mutable_with_idle() =
      GenerateTfStatsTable(host_tf_metrics_db, device_tf_metrics_db,
                           ridge_point, /*exclude_idle=*/false);
  *tf_stats_db.mutable_without_idle() =
      GenerateTfStatsTable(host_tf_metrics_db, device_tf_metrics_db,
                           ridge_point, /*exclude_idle=*/true);
  return tf_stats_db;
}

}  // namespace profiler
}  // namespace tensorflow
