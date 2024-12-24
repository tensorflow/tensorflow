/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/op_stats_to_hlo_stats.h"

#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/protobuf/hlo_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using tensorflow::profiler::OpMetrics;
using tensorflow::profiler::OpMetricsDb;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::PerfEnv;
using ::tensorflow::profiler::RunEnvironment;
using tensorflow::profiler::hlo_stats::HloStatsDatabase;
using tensorflow::profiler::hlo_stats::HloStatsRecord;
using tsl::profiler::IsOutsideCompilationOp;

HloStatsRecord ConvertOpMetricsToHloStatsRecord(const OpMetrics& metrics,
                                                const PerfEnv& perf_env,
                                                const RunEnvironment& run_env) {
  HloStatsRecord record;
  record.set_program_id(metrics.hlo_module_id());
  record.set_hlo_expression(metrics.long_name());
  record.set_tf_op_name(metrics.provenance());
  record.set_hlo_category(metrics.category());
  record.set_autotuned(metrics.autotuned());
  tensorflow::profiler::SetExecutionTimes(metrics, &record);
  tensorflow::profiler::SetTpuUnitFractions(metrics, &record);
  SetRooflineMetrics(metrics, perf_env, run_env, &record);
  record.set_rematerialization(tsl::profiler::IsRematerialization(
      /*hlo_expression=*/metrics.long_name(),
      /*framework_op_name=*/metrics.provenance()));
  record.set_outside_compilation(
      IsOutsideCompilationOp(metrics.provenance(), metrics.long_name()));
  return record;
}

}  // namespace

HloStatsDatabase ConvertOpStatsToHloStats(const OpStats& op_stats) {
  HloStatsDatabase hlo_stats_db;
  const OpMetricsDb& hlo_metrics_db = op_stats.device_op_metrics_db();
  double total_device_time_us =
      tsl::profiler::PicoToMicro(hlo_metrics_db.total_time_ps());
  HloStatsRecord sentinel;
  sentinel.set_rank(0);
  sentinel.set_cumulative_total_self_time_as_fraction(0.0);
  const HloStatsRecord* prev_record = &sentinel;
  for (const OpMetrics* metrics :
       tensorflow::profiler::SortedOpMetricsDb(hlo_metrics_db)) {
    if (metrics->occurrences() == 0) continue;
    HloStatsRecord* record = hlo_stats_db.add_hlo_stats_record();
    *record = ConvertOpMetricsToHloStatsRecord(*metrics, op_stats.perf_env(),
                                               op_stats.run_environment());
    tensorflow::profiler::SetRankAndTimeFractions(total_device_time_us,
                                                  *prev_record, record);
    prev_record = record;
  }
  return hlo_stats_db;
}

}  // namespace profiler
}  // namespace tensorflow
