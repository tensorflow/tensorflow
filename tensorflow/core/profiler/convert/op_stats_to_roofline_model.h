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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_ROOFLINE_MODEL_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_ROOFLINE_MODEL_H_

#include <cstdint>

#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/roofline_model.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace profiler {

using tensorflow::profiler::OpMetrics;
using tensorflow::profiler::roofline_model::RecordType;
using tensorflow::profiler::roofline_model::RooflineModelDatabase;
using tensorflow::profiler::roofline_model::RooflineModelRecord;

RooflineModelRecord ConvertOpMetricsToRooflineModelRecord(
    const OpStats& op_stats, const OpMetrics& metrics, RecordType record_type,
    uint32_t step_num, uint64_t total_time_ps,
    const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed);

RooflineModelRecord GenerateRooflineModelProgramRecord(
    const OpStats& op_stats, const OpMetricsDb& db, RecordType record_type,
    uint32_t step_num, const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed);

tsl::protobuf::RepeatedPtrField<RooflineModelRecord>
ConvertOpMetricsDbToRooflineModelRecords(
    const OpStats& op_stats, const OpMetricsDb& db, RecordType record_type,
    uint32_t step_num, const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed);

tensorflow::profiler::roofline_model::RooflineModelDatabase
ConvertOpStatsToRooflineModel(const tensorflow::profiler::OpStats& tf_op_stats,
                              bool include_infeed_outfeed);

tensorflow::profiler::roofline_model::RooflineModelDatabase
InitializeRooflineModelDatabaseFromOpStats(const OpStats& op_stats,
                                           bool include_infeed_outfeed);
// Generate RooflineModelRecord for the HLO DB over the entire profiling
// duration including incomplete steps.
inline void AddRooflineModelRecordForProfileDuration(
    const OpStats& op_stats, RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  *roofline_model_db.mutable_roofline_model_record() =
      ConvertOpMetricsDbToRooflineModelRecords(
          op_stats, op_stats.device_op_metrics_db(), RecordType::ALL,
          /*step_num=*/0, roofline_model_db, include_infeed_outfeed);
}

// Generate RooflineModelRecord for the HLO DB over complete steps only.
inline void AddRooflineModelRecordsForCompleteSteps(
    const OpStats& op_stats, RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  if (op_stats.has_hlo_metrics_db_complete_steps_only()) {
    *roofline_model_db.add_roofline_model_record() =
        GenerateRooflineModelProgramRecord(
            op_stats, op_stats.hlo_metrics_db_complete_steps_only(),
            RecordType::AVERAGE_STEP, /*step_num=*/0, roofline_model_db,
            include_infeed_outfeed);
  }
}

// Generate RooflineModelRecords for the per-step DBs.
inline void AddRooflineModelRecordsPerStep(
    const OpStats& op_stats, RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  for (const auto& step_info : op_stats.step_db().step_sequence()) {
    *roofline_model_db.add_roofline_model_record() =
        GenerateRooflineModelProgramRecord(
            op_stats, step_info.hlo_metrics_db(), RecordType::PER_STEP,
            step_info.step_num(), roofline_model_db, include_infeed_outfeed);
  }
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_ROOFLINE_MODEL_H_
