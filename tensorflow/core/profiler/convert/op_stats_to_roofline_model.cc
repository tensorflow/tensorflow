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

#include "tensorflow/core/profiler/convert/op_stats_to_roofline_model.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/check.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/roofline_model.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tsl/platform/protobuf.h"
#include "xprof/utils/diagnostics.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {
namespace {

using tensorflow::profiler::OpMetrics;
using tensorflow::profiler::OpMetricsDb;
using tensorflow::profiler::PerfEnv;
using tensorflow::profiler::roofline_model::RecordType;
using tensorflow::profiler::roofline_model::RooflineModelDatabase;
using tensorflow::profiler::roofline_model::RooflineModelRecord;

// The maximum number of records to generate.
const uint32_t kMaxNumRecords = 1000;
}  // namespace

RooflineModelRecord ConvertOpMetricsToRooflineModelRecord(
    const OpStats& op_stats, const OpMetrics& metrics, RecordType record_type,
    uint32_t step_num, uint64_t total_time_ps,
    const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  RooflineModelRecord record;
  record.set_hlo_name(metrics.name());
  record.set_hlo_category(metrics.category());
  record.set_hlo_module_id(metrics.hlo_module_id());
  record.set_record_type(record_type);
  record.set_step_num(step_num);
  SetExecutionTimes(metrics, &record);
  if (record_type == RecordType::AVERAGE_STEP) {
    // For RecordType::AVERAGE_STEP, divide by num_steps to show per-step
    // numbers when appropriate.
    int num_steps = op_stats.step_db().step_sequence_size();
    record.set_total_time_in_us(
        tsl::profiler::SafeDivide(record.total_time_in_us(), num_steps));
    record.set_total_self_time_in_us(
        tsl::profiler::SafeDivide(record.total_self_time_in_us(), num_steps));
  }
  record.set_total_time_per_core_in_us(tsl::profiler::SafeDivide(
      record.total_time_in_us(),
      op_stats.run_environment().device_core_count()));
  record.set_total_time_in_percentage(
      tsl::profiler::SafeDivide(metrics.time_ps(), total_time_ps));

  tensorflow::profiler::SetTpuUnitFractions(metrics, &record);

  // Set the roofline-specific fields.
  SetRooflineMetrics(metrics, op_stats.perf_env(), op_stats.run_environment(),
                     &record);
  const double cmem_wr_utilization =
      roofline_model_db.has_cmem()
          ? tsl::profiler::SafeDivide(record.cmem_write_bw(),
                                      roofline_model_db.peak_cmem_write_bw())
          : 0;
  const double cmem_rd_utilization =
      roofline_model_db.has_cmem()
          ? tsl::profiler::SafeDivide(record.cmem_read_bw(),
                                      roofline_model_db.peak_cmem_read_bw())
          : 0;
  const double vmem_rd_utilization =
      roofline_model_db.has_merged_vmem()
          ? tsl::profiler::SafeDivide(record.vmem_read_bw(),
                                      roofline_model_db.peak_vmem_read_bw())
          : 0;
  const double vmem_wr_utilization =
      roofline_model_db.has_merged_vmem()
          ? tsl::profiler::SafeDivide(record.vmem_write_bw(),
                                      roofline_model_db.peak_vmem_write_bw())
          : 0;
  const double flops_utilization = tsl::profiler::SafeDivide(
      record.measured_flop_rate(), roofline_model_db.peak_flop_rate());
  const double hbm_utilization = tsl::profiler::SafeDivide(
      record.hbm_bw(), roofline_model_db.peak_hbm_bw());

  const double max_mem_utilization =
      std::max({cmem_wr_utilization, cmem_rd_utilization, hbm_utilization,
                vmem_wr_utilization, vmem_rd_utilization});
  const double roofline_efficiency =
      std::max({max_mem_utilization, flops_utilization});
  // Note, copy-start/done can have utilizations above 1.0 since their
  // bytes/time are not accurate as they are asynchronous.
  record.set_optimal_flop_rate(tsl::profiler::SafeDivide(
      record.measured_flop_rate(), roofline_efficiency));
  record.set_roofline_efficiency(roofline_efficiency);
  record.set_flop_rate_relative_to_hw_limit(flops_utilization);
  record.set_memory_bw_relative_to_hw_limit(max_mem_utilization);

  record.set_include_infeed_outfeed(include_infeed_outfeed);

  return record;
}

RooflineModelRecord GenerateRooflineModelProgramRecord(
    const OpStats& op_stats, const OpMetricsDb& db, RecordType record_type,
    uint32_t step_num, const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  OpMetrics program_metrics;
  program_metrics.set_name("Program");
  program_metrics.set_category("Program");
  program_metrics.set_occurrences(1);
  uint64_t infeed_outfeed_time = 0;
  for (const OpMetrics& metrics : db.metrics_db()) {
    // Aggregate innermost ops only to avoid redundant counting.
    if (tsl::profiler::MayHaveInnerOps(metrics.category())) continue;
    if (!include_infeed_outfeed &&
        tsl::profiler::IsInfeedOrOutfeed(metrics.category())) {
      infeed_outfeed_time += metrics.time_ps();
      continue;
    }
    program_metrics.set_flops(program_metrics.flops() + metrics.flops());
    program_metrics.set_model_flops(program_metrics.model_flops() +
                                    metrics.model_flops());
    program_metrics.set_bytes_accessed(program_metrics.bytes_accessed() +
                                       metrics.bytes_accessed());
    CombineMemoryAccessedBreakdown(
        metrics.memory_accessed_breakdown(),
        program_metrics.mutable_memory_accessed_breakdown());
  }
  uint64_t total_time_ps = db.total_time_ps();
  if (!include_infeed_outfeed) total_time_ps -= infeed_outfeed_time;
  program_metrics.set_time_ps(total_time_ps);
  RooflineModelRecord program_record = ConvertOpMetricsToRooflineModelRecord(
      op_stats, program_metrics, record_type, step_num, total_time_ps,
      roofline_model_db, include_infeed_outfeed);
  program_record.set_rank(0);
  program_record.set_total_self_time_as_fraction(0.0);
  program_record.set_cumulative_total_self_time_as_fraction(0.0);
  return program_record;
}

tsl::protobuf::RepeatedPtrField<RooflineModelRecord>
ConvertOpMetricsDbToRooflineModelRecords(
    const OpStats& op_stats, const OpMetricsDb& db, RecordType record_type,
    uint32_t step_num, const RooflineModelDatabase& roofline_model_db,
    bool include_infeed_outfeed) {
  tsl::protobuf::RepeatedPtrField<RooflineModelRecord> roofline_model_records;
  RooflineModelRecord* program_record = roofline_model_records.Add();
  *program_record = GenerateRooflineModelProgramRecord(
      op_stats, db, record_type, step_num, roofline_model_db,
      include_infeed_outfeed);
  const RooflineModelRecord* prev_record = program_record;
  uint64_t infeed_outfeed_time = 0;
  if (!include_infeed_outfeed) {
    // Calculate the total time spent on infeed and outfeed ops.
    for (const OpMetrics& metrics : db.metrics_db()) {
      if (tsl::profiler::IsInfeedOrOutfeed(metrics.category())) {
        infeed_outfeed_time += metrics.time_ps();
      }
    }
  }
  uint64_t total_time_ps = db.total_time_ps() - infeed_outfeed_time;
  double total_time_us = tsl::profiler::PicoToMicro(total_time_ps);
  for (const auto* metrics : SortedOpMetricsDb(db, kMaxNumRecords)) {
    if (metrics->occurrences() == 0) continue;
    if (!include_infeed_outfeed &&
        tsl::profiler::IsInfeedOrOutfeed(metrics->category())) {
      continue;
    }
    RooflineModelRecord* record = roofline_model_records.Add();
    *record = ConvertOpMetricsToRooflineModelRecord(
        op_stats, *metrics, record_type, step_num, total_time_ps,
        roofline_model_db, include_infeed_outfeed);
    SetRankAndTimeFractions(total_time_us, *prev_record, record);
    prev_record = record;
  }
  return roofline_model_records;
}

RooflineModelDatabase InitializeRooflineModelDatabaseFromOpStats(
    const OpStats& op_stats, bool include_infeed_outfeed) {
  tensorflow::profiler::HardwareType hardware_type =
      op_stats.run_environment().hardware_type();
  DCHECK(hardware_type == GPU || hardware_type == TPU);

  RooflineModelDatabase roofline_model_db;
  const PerfEnv& perf_env = op_stats.perf_env();
  roofline_model_db.set_device_type(op_stats.run_environment().device_type());

  // Set peak flop rate in GFLOPs/s.
  roofline_model_db.set_peak_flop_rate(
      tsl::profiler::TeraToGiga((perf_env.peak_tera_flops_per_second())));
  roofline_model_db.set_peak_hbm_bw(
      tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 0)));

  if (hardware_type == HardwareType::TPU) {
    roofline_model_db.set_megacore(perf_env.has_megacore());

    roofline_model_db.set_has_cmem(perf_env.has_cmem());
    roofline_model_db.set_has_merged_vmem(perf_env.has_merged_vmem());
    if (roofline_model_db.has_cmem()) {
      roofline_model_db.set_peak_cmem_read_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 3)));
      roofline_model_db.set_peak_cmem_write_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 4)));
    } else if (roofline_model_db.has_merged_vmem()) {
      roofline_model_db.set_peak_vmem_read_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 5)));
      roofline_model_db.set_peak_vmem_write_bw(
          tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 6)));
    }
  } else if (hardware_type == HardwareType::GPU) {
    roofline_model_db.set_megacore(false);
    roofline_model_db.set_has_cmem(false);
    roofline_model_db.set_has_merged_vmem(true);
    roofline_model_db.set_peak_vmem_read_bw(
        tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 1)));
    roofline_model_db.set_peak_vmem_write_bw(
        tsl::profiler::GigaToGibi(GetMemoryPeakBandwidth(perf_env, 2)));
  }

  return roofline_model_db;
}

RooflineModelDatabase ConvertOpStatsToRooflineModel(
    const OpStats& op_stats, bool include_infeed_outfeed) {
  HardwareType hardware_type = op_stats.run_environment().hardware_type();
  if (hardware_type != GPU && hardware_type != TPU) {
    return RooflineModelDatabase();
  }

  RooflineModelDatabase roofline_model_db =
      InitializeRooflineModelDatabaseFromOpStats(op_stats,
                                                 include_infeed_outfeed);

  AddRooflineModelRecordForProfileDuration(op_stats, roofline_model_db,
                                           include_infeed_outfeed);
  AddRooflineModelRecordsForCompleteSteps(op_stats, roofline_model_db,
                                          include_infeed_outfeed);
  AddRooflineModelRecordsPerStep(op_stats, roofline_model_db,
                                 include_infeed_outfeed);
  PopulateStepDiagnostics(op_stats, roofline_model_db.mutable_diagnostics());
  return roofline_model_db;
}

}  // namespace profiler
}  // namespace tensorflow
