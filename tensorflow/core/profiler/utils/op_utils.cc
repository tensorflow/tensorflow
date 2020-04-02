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

#include "tensorflow/core/profiler/utils/op_utils.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

// Return capped performance. If time == 0, returns the original perf.
// Otherwise, returns the minimum of perf and the product of rate_limit
// and time.
double GetCappedPerf(double perf, uint64 time, double rate_limit) {
  if (perf <= 0) return 0;
  if (time == 0) return perf;
  return std::min(perf, time * rate_limit);
}

}  // namespace

void HostOpMetricsDbBuilder::EnterOp(absl::string_view name,
                                     absl::string_view category, uint64 time_ps,
                                     uint64 children_time_ps) {
  uint64 self_time_ps = time_ps - children_time_ps;
  DCHECK_GE(time_ps, self_time_ps);
  OpMetrics* op_metrics = LookupOrInsertNewOpMetrics(/*hlo_module_id=*/0, name);
  if (op_metrics->category().empty())
    op_metrics->set_category(category.data(), category.size());
  op_metrics->set_occurrences(op_metrics->occurrences() + 1);
  op_metrics->set_time_ps(op_metrics->time_ps() + time_ps);
  op_metrics->set_self_time_ps(op_metrics->self_time_ps() + self_time_ps);
  db()->set_total_op_time_ps(db()->total_op_time_ps() + self_time_ps);
}

void HostOpMetricsDbBuilder::UpdateHostInfeedEnqInfo(
    uint64 duration_ps, uint64 start_timestamp_ps_diff) {
  db()->set_total_host_infeed_enq_duration_ps(
      db()->total_host_infeed_enq_duration_ps() + duration_ps);
  db()->set_total_host_infeed_enq_start_timestamp_ps_diff(
      db()->total_host_infeed_enq_start_timestamp_ps_diff() +
      start_timestamp_ps_diff);
}

void DeviceOpMetricsDbBuilder::EnterOp(
    uint64 program_id, absl::string_view name, absl::string_view category,
    absl::string_view provenance, uint64 occurrences, uint64 time_ps,
    uint64 children_time_ps, int64 flops, int64 bytes_accessed) {
  uint64 self_time_ps = time_ps - children_time_ps;
  DCHECK_GE(time_ps, self_time_ps);
  OpMetrics* op_metrics = LookupOrInsertNewOpMetrics(program_id, name);
  if (op_metrics->category().empty())
    op_metrics->set_category(category == kUnknownOp ? "unknown"
                                                    : string(category));
  if (op_metrics->provenance().empty())
    op_metrics->set_provenance(string(provenance));
  op_metrics->set_occurrences(op_metrics->occurrences() + occurrences);
  op_metrics->set_time_ps(op_metrics->time_ps() + time_ps);
  op_metrics->set_self_time_ps(op_metrics->self_time_ps() + self_time_ps);
  op_metrics->set_flops(op_metrics->flops() +
                        GetCappedPerf(flops * occurrences, self_time_ps,
                                      peak_tera_flops_per_second_));
  op_metrics->set_bytes_accessed(
      op_metrics->bytes_accessed() +
      GetCappedPerf(bytes_accessed * occurrences, self_time_ps,
                    peak_hbm_bw_giga_bytes_per_second_ / 1000));
  db()->set_total_op_time_ps(db()->total_op_time_ps() + self_time_ps);
}

}  // namespace profiler
}  // namespace tensorflow
