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

#include "tensorflow/core/profiler/utils/op_utils.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

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
}  // namespace profiler
}  // namespace tensorflow
