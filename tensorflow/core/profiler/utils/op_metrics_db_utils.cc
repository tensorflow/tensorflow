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

#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

#include <algorithm>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"

namespace tensorflow {
namespace profiler {

const absl::string_view kIdle = "IDLE";

namespace {

class DeviceTfOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit DeviceTfOpMetricsDbBuilder(OpMetricsDb* db)
      : OpMetricsDbBuilder(db) {}

  void UpdateTfOpMetricsWithDeviceOpMetrics(
      absl::string_view tf_op_name, absl::string_view tf_op_type,
      const OpMetrics& device_op_metrics) {
    OpMetrics* tf_op_metrics = OpMetricsDbBuilder::LookupOrInsertNewOpMetrics(
        /*hlo_module_id=*/0, tf_op_name);
    if (tf_op_metrics->category().empty()) {
      tf_op_metrics->set_category(
          tf_op_type == kUnknownOp ? "Unknown" : std::string(tf_op_type));
    }
    tf_op_metrics->set_is_eager(device_op_metrics.is_eager());
    // The occurrences of a TF-op is the maximum among the occurrences of all
    // device ops that it contains.
    tf_op_metrics->set_occurrences(std::max(tf_op_metrics->occurrences(),
                                            device_op_metrics.occurrences()));
    tf_op_metrics->set_time_ps(tf_op_metrics->time_ps() +
                               device_op_metrics.time_ps());
    tf_op_metrics->set_self_time_ps(tf_op_metrics->self_time_ps() +
                                    device_op_metrics.self_time_ps());
    tf_op_metrics->set_flops(tf_op_metrics->flops() +
                             device_op_metrics.flops());
    tf_op_metrics->set_bytes_accessed(tf_op_metrics->bytes_accessed() +
                                      device_op_metrics.bytes_accessed());
  }
};

}  // namespace

OpMetricsDbBuilder::OpMetricsDbBuilder(OpMetricsDb* db) : db_(db) {
  DCHECK_NE(db_, nullptr);
  DCHECK_EQ(db_->metrics_db_size(), 0);
}

OpMetrics* OpMetricsDbBuilder::LookupOrInsertNewOpMetrics(
    uint64 hlo_module_id, absl::string_view name) {
  OpMetrics*& op_metrics = op_metrics_map_[hlo_module_id][name];
  if (op_metrics == nullptr) {
    op_metrics = db_->add_metrics_db();
    op_metrics->set_hlo_module_id(hlo_module_id);
    op_metrics->set_name(name.data(), name.size());
  }
  return op_metrics;
}

double IdleTimeRatio(const OpMetricsDb& db) {
  return 1.0 - SafeDivide(db.total_op_time_ps(), db.total_time_ps());
}

uint64 IdleTimePs(const OpMetricsDb& db) {
  DCHECK_GE(db.total_time_ps(), db.total_op_time_ps());
  return db.total_time_ps() - db.total_op_time_ps();
}

void SetIdleOp(uint64_t idle_time_ps, OpMetrics& metrics) {
  metrics.set_name(std::string(kIdle));
  metrics.set_category(std::string(kIdle));
  metrics.set_occurrences(0);
  metrics.set_time_ps(idle_time_ps);
  metrics.set_self_time_ps(idle_time_ps);
}

void AddIdleOp(OpMetricsDb& db) {
  uint64 idle_time_ps = IdleTimePs(db);
  SetIdleOp(idle_time_ps, *db.add_metrics_db());
}

absl::optional<double> HostInfeedEnqueueRatio(const OpMetricsDb& db) {
  if (db.total_host_infeed_enq_start_timestamp_ps_diff() > 0) {
    // We use total_host_infeed_enq_start_timestamp_ps_diff to approximate the
    // total host time.
    return SafeDivide(db.total_host_infeed_enq_duration_ps(),
                      db.total_host_infeed_enq_start_timestamp_ps_diff());
  }
  return absl::nullopt;
}

OpMetricsDb CreateTfMetricsDbFromDeviceOpMetricsDb(
    const OpMetricsDb& device_op_metrics_db, bool with_idle) {
  OpMetricsDb tf_op_metrics_db;
  DeviceTfOpMetricsDbBuilder builder(&tf_op_metrics_db);
  for (const auto& device_op_metrics : device_op_metrics_db.metrics_db()) {
    if (IsIdleOp(device_op_metrics)) {
      if (with_idle) {
        builder.UpdateTfOpMetricsWithDeviceOpMetrics(kIdle, kIdle,
                                                     device_op_metrics);
      }
    } else if (device_op_metrics.provenance().empty()) {
      builder.UpdateTfOpMetricsWithDeviceOpMetrics(
          device_op_metrics.name(), kUnknownOp, device_op_metrics);
    } else {
      TfOp tf_op = ParseTfOpFullname(device_op_metrics.provenance());
      builder.UpdateTfOpMetricsWithDeviceOpMetrics(tf_op.name, tf_op.type,
                                                   device_op_metrics);
    }
  }
  tf_op_metrics_db.set_total_op_time_ps(
      device_op_metrics_db.total_op_time_ps());

  tf_op_metrics_db.set_total_time_ps(
      with_idle ? device_op_metrics_db.total_time_ps()
                : device_op_metrics_db.total_op_time_ps());

  return tf_op_metrics_db;
}

}  // namespace profiler
}  // namespace tensorflow
