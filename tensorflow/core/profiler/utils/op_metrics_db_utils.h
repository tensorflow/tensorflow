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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_OP_METRICS_DB_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_OP_METRICS_DB_UTILS_H_

#include <algorithm>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

// The name of OpMetrics to represent the idle time.
TF_CONST_INIT extern const absl::string_view kIdle;

// Helps build an op metrics database (borrowed).
// Enables fast lookup of existing ops and prevents the creation of duplicate
// ops. It is the user's responsibility to ensure an op metrics database
// outlives its builder, and that no ops are added to the database outside of
// the builder.
class OpMetricsDbBuilder {
 public:
  // Create with a borrowed op database.
  // REQUIRED: The op database must be empty.
  explicit OpMetricsDbBuilder(OpMetricsDb* db);

 protected:
  // Looks up the given OP name. If it is already in the database,
  // return its OpMetrics; otherwise, insert a new one.
  OpMetrics* LookupOrInsertNewOpMetrics(uint64 hlo_module_id,
                                        absl::string_view name);

  OpMetricsDb* db() { return db_; }

 private:
  // Map op (hlo_module_id, name) to the corresponding metrics in the op
  // database.
  absl::flat_hash_map<uint64 /*hlo_module_id*/,
                      absl::flat_hash_map<std::string /*name*/, OpMetrics*>>
      op_metrics_map_;

  // The op database.
  OpMetricsDb* db_;
};

// Sets the total time for OpMetricsDb, ensuring idle time is not negative.
inline void SetTotalTimePs(OpMetricsDb& db, uint64_t total_time_ps) {
  db.set_total_time_ps(std::max(db.total_op_time_ps(), total_time_ps));
}

// Returns the total time in OpMetricsDb, optionally excluding the idle time.
inline uint64_t TotalTimePs(const OpMetricsDb& db, bool exclude_idle = false) {
  return exclude_idle ? db.total_op_time_ps() : db.total_time_ps();
}

// Returns the ratio of time that is idle (no op execution) over total time.
double IdleTimeRatio(const OpMetricsDb& db);

// Returns the idle time in picoseconds.
uint64 IdleTimePs(const OpMetricsDb& db);

// Populates an OpMetrics record representing idle time, i.e., the amount of
// time spent without any op execution.
void SetIdleOp(uint64_t idle_time_ps, OpMetrics& metrics);

// Adds an OpMetrics record representing idle time, i.e., the amount of time
// spent without any op execution.
// REQUIRED: All ops must have been added to the database and the total time
// must have been set.
void AddIdleOp(OpMetricsDb& db);

// Returns true if the given metrics represents idle time.
inline bool IsIdleOp(const OpMetrics& metrics) {
  return metrics.category() == kIdle;
}

// Returns the time spent in children (nested) ops.
inline uint64_t ChildrenTimePs(const OpMetrics& metrics) {
  return metrics.time_ps() - metrics.self_time_ps();
}

// Returns the ratio of time spent sending data from the host to the device
// relative to the total time the host was active.
absl::optional<double> HostInfeedEnqueueRatio(const OpMetricsDb& db);

// Converts from the device op metrics to Tf-op metrics.
OpMetricsDb CreateTfMetricsDbFromDeviceOpMetricsDb(
    const OpMetricsDb& device_op_metrics_db, bool with_idle = true);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_OP_METRICS_DB_UTILS_H_
