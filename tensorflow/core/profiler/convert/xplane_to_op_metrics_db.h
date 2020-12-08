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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_OP_METRICS_DB_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_OP_METRICS_DB_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/op_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// Data per host thread for TensorFlow Op Metrics Database.
struct TfMetricsDbData {
  // The start timestamp in ps of the last infeed enqueue op on this core.
  uint64 last_infeed_enq_start_timestamp_ps = 0;
  // The duration in ps of the last infeed enqueue op on this core.
  uint64 last_infeed_enq_duration_ps = 0;

  // A database of TF-Op metrics for this core.
  OpMetricsDb tf_metrics_db;
  HostOpMetricsDbBuilder tf_metrics_db_builder{&tf_metrics_db};
};

absl::flat_hash_map<int64, TfOp> CollectTfOpsFromHostThreadsXPlane(
    const XPlane& host_trace);

TfMetricsDbData ConvertHostThreadsXLineToTfMetricsDbData(
    const XLineVisitor& line, const absl::flat_hash_map<int64, TfOp>& tf_ops);

void ConsumeTfMetricsDbData(TfMetricsDbData src, OpMetricsDbCombiner* dst);

OpMetricsDb ConvertHostThreadsXPlaneToOpMetricsDb(const XPlane& host_trace);

OpMetricsDb ConvertDeviceTraceXPlaneToOpMetricsDb(const XPlane& device_trace);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_OP_METRICS_DB_H_
