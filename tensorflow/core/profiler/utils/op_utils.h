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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_OP_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_OP_UTILS_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

class HostOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit HostOpMetricsDbBuilder(OpMetricsDb* db) : OpMetricsDbBuilder(db) {}

  // A function that will be called when the end of an OP is
  // observed on a trace, where:
  //   name = the OP name.
  //   category = the OP category.
  //   is_eager = whether this OP is eagerly executed.
  //   time_ps = the total execution time of the OP in picoseconds, including
  //             the execution time of its children.
  //   children_time_ps = the execution time of the children of this OP in
  //                      picoseconds
  void EnterOp(absl::string_view name, absl::string_view category,
               bool is_eager, uint64 time_ps, uint64 children_time_ps);

  // Updates total_host_infeed_enq_duration_ps_ and
  // total_host_infeed_enq_duration_ps_.
  void EnterHostInfeedEnqueue(tsl::profiler::Timespan host_infeed_enqueue);

 private:
  // The tsl::profiler::Timespan of the last InfeedEnqueue op on this thread.
  tsl::profiler::Timespan last_host_infeed_enqueue_;
};

class DeviceOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit DeviceOpMetricsDbBuilder(OpMetricsDb* db) : OpMetricsDbBuilder(db) {}

  // A function that will be called when the end of an OP is
  // observed on a trace, where:
  //   program_id = the ID of the program that contains this OP.
  //   name = the OP name.
  //   category = the OP category.
  //   provenance = the provenance of this OP (e.g. original TF OP).
  //   is_eager = whether this OP is eagerly executed.
  //   occurrences = the number of occurrences of this OP.
  //   time_ps = the total execution time of the OP in picoseconds, including
  //             the execution time of its children.
  //   children_time_ps = the execution time of the children of this OP in
  //                      picoseconds.
  //   flops = the number of floating-point operations computed.
  //   bytes_accessed = the sum of bytes read and bytes written by this OP.
  //   memory_accessed_breakdown = the breakdown of memory accessed by operation
  //                               type and memory space.
  void EnterOp(uint64 program_id, absl::string_view name,
               absl::string_view category, absl::string_view provenance,
               bool is_eager, uint64 occurrences, uint64 time_ps,
               uint64 children_time_ps, int64_t flops, int64_t bytes_accessed,
               const protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>&
                   memory_accessed_breakdown = {},
               int64_t model_flops = 0);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_OP_UTILS_H_
