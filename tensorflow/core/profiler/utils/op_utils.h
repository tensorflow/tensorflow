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
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

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
  void UpdateHostInfeedEnqInfo(uint64 duration_ps,
                               uint64 start_timestamp_ps_diff);
};

class DeviceOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit DeviceOpMetricsDbBuilder(OpMetricsDb* db,
                                    double peak_tera_flops_per_second,
                                    double peak_hbm_bw_giga_bytes_per_second)
      : OpMetricsDbBuilder(db),
        peak_tera_flops_per_second_(peak_tera_flops_per_second),
        peak_hbm_bw_giga_bytes_per_second_(peak_hbm_bw_giga_bytes_per_second) {}

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
  void EnterOp(uint64 program_id, absl::string_view name,
               absl::string_view category, absl::string_view provenance,
               bool is_eager, uint64 occurrences, uint64 time_ps,
               uint64 children_time_ps, int64 flops, int64 bytes_accessed);

 protected:
  // Peak performance of a TensorCore or a GPU in TFLOP/s.
  double peak_tera_flops_per_second_;
  // Peak memory bandwidth of a TensorCore or a GPU in GiBs/s.
  double peak_hbm_bw_giga_bytes_per_second_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_OP_UTILS_H_
