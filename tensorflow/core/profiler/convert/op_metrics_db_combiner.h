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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_DB_COMBINER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_DB_COMBINER_H_

#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

// Copies OpMetrics metadata (e.g., category, provenance) from src to dst.
void CopyOpMetricsMetadata(const OpMetrics& src, OpMetrics* dst);

// Combines OpMetrics data (e.g., occurrences, time) from src into dst.
void CombineOpMetrics(const OpMetrics& src, OpMetrics* dst);

// Combines the memory access breakdown.
void CombineMemoryAccessedBreakdown(
    const protobuf::RepeatedPtrField<OpMetrics_MemoryAccessed>& src,
    protobuf::RepeatedPtrField<OpMetrics_MemoryAccessed>* dst);

// Helper to combine op metrics databases.
class OpMetricsDbCombiner : public OpMetricsDbBuilder {
 public:
  explicit OpMetricsDbCombiner(OpMetricsDb* dst) : OpMetricsDbBuilder(dst) {}

  void Combine(const OpMetricsDb& src);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_DB_COMBINER_H_
