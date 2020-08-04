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

#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using OperationType = OpMetrics::MemoryAccessed::OperationType;

// Combines the src OpMetrics into the dst OpMetrics.
void CombineOpMetrics(const OpMetrics& src, OpMetrics* dst) {
  DCHECK(dst != nullptr);
  DCHECK_EQ(src.hlo_module_id(), dst->hlo_module_id());
  DCHECK_EQ(src.name(), dst->name());
  dst->set_category(src.category());
  dst->set_provenance(src.provenance());
  dst->set_is_eager(dst->is_eager() || src.is_eager());
  dst->set_deduplicated_name(src.deduplicated_name());
  if (!dst->has_layout() && src.has_layout()) {
    *dst->mutable_layout() = src.layout();
  }
  if (!dst->has_children() && src.has_children()) {
    *dst->mutable_children() = src.children();
  }
  dst->set_occurrences(src.occurrences() + dst->occurrences());
  dst->set_time_ps(src.time_ps() + dst->time_ps());
  dst->set_self_time_ps(src.self_time_ps() + dst->self_time_ps());
  dst->set_flops(src.flops() + dst->flops());
  dst->set_bytes_accessed(src.bytes_accessed() + dst->bytes_accessed());
  CombineMemoryAccessedBreakdown(src.memory_accessed_breakdown(),
                                 dst->mutable_memory_accessed_breakdown());
  dst->set_dma_stall_ps(src.dma_stall_ps() + dst->dma_stall_ps());
}

void CombinePrecisionStats(const PrecisionStats& src, PrecisionStats* dst) {
  dst->set_compute_16bit_ps(src.compute_16bit_ps() + dst->compute_16bit_ps());
  dst->set_compute_32bit_ps(src.compute_32bit_ps() + dst->compute_32bit_ps());
}

}  // namespace

void CombineMemoryAccessedBreakdown(
    const protobuf::RepeatedPtrField<OpMetrics_MemoryAccessed>& src,
    protobuf::RepeatedPtrField<OpMetrics_MemoryAccessed>* dst) {
  absl::flat_hash_map<std::pair<uint64 /*memory_space*/, OperationType>,
                      OpMetrics_MemoryAccessed*>
      dst_memory_accessed_map;
  for (auto& dst_memory_accessed : *dst) {
    dst_memory_accessed_map[{dst_memory_accessed.memory_space(),
                             dst_memory_accessed.operation_type()}] =
        &dst_memory_accessed;
  }
  for (const auto& src_memory_accessed : src) {
    uint64 memory_space = src_memory_accessed.memory_space();
    OperationType operation_type = src_memory_accessed.operation_type();
    auto*& dst_memory_accessed =
        dst_memory_accessed_map[{memory_space, operation_type}];
    if (dst_memory_accessed == nullptr) {
      dst_memory_accessed = dst->Add();
      dst_memory_accessed->set_memory_space(memory_space);
      dst_memory_accessed->set_operation_type(operation_type);
    }
    dst_memory_accessed->set_bytes_accessed(
        src_memory_accessed.bytes_accessed() +
        dst_memory_accessed->bytes_accessed());
  }
}

void OpMetricsDbCombiner::Combine(const OpMetricsDb& src) {
  OpMetricsDb* dst = db();
  dst->set_total_host_infeed_enq_duration_ps(
      src.total_host_infeed_enq_duration_ps() +
      dst->total_host_infeed_enq_duration_ps());
  dst->set_total_host_infeed_enq_start_timestamp_ps_diff(
      src.total_host_infeed_enq_start_timestamp_ps_diff() +
      dst->total_host_infeed_enq_start_timestamp_ps_diff());
  dst->set_total_time_ps(src.total_time_ps() + dst->total_time_ps());
  dst->set_total_op_time_ps(src.total_op_time_ps() + dst->total_op_time_ps());
  CombinePrecisionStats(src.precision_stats(), dst->mutable_precision_stats());

  for (const auto& src_metrics : src.metrics_db()) {
    auto* dst_metrics = LookupOrInsertNewOpMetrics(src_metrics.hlo_module_id(),
                                                   src_metrics.name());
    CombineOpMetrics(src_metrics, dst_metrics);
  }
}

}  // namespace profiler
}  // namespace tensorflow
