/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_indexing_performance_model.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

int64_t GpuPerformanceModelWithIndexingAnalysis::FlopsPerElement(
    const HloInstruction* instr) const {
  // TODO(shyshkov): Replace dependency on GpuHloCostAnalysis with independent
  // flops calculation.
  GpuHloCostAnalysis::Options cost_analysis_options{
      shape_size_,
      /*per_second_rates=*/{},
      /*count_multiple_input_accesses=*/true};
  GpuHloCostAnalysis cost_analysis(cost_analysis_options, device_info_);
  TF_CHECK_OK(
      cost_analysis.RevisitInstruction(const_cast<HloInstruction*>(instr)));

  int64_t num_elements = [&] {
    if (instr->opcode() == HloOpcode::kReduce && instr->shape().IsTuple()) {
      return ShapeUtil::ElementsInRecursive(instr->shape().tuple_shapes(0));
    }
    return ShapeUtil::ElementsInRecursive(instr->shape());
  }();

  return cost_analysis.flop_count(*instr) / num_elements;
}

int64_t GpuPerformanceModelWithIndexingAnalysis::GetShapeSizeRecursive(
    const Shape& shape) const {
  CHECK(shape.IsArray() || shape.IsTuple());
  if (shape.IsArray()) {
    return shape_size_(shape);
  }

  int64_t total_size = 0;
  for (const auto& element_shape : shape.tuple_shapes()) {
    total_size += GetShapeSizeRecursive(element_shape);
  }
  return total_size;
}

int64_t GetIterationSpaceSize(const IndexingMap& indexing_map,
                              const HloInstruction* instr) {
  if (indexing_map.IsUndefined()) {
    return ShapeUtil::ElementsInRecursive(instr->shape());
  }

  if (indexing_map.IsKnownEmpty()) {
    return 0;
  }

  auto get_ranges_iteration_space_size =
      [](const std::vector<Interval>& ranges) {
        int64_t num_iters = 1;
        for (const Interval& range : ranges) {
          num_iters *= range.upper - range.lower + 1;
        }
        return num_iters;
      };

  return get_ranges_iteration_space_size(indexing_map.GetSymbolBounds()) *
         get_ranges_iteration_space_size(indexing_map.GetDimensionBounds());
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForFusion(
    const HloFusionAnalysis& fusion_analysis, bool is_coalesced) {
  auto& fusion_adaptor = fusion_analysis.fusion();
  VLOG(5) << "EstimateRunTimeForFusion: " << fusion_adaptor.ToString();

  auto roots = fusion_adaptor.GetRoots();
  CHECK_EQ(roots.size(), 1)
      << "Indexing cost model doesn't support multi-output fusions.";
  auto root_shape = roots.front().shape();

  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(ShapeUtil::ElementsInRecursive(root_shape),
                                     fusion_analysis, *device_info_);

  int64_t num_threads = launch_dimensions.launch_bound();
  int64_t num_blocks = launch_dimensions.num_blocks();

  // Compute indexing from root to each instruction in the fusion and fusion
  // operands. For each instruction, tells which elements of the instructions
  // result will be used to compute one result element of the fusion.
  auto grouped_fusion_indexing = ComputeGroupedOutputToInputIndexing(
      fusion_adaptor, roots[0], mlir_context_);

  int64_t flops = 0;
  int64_t bytes_read = 0;
  absl::Duration read_time = absl::ZeroDuration();

  for (const auto& [instr, indexing_maps] : grouped_fusion_indexing) {
    VLOG(10) << "instr: " << instr->name();
    HloInstructionAdaptor instr_adaptor(*instr);

    // Instructions inside the fusion are computation and account for FLOPs
    // count. Instructions outside the fusion are operands of the fusion and
    // account for memory read time.
    bool is_operand = !fusion_adaptor.ContainsInstruction(instr_adaptor);

    auto element_type = instr->shape().element_type();
    int64_t n_bytes_total = 0;
    for (const auto& indexing_map : indexing_maps) {
      VLOG(10) << indexing_map.ToString();

      int64_t num_iters = GetIterationSpaceSize(indexing_map, instr);

      if (is_operand) {
        int64_t type_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
        n_bytes_total += type_size * num_iters;
      } else {
        int64_t flops_per_element = FlopsPerElement(instr);
        flops += flops_per_element * num_iters;
      }
    }

    if (is_operand) {
      int64_t operand_size = shape_size_(instr->shape());
      int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
      bytes_read += n_bytes_total;

      VLogOperandRead(instr, n_bytes_total, n_bytes_net, is_coalesced);

      read_time +=
          ReadTimeWithDRAMHeuristic(*device_info_, num_blocks, n_bytes_net,
                                    n_bytes_total, element_type, is_coalesced);
    }
  }

  int64_t bytes_written = GetShapeSizeRecursive(root_shape);

  absl::Duration compute_time = ComputeTime(*device_info_, flops, num_threads);
  absl::Duration write_time = WriteTime(*device_info_, bytes_written);
  absl::Duration memory_access_time = read_time + write_time;
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, memory_access_time,
      GpuPerformanceModelOptions::PriorityFusion());

  VLogResult(flops, bytes_read, bytes_written, num_threads, compute_time,
             read_time, write_time, exec_time);

  return EstimateRunTimeData{flops,      bytes_written, num_threads, read_time,
                             write_time, compute_time,  exec_time};
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForInstruction(
    const HloInstruction* producer) {
  // Stand-alone bitcast is always no-op during runtime.
  if (producer->opcode() == HloOpcode::kBitcast) {
    return {0, 0, 0, absl::ZeroDuration(), absl::ZeroDuration()};
  }

  auto fusion_analysis = AnalyzeFusion(*producer, *device_info_);

  bool is_coalesced = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForProducerConsumer(
    const HloInstruction* producer, const HloInstruction* consumer) {
  auto fusion_analysis =
      AnalyzeProducerConsumerFusion(*producer, *consumer, *device_info_);

  bool is_coalesced = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer, consumer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}

/*static*/
GpuPerformanceModelWithIndexingAnalysis::RunTimes
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimes(
    const HloInstruction* producer,
    absl::Span<const HloInstruction* const> fused_consumers) {
  auto producer_runtime = EstimateRunTimeForInstruction(producer);

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();

  for (const auto& consumer : fused_consumers) {
    time_unfused += EstimateRunTimeForInstruction(consumer).exec_time;
    time_fused +=
        EstimateRunTimeForProducerConsumer(producer, consumer).exec_time;
  }

  return {time_unfused, time_fused};
}

}  // namespace gpu
}  // namespace xla
