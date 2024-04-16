/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/loop.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

const Shape& GetElementShape(const HloFusionAnalysis& analysis) {
  const Shape* shape = &analysis.fusion_roots().front()->shape();
  while (shape->IsTuple()) {
    shape = &shape->tuple_shapes(0);
  }
  return *shape;
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(int64_t num_elements) {
  constexpr int kMaxUnrollFactor = 4;
  for (int i = kMaxUnrollFactor; i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }
  return 1;
}

// Determines if we enable the row optimized codegen. When we have a fusion with
// only pointwise operations, scalar broadcasting and row broadcasting, we can
// trigger a kernel that vectorizes the row loads. This speeds up the kernel, in
// particular on A100. The int is the number of inputs with rank `out_rank`. Its
// value is only defined if row vectorization is enabled.
std::pair<bool /*enabled*/, int> RowVectorizationEnabled(
    const HloFusionAdaptor& fusion, int64_t out_rank) {
  auto roots = fusion.GetRoots();
  const auto is_row_major = [](auto instr) {
    // Only tested when the inputs are row-major. So only enable that case.
    // Maybe it would work if only the inner dimensions is contiguous.
    return LayoutUtil::IsMonotonicWithDim0Major(instr.shape().layout());
  };
  bool row_vectorized = roots.size() == 1 && !roots[0].shape().IsTuple() &&
                        is_row_major(roots[0]);
  if (!row_vectorized) {
    return {false, 0};
  }

  // Check that the operations in the fusion are supported.  Each
  // supported operation (or category) must be manually vetted as XLA
  // only unrolls and relies on LLVM to vectorize. But this is brittle.
  // Currently tested and supported operations:
  // Elementwise, scalar and row broadcasting.
  //
  // We also detect at the same time if there is a row broadcasting
  // operation.
  int num_big_inputs = 0;
  bool some_row_broadcasting = false;
  HloBfsConsumersFirstTraversal(
      roots, fusion,
      [&](auto node) -> TraversalResult {
        if (!row_vectorized) {
          return TraversalResult::kInterrupt;
        }

        if (node.instruction().IsElementwise()) {
          return TraversalResult::kAdvance;
        }

        switch (node.opcode()) {
          case HloOpcode::kConstant:
            return TraversalResult::kSkip;
          case HloOpcode::kParameter:
            return TraversalResult::kAdvance;
          case HloOpcode::kBroadcast: {
            auto dims = node.instruction().dimensions();
            if (dims.empty()) {
              return TraversalResult::kAdvance;
            }

            if (dims.size() == 1 && dims.front() == node.shape().rank() - 1) {
              some_row_broadcasting = true;
              return TraversalResult::kAdvance;
            }
            TF_FALLTHROUGH_INTENDED;
          }
          default:
            VLOG(2) << "Row vectorization not enabled due to: "
                    << node.ToString();
            row_vectorized = false;
            return TraversalResult::kInterrupt;
        }
      },
      [&](auto argument) {
        if (argument.shape().rank() == out_rank) {
          ++num_big_inputs;
        }
        if (!is_row_major(argument)) {
          row_vectorized = false;
        }
      });
  // Trigger only when there is a row broadcasting.
  return std::make_pair(row_vectorized && some_row_broadcasting,
                        num_big_inputs);
}

}  // namespace

LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis) {
  int unroll_factor = 1;
  // Unrolling is good to read large inputs with small elements
  // due to vector loads, but increases the register pressure when one
  // thread has to produce multiple output elements.
  // Therefore for fusions with small outputs prefer to use one thread
  // per output element = no unroll.
  // Call 'small' fusions that use less threads than the GPU has.
  const auto& element_shape = GetElementShape(analysis);
  int64_t num_elements = ShapeUtil::ElementsIn(element_shape);
  int64_t n_threads_max = analysis.device_info().threads_per_core_limit() *
                          analysis.device_info().core_count();
  if (num_elements >= n_threads_max &&
      !MayPreventVectorization(analysis.fusion())) {
    unroll_factor = ComputeMaxUnrollFactor(num_elements);
  }
  // CHECK that unroll_factor is a power-of-2, as needed by the logic below.
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  if (analysis.input_output_info().has_4_bit_output && unroll_factor == 1) {
    // Ensure a single thread writes to a byte containing two int4 values by
    // setting unroll_factor to 2. unroll_factor is always a power of 2, so
    // setting it to 2 here ensures unroll_factor is even when there are 4-bit
    // outputs. Setting unroll_factor is safe even if there are an odd number of
    // elements, as the parallel loop emitter will insert a bounds check in this
    // case to ensure the out-of-bounds element is not computed and written.
    // Setting unroll_factor is safe even if MayPreventVectorization returns
    // false, as the MayPreventVectorization check is an optimization, not a
    // correctness requirement.
    unroll_factor = 2;
  }
  VLOG(2) << "Unroll factor: " << unroll_factor;

  bool row_vectorized;
  int num_big_inputs;
  std::tie(row_vectorized, num_big_inputs) =
      RowVectorizationEnabled(analysis.fusion(), element_shape.rank());
  bool few_waves = !HloAnyOf(
      analysis.fusion().GetRoots(), analysis.fusion(), [&](auto instr) {
        if (instr.opcode() == HloOpcode::kParameter ||
            instr.opcode() == HloOpcode::kConstant ||
            HloInstruction::IsOpElementwise(instr.opcode())) {
          return false;
        }
        if (auto broadcast =
                DynCast<HloBroadcastInstruction>(&instr.instruction())) {
          if (broadcast->dimensions().empty() ||
              // More than 3 big inputs cause a speed regression.
              (row_vectorized && num_big_inputs <= 3)) {
            return false;
          }
        }
        VLOG(2) << "few_waves not enabled due to: "
                << instr.instruction().ToString();
        return true;
      });

  LaunchDimensionsConfig launch_config{unroll_factor, few_waves,
                                       row_vectorized};
  // Check that the shapes is supported.
  if (launch_config.row_vectorized &&
      ThreadsPerBlockRowVectorized(element_shape, analysis.device_info(),
                                   launch_config) <= 0) {
    VLOG(2) << "Cancelling row_vectorization as the shape isn't supported.";
    launch_config.row_vectorized = false;
    launch_config.few_waves = false;
  }
  return launch_config;
}

LoopFusion::LoopFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis), config_(ComputeLoopFusionConfig(analysis)) {}

std::optional<IndexingMap> LoopFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  auto launch_dims = launch_dimensions();
  return GetDefaultThreadIdIndexingMap(launch_dims, config_.unroll_factor,
                                       GetElementShape(analysis_), ctx);
}

std::optional<IndexingMap> LoopFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  std::optional<IndexingMap> thread_id_to_output_indexing =
      ComputeThreadIdToOutputIndexing(root_index, ctx);
  if (!thread_id_to_output_indexing.has_value()) {
    return std::nullopt;
  }
  const HloInstruction* fusion_root = analysis_.fusion_roots()[root_index];
  auto output_to_input_indexing =
      ComputeOutputToInputIndexing(fusion_root, /*output_id=*/0, ctx);
  IndexingMapSet output_to_input_indexing_set =
      output_to_input_indexing.indexing_maps[hero_operand_index];
  // Since we are computing the indexing for a non-fusion op, there is only one
  // indexing map per operand.
  CHECK_EQ(output_to_input_indexing_set.size(), 1);
  IndexingMap thread_id_to_input_indexing_map = ComposeIndexingMaps(
      *thread_id_to_output_indexing, *output_to_input_indexing_set.begin());
  thread_id_to_input_indexing_map.Simplify(GetIndexingMapForInstruction);
  return thread_id_to_input_indexing_map;
}

absl::Status LoopFusion::EmitKernel(IrEmitterContext& ir_emitter_context,
                                    const HloFusionInstruction& fusion,
                                    const LaunchDimensions& launch_dims,
                                    std::vector<llvm_ir::IrArray> inputs,
                                    std::vector<llvm_ir::IrArray> outputs,
                                    llvm::IRBuilder<>* builder) const {
  GpuElementalIrEmitter elemental_emitter(ir_emitter_context, builder);
  FusedIrEmitter fused_emitter(elemental_emitter);
  for (int i = 0; i < fusion.fused_parameters().size(); i++) {
    fused_emitter.BindGenerator(
        *fusion.fused_parameter(i), [&, i](llvm_ir::IrArray::Index index) {
          return inputs[i].EmitReadArrayElement(index, builder);
        });
  }
  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(*fusion.fused_expression_root()));

  llvm::Type* index_type =
      GetIndexTypeForKernel(&fusion, launch_dims.launch_bound(), builder);

  return ParallelLoopEmitter(element_generator, outputs, launch_dims, builder,
                             config_)
      .EmitLoop(fusion.name(), index_type);
}

LaunchDimensions LoopFusion::launch_dimensions() const {
  return CalculateLaunchDimensions(GetElementShape(analysis_),
                                   analysis_.device_info(), config_);
}

}  // namespace gpu
}  // namespace xla
