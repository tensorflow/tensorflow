/* Copyright 2026 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/thread_block_specialized.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialectDecl.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "xla/backends/gpu/codegen/emitters/concatenate.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/reduction.h"
#include "xla/backends/gpu/codegen/emitters/scatter.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/emitters/transpose.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/nested_kernel_utils.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/util.h"
#include "tsl/platform/platform.h"

namespace xla::gpu {
namespace {

using NestedKernelSpec = ThreadBlockSpecializedFusion::NestedKernelSpec;
using LeafDecls =
    absl::flat_hash_map<const HloFusionInstruction*, mlir::LLVM::LLVMFuncOp>;

std::vector<const NestedKernelSpec*> CollectLeafSpecs(
    const NestedKernelSpec& spec) {
  std::vector<const NestedKernelSpec*> leaf_specs;
  auto collect = [&](auto& self, const NestedKernelSpec& node) -> void {
    if (node.kind == NestedKernelSpec::Kind::kLeaf) {
      leaf_specs.push_back(&node);
      return;
    }
    for (const NestedKernelSpec& child : node.children) {
      self(self, child);
    }
  };
  collect(collect, spec);
  return leaf_specs;
}

void EmitNode(mlir::ImplicitLocOpBuilder& b, const NestedKernelSpec& spec,
              mlir::Value rel_bid, mlir::Value tid, int64_t active_blocks,
              int64_t active_threads, absl::Span<const mlir::Value> arg_ptrs,
              const LeafDecls& leaf_decls);

void EmitLeaf(mlir::ImplicitLocOpBuilder& b, const NestedKernelSpec& spec,
              mlir::Value rel_bid, mlir::Value tid, int64_t active_blocks,
              int64_t active_threads, absl::Span<const mlir::Value> arg_ptrs,
              const LeafDecls& leaf_decls) {
  int64_t leaf_blocks = spec.launch_dims.num_blocks();
  if (leaf_blocks < active_blocks) {
    mlir::Value block_bound =
        mlir::arith::ConstantIntOp::create(b, leaf_blocks, 32);
    mlir::Value block_active = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::ult, rel_bid, block_bound);
    auto block_if =
        mlir::scf::IfOp::create(b, block_active, /*withElseRegion=*/false);
    b.setInsertionPointToStart(&block_if.getThenRegion().front());
  }

  int64_t leaf_threads = spec.launch_dims.num_threads_per_block();
  if (leaf_threads < active_threads) {
    mlir::Value tid_bound =
        mlir::arith::ConstantIntOp::create(b, leaf_threads, 32);
    mlir::Value tid_active = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::ult, tid, tid_bound);
    auto tid_if =
        mlir::scf::IfOp::create(b, tid_active, /*withElseRegion=*/false);
    b.setInsertionPointToStart(&tid_if.getThenRegion().front());
  }

  llvm::SmallVector<mlir::Value> call_args;
  call_args.reserve(spec.input_arg_indices.size() +
                    spec.output_arg_indices.size() + 1);
  for (int64_t arg_idx : spec.input_arg_indices) {
    call_args.push_back(arg_ptrs[arg_idx]);
  }
  for (int64_t arg_idx : spec.output_arg_indices) {
    call_args.push_back(arg_ptrs[arg_idx]);
  }
  call_args.push_back(rel_bid);
  mlir::LLVM::CallOp::create(b, leaf_decls.at(spec.fusion), call_args);
}

void EmitHorizontal(mlir::ImplicitLocOpBuilder& b, const NestedKernelSpec& spec,
                    mlir::Value rel_bid, mlir::Value tid,
                    int64_t active_threads,
                    absl::Span<const mlir::Value> arg_ptrs,
                    const LeafDecls& leaf_decls) {
  int64_t cluster_x = spec.cluster_dim.has_value()
                          ? static_cast<int64_t>(spec.cluster_dim->x)
                          : 1;
  int64_t block_offset = 0;
  for (size_t idx = 0; idx < spec.children.size(); ++idx) {
    const NestedKernelSpec& child = spec.children[idx];
    int64_t child_blocks = child.launch_dims.num_blocks();
    int64_t allocated_blocks = child.cluster_dim.has_value()
                                   ? RoundUpTo<int64_t>(child_blocks, cluster_x)
                                   : child_blocks;
    int64_t upper_bound = block_offset + allocated_blocks;
    mlir::Value bound_val =
        mlir::arith::ConstantIntOp::create(b, upper_bound, 32);
    mlir::Value in_range = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::ult, rel_bid, bound_val);

    bool has_else = (idx + 1 < spec.children.size());
    auto if_op =
        mlir::scf::IfOp::create(b, in_range, /*withElseRegion=*/has_else);
    b.setInsertionPointToStart(&if_op.getThenRegion().front());

    mlir::Value child_rel_bid = rel_bid;
    if (block_offset != 0) {
      mlir::Value offset_val =
          mlir::arith::ConstantIntOp::create(b, block_offset, 32);
      child_rel_bid = mlir::arith::SubIOp::create(b, rel_bid, offset_val);
    }
    block_offset = upper_bound;

    EmitNode(b, child, child_rel_bid, tid, allocated_blocks, active_threads,
             arg_ptrs, leaf_decls);

    if (has_else) {
      b.setInsertionPointToStart(&if_op.getElseRegion().front());
    }
  }
}

void EmitPhased(mlir::ImplicitLocOpBuilder& b, const NestedKernelSpec& spec,
                mlir::Value rel_bid, mlir::Value tid, int64_t active_blocks,
                int64_t active_threads, absl::Span<const mlir::Value> arg_ptrs,
                const LeafDecls& leaf_decls) {
  mlir::OpBuilder::InsertPoint caller_ip = b.saveInsertionPoint();
  auto is_single_block = [](const NestedKernelSpec& child) {
    return child.launch_dims.num_blocks() == 1 &&
           !child.cluster_dim.has_value();
  };
  // Contiguous single-block phases execute exclusively on relative block 0
  // (`rel_bid < 1`) and share a single block-range guard so block barriers
  // synchronize them without flushing L1 via a cluster barrier.
  mlir::Value c0 = mlir::arith::ConstantIntOp::create(b, 0, 32);
  std::optional<mlir::scf::IfOp> single_block_if;
  for (size_t idx = 0; idx < spec.children.size(); ++idx) {
    const NestedKernelSpec& child = spec.children[idx];
    const bool child_single_block = is_single_block(child);
    if (child_single_block) {
      if (active_blocks > 1) {
        // While block 0 executes single-block phases, all other blocks in the
        // cluster skip ahead avoiding any synchronization between phases.
        if (!single_block_if.has_value()) {
          b.restoreInsertionPoint(caller_ip);
          mlir::Value c1 = mlir::arith::ConstantIntOp::create(b, 1, 32);
          mlir::Value in_block0 = mlir::arith::CmpIOp::create(
              b, mlir::arith::CmpIPredicate::ult, rel_bid, c1);
          single_block_if =
              mlir::scf::IfOp::create(b, in_block0, /*withElseRegion=*/false);
        }
        b.setInsertionPoint(single_block_if->thenYield());
      } else {
        b.restoreInsertionPoint(caller_ip);
      }
      EmitNode(b, child, c0, tid, /*active_blocks=*/1, active_threads, arg_ptrs,
               leaf_decls);
    } else {
      single_block_if.reset();
      b.restoreInsertionPoint(caller_ip);
      EmitNode(b, child, rel_bid, tid, active_blocks, active_threads, arg_ptrs,
               leaf_decls);
    }

    if (idx + 1 < spec.children.size()) {
      if (child_single_block && is_single_block(spec.children[idx + 1])) {
        // Both phases run exclusively on relative block 0. Emit a block barrier
        // outside any child `tid` guard so every active thread of block 0
        // participates.
        mlir::gpu::BarrierOp::create(b);
      } else {
        // Emit a cluster barrier at `caller_ip` level outside all child block
        // and thread `if` guards so every thread of every block in the cluster
        // participates and phase `idx` writes become visible across the
        // cluster.
        single_block_if.reset();
        b.restoreInsertionPoint(caller_ip);
        mlir::NVVM::ClusterArriveOp::create(b, b.getUnitAttr());
        mlir::NVVM::ClusterWaitOp::create(b, b.getUnitAttr());
      }
    }
  }
}

void EmitNode(mlir::ImplicitLocOpBuilder& b, const NestedKernelSpec& spec,
              mlir::Value rel_bid, mlir::Value tid, int64_t active_blocks,
              int64_t active_threads, absl::Span<const mlir::Value> arg_ptrs,
              const LeafDecls& leaf_decls) {
  mlir::OpBuilder::InsertionGuard guard(b);

  switch (spec.kind) {
    case NestedKernelSpec::Kind::kLeaf:
      EmitLeaf(b, spec, rel_bid, tid, active_blocks, active_threads, arg_ptrs,
               leaf_decls);
      break;
    case NestedKernelSpec::Kind::kHorizontal:
      EmitHorizontal(b, spec, rel_bid, tid, active_threads, arg_ptrs,
                     leaf_decls);
      break;
    case NestedKernelSpec::Kind::kPhased:
      EmitPhased(b, spec, rel_bid, tid, active_blocks, active_threads, arg_ptrs,
                 leaf_decls);
      break;
  }
}

NestedKernelSpec BuildNestedKernelSpec(const HloFusionAnalysis& analysis,
                                       std::vector<int64_t> input_arg_indices,
                                       std::vector<int64_t> output_arg_indices,
                                       bool unroll_small_fusions) {
  NestedKernelSpec spec;
  spec.input_arg_indices = std::move(input_arg_indices);
  spec.output_arg_indices = std::move(output_arg_indices);

  if (std::unique_ptr<MlirKernelEmitter> emitter =
          ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(
              analysis, /*unroll_small_fusions=*/unroll_small_fusions)) {
    spec.kind = NestedKernelSpec::Kind::kLeaf;
    spec.launch_dims = emitter->launch_dimensions();
    spec.unroll_small_fusions = unroll_small_fusions;
    return spec;
  }

  absl::string_view fusion_kind = analysis.fusion_backend_config().kind();
  if (fusion_kind == kThreadBlockSpecializedFusionKind) {
    spec.kind = NestedKernelSpec::Kind::kHorizontal;
  } else if (fusion_kind == kPhasedBlockClusterFusionKind) {
    spec.kind = NestedKernelSpec::Kind::kPhased;
  } else {
    LOG(FATAL) << "Unsupported nested fusion kind: " << fusion_kind;
  }

  std::vector<const HloFusionInstruction*> child_fusions;
  absl::flat_hash_map<const HloInstruction*, std::vector<int64_t>>
      fusion_to_child_output_arg_indices;

  for (int64_t root_idx = 0; root_idx < analysis.fusion_root_count();
       ++root_idx) {
    const HloInstruction* inst = &analysis.fusion_root(root_idx).instruction();
    CHECK(!inst->shape().IsTuple())
        << "Nested fusion tuple output must be unpacked via get-tuple-element: "
        << inst->ToString();
    int64_t tuple_idx = 0;
    if (inst->opcode() == HloOpcode::kGetTupleElement) {
      tuple_idx = inst->tuple_index();
      inst = inst->operand(0);
    }
    const auto* child_fusion = Cast<HloFusionInstruction>(inst);
    auto [it, inserted] =
        fusion_to_child_output_arg_indices.try_emplace(child_fusion);
    if (inserted) {
      child_fusions.push_back(child_fusion);
      it->second.assign(ShapeUtil::GetLeafCount(child_fusion->shape()), -1);
    }
    CHECK_LT(tuple_idx, it->second.size());
    CHECK_EQ(it->second[tuple_idx], -1)
        << "Duplicate root referencing output " << tuple_idx << " of "
        << child_fusion->ToString();
    it->second[tuple_idx] = spec.output_arg_indices[root_idx];
  }

  if (spec.kind == NestedKernelSpec::Kind::kPhased) {
    absl::flat_hash_map<const HloInstruction*, int64_t> post_order_indices;
    int64_t idx = 0;
    for (const HloInstructionAdaptor& instr :
         analysis.fusion().MakeInstructionPostOrder()) {
      post_order_indices[&instr.instruction()] = idx++;
    }
    absl::c_sort(child_fusions, [&](const HloFusionInstruction* a,
                                    const HloFusionInstruction* b) {
      return post_order_indices.at(a) < post_order_indices.at(b);
    });
  }

  spec.children.reserve(child_fusions.size());
  for (const HloFusionInstruction* child_fusion : child_fusions) {
    std::vector<int64_t> child_input_arg_indices;
    child_input_arg_indices.reserve(child_fusion->operand_count());
    for (const HloInstruction* operand : child_fusion->operands()) {
      if (operand->opcode() == HloOpcode::kParameter) {
        child_input_arg_indices.push_back(
            spec.input_arg_indices[operand->parameter_number()]);
        continue;
      }

      CHECK(spec.kind == NestedKernelSpec::Kind::kPhased);
      const HloInstruction* prod_inst = operand;
      int64_t prod_tuple_idx = 0;
      if (prod_inst->opcode() == HloOpcode::kGetTupleElement) {
        prod_tuple_idx = prod_inst->tuple_index();
        prod_inst = prod_inst->operand(0);
      }
      auto prod_it = absl::c_find_if(
          spec.children,
          [&](const NestedKernelSpec& c) { return c.fusion == prod_inst; });
      CHECK(prod_it != spec.children.end());
      CHECK_LT(prod_tuple_idx, prod_it->output_arg_indices.size());
      child_input_arg_indices.push_back(
          prod_it->output_arg_indices[prod_tuple_idx]);
    }

    std::vector<int64_t>& child_output_arg_indices =
        fusion_to_child_output_arg_indices[child_fusion];
    for (int64_t out_arg_idx : child_output_arg_indices) {
      CHECK_GE(out_arg_idx, 0)
          << "Multi-output nested fusion has unreferenced output in "
          << child_fusion->ToString();
    }
    HloFusionAnalysis child_analysis =
        HloFusionAnalysis::Create(*child_fusion, analysis.device_info());
    // TODO(b/480040242): Consider unrolling small, grid-bound fusions inside
    // kHorizontal fusions as well.
    bool child_unroll_small_fusions =
        unroll_small_fusions ||
        (spec.kind == NestedKernelSpec::Kind::kPhased &&
         analysis.device_info().cuda_compute_capability().IsAtLeastHopper());
    NestedKernelSpec child = BuildNestedKernelSpec(
        child_analysis, std::move(child_input_arg_indices),
        std::move(child_output_arg_indices),
        /*unroll_small_fusions=*/child_unroll_small_fusions);
    child.fusion = child_fusion;
    spec.children.push_back(std::move(child));
  }

  if (spec.kind == NestedKernelSpec::Kind::kPhased) {
    uint64_t max_blocks = 0;
    uint64_t max_threads = 0;
    bool child_uses_cluster = false;
    for (const NestedKernelSpec& child : spec.children) {
      CHECK_LE(child.launch_dims.num_blocks(),
               ThreadBlockSpecializedFusion::kMaxBlocksPerCluster);
      max_blocks = std::max(max_blocks, child.launch_dims.num_blocks());
      max_threads =
          std::max(max_threads, child.launch_dims.num_threads_per_block());
      child_uses_cluster |= child.cluster_dim.has_value();
    }
    // Cluster barriers only synchronize blocks within one cluster, so allocate
    // a portable cluster of power-of-2 size in {2, 4, 8} when multiple blocks
    // are needed, or a single block without a cluster when `max_blocks <= 1`.
    uint64_t cluster_blocks = absl::bit_ceil(std::min<uint64_t>(
        max_blocks, ThreadBlockSpecializedFusion::kMaxBlocksPerCluster));
    spec.launch_dims = LaunchDimensions(cluster_blocks, max_threads);
    if (cluster_blocks > 1 || child_uses_cluster) {
      spec.cluster_dim = se::ClusterDim(cluster_blocks, 1, 1);
    }
    return spec;
  }

  // Place clustered children first so each starts on a cluster boundary and
  // occupies a multiple of `cluster_x` CTAs, while non-clustered children pack
  // contiguously afterward without per-child padding.
  absl::c_stable_partition(spec.children, [](const NestedKernelSpec& child) {
    return child.cluster_dim.has_value();
  });

  uint64_t max_threads = 0;
  std::optional<uint64_t> cluster_x;
  for (const NestedKernelSpec& child : spec.children) {
    max_threads =
        std::max(max_threads, child.launch_dims.num_threads_per_block());
    if (child.cluster_dim.has_value()) {
      cluster_x = std::max(cluster_x.value_or(1), child.cluster_dim->x);
    }
  }

  uint64_t total_blocks = 0;
  for (const NestedKernelSpec& child : spec.children) {
    if (child.cluster_dim.has_value()) {
      total_blocks +=
          RoundUpTo<uint64_t>(child.launch_dims.num_blocks(), *cluster_x);
    } else {
      total_blocks += child.launch_dims.num_blocks();
    }
  }
  total_blocks = RoundUpTo<uint64_t>(total_blocks, cluster_x.value_or(1));
  spec.launch_dims = LaunchDimensions(total_blocks, max_threads);
  if (cluster_x.has_value()) {
    spec.cluster_dim = se::ClusterDim(*cluster_x, 1, 1);
  }
  return spec;
}

std::string GetSpecCacheDiscriminator(const NestedKernelSpec& root) {
  std::string out = "ThreadBlockSpecializedFusion:";
  auto append = [&](auto& self, const NestedKernelSpec& node) -> void {
    switch (node.kind) {
      case NestedKernelSpec::Kind::kLeaf:
        out.append("L[");
        for (size_t i = 0; i < node.output_arg_indices.size(); ++i) {
          if (i > 0) {
            out.push_back(',');
          }
          absl::StrAppend(&out, node.output_arg_indices[i]);
        }
        out.push_back(']');
        return;
      case NestedKernelSpec::Kind::kHorizontal:
        out.append("H(");
        break;
      case NestedKernelSpec::Kind::kPhased:
        out.append("P(");
        break;
    }
    for (size_t i = 0; i < node.children.size(); ++i) {
      if (i > 0) {
        out.push_back(',');
      }
      self(self, node.children[i]);
    }
    out.push_back(')');
  };
  append(append, root);
  return out;
}

}  // namespace

/*static*/
std::unique_ptr<MlirKernelEmitter>
ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(
    const HloFusionAnalysis& nested_analysis, bool unroll_small_fusions) {
  switch (nested_analysis.emitter_fusion_kind()) {
    case HloFusionAnalysis::EmitterFusionKind::kLoop:
      return std::make_unique<LoopFusion>(nested_analysis,
                                          unroll_small_fusions);
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return CreateReductionFusion(nested_analysis);
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
      return CreateTransposeFusion(nested_analysis);
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate:
      return std::make_unique<ConcatenateFusion>(nested_analysis);
    default:
      return nullptr;
  }
}

ThreadBlockSpecializedFusion::ThreadBlockSpecializedFusion(
    const HloFusionAnalysis& analysis)
    : device_info_(analysis.device_info()) {
  int64_t num_inputs = analysis.fusion().GetParameters().size();
  int64_t num_outputs = analysis.fusion_root_count();
  std::vector<int64_t> input_arg_indices(num_inputs);
  absl::c_iota(input_arg_indices, 0);
  std::vector<int64_t> output_arg_indices(num_outputs);
  absl::c_iota(output_arg_indices, num_inputs);

  root_spec_ = BuildNestedKernelSpec(analysis, std::move(input_arg_indices),
                                     std::move(output_arg_indices),
                                     /*unroll_small_fusions=*/false);
}

absl::Status ThreadBlockSpecializedFusion::EmitLauncherBody(
    mlir::ModuleOp launcher_module, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion,
    absl::Span<const NestedKernelSpec* const> leaf_specs,
    absl::Span<const mlir::OwningOpRef<mlir::ModuleOp>> nested_modules) const {
  mlir::MLIRContext* ctx = launcher_module.getContext();
  mlir::ImplicitLocOpBuilder b(entry_function.getLoc(), ctx);
  mlir::Type i32_ty = b.getI32Type();
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(ctx);

  // Declare external llvm.func symbols for each lowered leaf kernel.
  LeafDecls leaf_decls;
  leaf_decls.reserve(leaf_specs.size());
  b.setInsertionPointToStart(launcher_module.getBody());
  for (size_t idx = 0; idx < leaf_specs.size(); ++idx) {
    std::string nested_fn_name =
        GetNestedKernelFunctionName(entry_function.getName(), idx);
    auto nested_func =
        (*nested_modules[idx])
            .lookupSymbol<mlir::LLVM::LLVMFuncOp>(nested_fn_name);
    TF_RET_CHECK(nested_func != nullptr)
        << "Nested kernel function " << nested_fn_name << " not found.";
    auto decl = mlir::LLVM::LLVMFuncOp::create(b, nested_fn_name,
                                               nested_func.getFunctionType(),
                                               mlir::LLVM::Linkage::External);
    leaf_decls.emplace(leaf_specs[idx]->fusion, decl);
  }

  mlir::Block* entry_block = entry_function.addEntryBlock();
  b.setInsertionPointToStart(entry_block);

  // Add unrealized casts to LLVM pointers to be used when calling leaf kernels
  // as llvm functions.
  llvm::SmallVector<mlir::Value> arg_ptrs;
  arg_ptrs.reserve(entry_function.getNumArguments());
  for (mlir::Value arg : entry_function.getArguments()) {
    arg_ptrs.push_back(
        mlir::UnrealizedConversionCastOp::create(b, ptr_ty, arg).getResult(0));
  }

  // Emit block and thread IDs.
  mlir::Value bid_idx =
      mlir::gpu::BlockIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value tid_idx =
      mlir::gpu::ThreadIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value bid = mlir::arith::IndexCastUIOp::create(b, i32_ty, bid_idx);
  mlir::Value tid = mlir::arith::IndexCastUIOp::create(b, i32_ty, tid_idx);

  EmitNode(b, root_spec_, bid, tid, launch_dimensions().num_blocks(),
           launch_dimensions().num_threads_per_block(), arg_ptrs, leaf_decls);

  b.setInsertionPointToEnd(entry_block);
  mlir::func::ReturnOp::create(
      b, entry_function.getArguments().drop_front(fusion.operand_count()));
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ThreadBlockSpecializedFusion::CreateAndLowerMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    absl::string_view entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  mlir_context.appendDialectRegistry(MlirKernelEmitter::GetDialectRegistry());
  mlir_context.loadAllAvailableDialects();

  bool should_verify = (fusion.GetModule()
                            ->config()
                            .debug_options()
                            .xla_gpu_llvm_verification_level() >= 1);
  if constexpr (tsl::kIsDebugBuild) {
    should_verify = true;
  }

  std::vector<const NestedKernelSpec*> leaf_specs =
      CollectLeafSpecs(root_spec_);

  int64_t max_shmem_bytes = 0;
  int64_t warp_size = device_info_.threads_per_warp();
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> nested_modules;
  nested_modules.reserve(leaf_specs.size());

  for (size_t i = 0; i < leaf_specs.size(); ++i) {
    const NestedKernelSpec& leaf_spec = *leaf_specs[i];
    std::string nested_fn_name =
        GetNestedKernelFunctionName(entry_function_name, i);

    HloFusionAnalysis nested_analysis =
        HloFusionAnalysis::Create(*leaf_spec.fusion, device_info_);
    std::unique_ptr<MlirKernelEmitter> nested_emitter =
        CreateNativeNestedEmitter(
            nested_analysis,
            /*unroll_small_fusions=*/leaf_spec.unroll_small_fusions);
    TF_RET_CHECK(nested_emitter != nullptr);
    ABSL_ASSIGN_OR_RETURN(
        MlirKernelSource nested_source,
        nested_emitter->Emit(&mlir_context, *leaf_spec.fusion, nested_fn_name,
                             /*buffer_assignment=*/nullptr));
    ABSL_ASSIGN_OR_RETURN(nested_source,
                     CompileMlirToLlvmDialect(
                         device_info_, *fusion.GetModule(), nested_fn_name,
                         nested_emitter->unroll_factor(), mlir_context,
                         std::move(nested_source)));
    mlir::OwningOpRef<mlir::ModuleOp> nested_module =
        std::move(nested_source).TakeModule();

    VirtualizeSubKernelPassOptions virtualize_options;
    virtualize_options.entry_function_name_ = nested_fn_name;
    virtualize_options.expected_num_tensor_args_ =
        leaf_spec.input_arg_indices.size() +
        leaf_spec.output_arg_indices.size();
    virtualize_options.max_threads_ =
        launch_dimensions().num_threads_per_block();
    virtualize_options.warp_size_ = warp_size;
    const se::BlockDim& block_counts = leaf_spec.launch_dims.block_counts();
    virtualize_options.block_dim_x_ = block_counts.x;
    virtualize_options.block_dim_y_ = block_counts.y;
    virtualize_options.block_dim_z_ = block_counts.z;
    const se::ThreadDim& thread_counts =
        leaf_spec.launch_dims.thread_counts_per_block();
    virtualize_options.thread_dim_x_ = thread_counts.x;
    virtualize_options.thread_dim_y_ = thread_counts.y;
    virtualize_options.thread_dim_z_ = thread_counts.z;

    mlir::PassManager virtualize_pm(&mlir_context);
    virtualize_pm.enableVerifier(should_verify);
    virtualize_pm.addPass(createVirtualizeSubKernelPass(virtualize_options));
    tsl::StatusScopedDiagnosticHandler diagnostic_handler(
        nested_module->getContext());
    ABSL_RETURN_IF_ERROR(
        diagnostic_handler.consumeStatus(virtualize_pm.run(*nested_module)));

    if (auto shmem_attr =
            nested_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                "xla.shared_memory_bytes")) {
      max_shmem_bytes = std::max(max_shmem_bytes, shmem_attr.getInt());
    }

    nested_modules.push_back(std::move(nested_module));
  }

  mlir::OpBuilder builder(&mlir_context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> launcher_module =
      llvm_ir::CreateMlirModuleOp(loc);

  ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                   emitters::EmitKernelApi(
                       *launcher_module, fusion, buffer_assignment,
                       GetDefaultBufferAlignment(), entry_function_name));
  // Intermediate output buffers written before a block or cluster barrier and
  // read in a later phase appear in both `output_arg_indices` (from the root)
  // and a leaf's `input_arg_indices`. Mark them `xla.aliased` so
  // `MergePointersToSameSlicePass` omits `llvm.noalias` on those parameters
  // (and any aliased parameters sharing the same `xla.slice_index`), preventing
  // LLVM `BasicAA` from reordering cross-phase accesses across the barrier
  // after inlining.
  for (const NestedKernelSpec* leaf : leaf_specs) {
    for (int64_t arg_idx : leaf->input_arg_indices) {
      if (arg_idx >= fusion.operand_count()) {
        entry_func.setArgAttr(arg_idx, emitters::kXlaAliasedAttr,
                              builder.getUnitAttr());
      }
    }
  }
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);
  emitters::SetIndexDataLayout(*launcher_module, fusion);
  ABSL_RETURN_IF_ERROR(EmitLauncherBody(*launcher_module, entry_func, fusion,
                                   leaf_specs, nested_modules));
  ABSL_RETURN_IF_ERROR(LowerLauncherModuleToLlvmDialect(
      *launcher_module, device_info_, should_verify));

  if (std::optional<se::ClusterDim> cluster_dim = this->cluster_dim()) {
    auto launcher_entry = launcher_module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
        entry_function_name);
    TF_RET_CHECK(launcher_entry != nullptr);
    launcher_entry->setAttr(
        mlir::NVVM::NVVMDialect::getClusterDimAttrName(),
        builder.getDenseI32ArrayAttr({static_cast<int32_t>(cluster_dim->x),
                                      static_cast<int32_t>(cluster_dim->y),
                                      static_cast<int32_t>(cluster_dim->z)}));
  }

  for (size_t i = 0; i < leaf_specs.size(); ++i) {
    ABSL_RETURN_IF_ERROR(SpliceNestedKernelModuleIntoLauncher(
        *launcher_module, *nested_modules[i],
        GetNestedKernelFunctionName(entry_function_name, i)));
  }

  (*launcher_module)
      ->setAttr("xla.shared_memory_bytes",
                builder.getI64IntegerAttr(max_shmem_bytes));

  return launcher_module;
}

absl::StatusOr<ThreadBlockSpecializedFusion::LlvmKernelSourceWrapper>
ThreadBlockSpecializedFusion::CreateLLVMModule(
    const se::DeviceDescription& device, const HloFusionInstruction& fusion,
    absl::string_view entry_function_name,
    const BufferAssignment* buffer_assignment,
    mlir::MLIRContext& mlir_context) const {
  ABSL_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      CreateAndLowerMLIRModule(mlir_context, fusion, entry_function_name,
                               buffer_assignment));
  int64_t max_shmem_bytes =
      (*module)
          ->getAttrOfType<mlir::IntegerAttr>("xla.shared_memory_bytes")
          .getInt();

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(*module, *llvm_context);
  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate thread-block-specialized fusion MLIR module to "
         "LLVM IR.";

  return LlvmKernelSourceWrapper{
      LlvmKernelSource(std::move(llvm_context), std::move(llvm_module)),
      max_shmem_bytes};
}

AsyncThunkSequence ThreadBlockSpecializedFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  VLOG(4) << "ThreadBlockSpecializedFusion: "
          << fusion.fused_instructions_computation()->ToString();
  if (cluster_dim().has_value()) {
    const auto* cuda_cc =
        ir_emitter_context.gpu_compute_capability().cuda_compute_capability();
    if (cuda_cc == nullptr || !cuda_cc->IsAtLeastHopper()) {
      return absl::UnimplementedError(absl::StrCat(
          "PhasedBlockClusterFusion requires CUDA Hopper (SM90+), got: ",
          ir_emitter_context.gpu_compute_capability().ToString()));
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      emitters::KernelArguments args,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(), &fusion));

  std::string discriminator = GetSpecCacheDiscriminator(root_spec_);
  auto [future_entry, cached] = ir_emitter_context.kernel_cache().GetWithStatus(
      fusion.fused_instructions_computation(), args, discriminator,
      [&]() -> xla::Future<KernelReuseCache::Entry> {
        std::string kernel_name = ir_emitter_context.GetSanitizedUniqueName(
            std::string(fusion.name()));
        BorrowedMlirContext borrowed_context =
            ir_emitter_context.BorrowMlirContext();
        ABSL_ASSIGN_OR_RETURN(
            LlvmKernelSourceWrapper llvm_source_wrapper,
            CreateLLVMModule(
                ir_emitter_context.gpu_device_info(), fusion, kernel_name,
                &ir_emitter_context.buffer_assignment(), **borrowed_context));

        llvm::orc::ThreadSafeModule safe_module =
            std::move(llvm_source_wrapper.source).thread_safe_module();
        LaunchDimensions launch_dims = launch_dimensions();
        ABSL_RETURN_IF_ERROR(AnnotateMlirKernelLlvmModule(
            ir_emitter_context.gpu_device_info(), launch_dims, kernel_name,
            ir_emitter_context.data_layout(),
            ir_emitter_context.target_triple(),
            safe_module.getModuleUnlocked()));

        bool use_pdl =
            IsPdlEnabled(fusion.GetModule()->config().debug_options(),
                         ir_emitter_context.gpu_compute_capability());

        GpuKernelSpec spec{std::move(kernel_name), std::move(launch_dims), args,
                           std::nullopt, use_pdl};
        return ir_emitter_context.kernel_compiler()
            ->CompileToTargetBinary(LlvmKernelSource(std::move(safe_module)),
                                    spec)
            .Map([spec = std::move(spec), cluster_dim = cluster_dim(),
                  max_shmem_bytes = llvm_source_wrapper.max_shmem_bytes](
                     const std::vector<uint8_t>& cubin) mutable {
              KernelReuseCache::Entry entry{
                  std::string{spec.name()}, spec.launch_dimensions(),
                  cluster_dim, max_shmem_bytes,
                  std::make_shared<const std::vector<uint8_t>>(cubin)};
              entry.use_pdl = spec.use_pdl();
              return entry;
            });
      });

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      &fusion, ir_emitter_context.GetNextThunkId());
  bool kernel_cached = cached;
  return future_entry.Map(
      [&fusion, thunk_info = std::move(thunk_info), kernel_cached,
       devices_in_process =
           ir_emitter_context.gpu_topology().num_devices_per_process()](
          const KernelReuseCache::Kernel& entry) mutable
          -> absl::StatusOr<ThunkSequence> {
        if (kernel_cached) {
          VLOG(3) << "Reuse: " << fusion.name() << " -> " << entry.spec.name();
        }
        ABSL_ASSIGN_OR_RETURN(
            CustomKernel custom_kernel,
            kernel::CreateSharedCubinCustomKernel(entry.spec, entry.binary));

        return ThunkSequence::Of<CustomKernelThunk>(
            thunk_info, std::move(custom_kernel), entry.spec.kernel_arguments(),
            devices_in_process, entry.spec.use_pdl());
      });
}

}  // namespace xla::gpu
