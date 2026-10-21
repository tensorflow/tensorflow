/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/collective_emitter.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/lowering_util.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/all_to_all.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/reduce_scatter.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/codegen/emitters/ir/xla_ops.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_attrs.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/layout_util.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace xla::gpu {
namespace {

using ::xla::se::gpu::AllReduceStrategy;
using ::xla::xtile::BlockLevelFusionConfig;

namespace ttir = ::mlir::triton;
namespace mtx = ::mlir::triton::xla;
namespace arith = ::mlir::arith;

using ReductionComputationEmitter = absl::AnyInvocable<xtile::TensorValue(
    mlir::ImplicitLocOpBuilder&, xtile::TensorValue, xtile::TensorValue)>;

// The main memory space on a device (HBM).
static constexpr auto kGlobalAddressSpace = ttir::PtrAddrSpace::Global;

// Metadata arguments for the collective emitter.
// device_rank, signal_value, signal_buffers.
static constexpr int32_t kNumCollectiveMetadataArgs = 3;
static constexpr int32_t kNumTileIndexArgs = 1;

// Per-operand state for reduction emitters (all-reduce and reduce-scatter).
struct ReductionOperandContext {
  // The input tile to the collective op.
  xtile::TensorValue input_tile;
  // The extract tile op that produced the input tile.
  xtile::ExtractTileOp input_extract;
  // !tt.ptr<i64> table of the operand's per-rank remote buffers.
  mlir::Value remote_input_buffers;
  // The entire shape of the input to the collective op.
  llvm::SmallVector<int64_t, 4> non_tiled_input_shape;
  PrimitiveType element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID;
  // Total number of elements in the input to the collective op.
  int64_t num_elements{0};
  // Size of one rank's part along the scatter dimension (for reduce-scatter).
  int64_t per_rank_size{0};
  llvm::SmallVector<int64_t, 4> output_tile_shape;
};

// Common context for reduction emitters (all-reduce and reduce-scatter).
struct ReductionEmitterContext {
  mlir::Operation* op = nullptr;
  int32_t num_input_output_args{0};
  int32_t num_scratch_buffers{0};
  bool has_invocation_count{true};
  // The entry function of the collective op.
  xtile::EntryFuncOp xtile_entry_fn;
  // The total number of devices in the collective group.
  int64_t world_size{0};
  llvm::SmallVector<ReductionOperandContext, 2> operands;
};

struct AllReduceEmitterContext : public ReductionEmitterContext {
  AllReduceStrategy strategy;
};

struct ReduceScatterEmitterContext : public ReductionEmitterContext {
  int64_t scatter_dimension{0};
  int64_t signal_stride{0};
};

template <typename ContextT, typename OpTy>
absl::StatusOr<ContextT> CreateReductionEmitterContext(
    OpTy op, bool has_invocation_count = true,
    bool allow_multiple_operands = false) {
  ContextT ctx;
  if (op->getNumOperands() < 1 ||
      (!allow_multiple_operands && op->getNumOperands() != 1)) {
    return absl::InvalidArgumentError(
        "Collective op must have at least one operand (or exactly one when "
        "variadic is unsupported) in order to be lowered to triton.");
  }
  if (op->getNumOperands() != op->getNumResults()) {
    return absl::InvalidArgumentError(
        "Collective op must have one result per operand.");
  }
  ctx.xtile_entry_fn = op->template getParentOfType<xtile::EntryFuncOp>();
  if (!ctx.xtile_entry_fn) {
    return absl::InvalidArgumentError(
        "Collective op must be in an XTile entry function in order to be "
        "lowered to triton.");
  }
  // Layout: 2 arguments per operand for input/output, metadata arguments
  // (3 with invocation count, 2 without), one per operand for scratch buffers,
  // plus 1 for the tile index.
  ctx.num_input_output_args = op->getNumOperands() * 2;
  ctx.num_scratch_buffers = op->getNumOperands();
  ctx.has_invocation_count = has_invocation_count;
  const int32_t num_metadata_args = has_invocation_count
                                        ? kNumCollectiveMetadataArgs
                                        : kNumCollectiveMetadataArgs - 1;
  const int32_t expected_num_args = ctx.num_input_output_args +
                                    ctx.num_scratch_buffers +
                                    num_metadata_args + kNumTileIndexArgs;
  if (ctx.xtile_entry_fn.getNumArguments() != expected_num_args) {
    return absl::InvalidArgumentError(
        absl::StrCat("Collective op must have ", expected_num_args,
                     " arguments in order to "
                     "be lowered to triton, but it has ",
                     ctx.xtile_entry_fn.getNumArguments()));
  }
  const int32_t buffers_start_idx =
      ctx.num_input_output_args + (has_invocation_count ? 2 : 1);
  ctx.operands.reserve(op->getNumOperands());
  for (int64_t i = 0; i < op->getNumOperands(); ++i) {
    ReductionOperandContext operand;
    mlir::Type element_type =
        mlir::cast<mlir::ShapedType>(op->getOperand(i).getType())
            .getElementType();
    operand.element_type = xla::ConvertMlirTypeToPrimitiveType(element_type);
    if (operand.element_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) {
      std::string type_string;
      llvm::raw_string_ostream stream(type_string);
      op->getOperand(i).print(stream);
      return absl::InvalidArgumentError(absl::StrFormat(
          "Could not convert operand type to a valid PrimitiveType."
          "Operand Type: %s",
          type_string));
    }
    operand.input_tile = mlir::cast<xtile::TensorValue>(op->getOperand(i));
    // We assume the input to the reduction collective is an
    // xtile::ExtractTileOp, or that the parent of the input is an
    // xtile::ExtractTileOp (edge case for booleans).
    operand.input_extract = llvm::dyn_cast_if_present<xtile::ExtractTileOp>(
        operand.input_tile.getDefiningOp());
    if (!operand.input_extract && operand.input_tile.getDefiningOp() &&
        operand.input_tile.getDefiningOp()->getNumOperands() > 0) {
      // Workaround(i1_to_i8_workaround).
      // Go one place up this is an edge case for booleans
      // Booleans are stored as i8 and then casted to i1 so the tile we get is
      // after the cast. To get the extract tile we need to go one step up.
      operand.input_extract = llvm::dyn_cast_if_present<xtile::ExtractTileOp>(
          operand.input_tile.getDefiningOp()->getOperand(0).getDefiningOp());
    }
    if (!operand.input_extract) {
      return absl::InvalidArgumentError(
          "Collective op must have an extract tile op as operand in order to "
          "be lowered to triton.");
    }
    operand.remote_input_buffers =
        ctx.xtile_entry_fn.getArgument(buffers_start_idx + 1 + i);
    // The source for the extract is the non-tiled input (memref).
    operand.non_tiled_input_shape = llvm::SmallVector<int64_t, 4>(
        operand.input_extract.getSource().getType().getShape());
    operand.num_elements = Product(operand.non_tiled_input_shape);
    ctx.operands.push_back(std::move(operand));
  }
  ctx.op = op.getOperation();
  ABSL_ASSIGN_OR_RETURN(auto replica_groups,
                   xla::ConvertReplicaGroups(op.getReplicaGroups(), op));
  ctx.world_size = replica_groups->num_devices_per_group();
  return ctx;
}

absl::StatusOr<AllReduceEmitterContext> CreateAllReduceEmitterContext(
    mlir::stablehlo::AllReduceOp op) {
  ABSL_ASSIGN_OR_RETURN(AllReduceEmitterContext ctx,
                   CreateReductionEmitterContext<AllReduceEmitterContext>(op));
  std::optional<xtile::CollectiveStrategy> strategy =
      xtile::GetCollectiveStrategy(op);
  if (!strategy.has_value()) {
    return absl::InvalidArgumentError(
        "AllReduce op must have a collective_strategy attribute.");
  }
  switch (*strategy) {
    case xtile::CollectiveStrategy::kOneShot:
      ctx.strategy = AllReduceStrategy::kOneShot;
      break;
    case xtile::CollectiveStrategy::kTwoShot:
      ctx.strategy = AllReduceStrategy::kTwoShot;
      break;
  }
  return ctx;
}

// Validates the preconditions for emitting a reduce-scatter (actual validation
// happens when the collective fusion is formed) and creates the
// ReduceScatterEmitterContext.
absl::StatusOr<ReduceScatterEmitterContext> CreateReduceScatterEmitterContext(
    xtile::ReduceScatterOp op) {
  ABSL_ASSIGN_OR_RETURN(ReduceScatterEmitterContext ctx,
                   CreateReductionEmitterContext<ReduceScatterEmitterContext>(
                       op, /*has_invocation_count=*/false,
                       /*allow_multiple_operands=*/true));
  // The rank a block copies for is derived from the program id (see
  // signal_slot below).
  // This requires one tile per program, i.e. no tile loop around the op.
  if (op->getParentOp() != ctx.xtile_entry_fn.getOperation()) {
    return absl::InvalidArgumentError(
        "ReduceScatter op must be directly nested in the XTile entry "
        "function.");
  }
  const int64_t scatter_dim = op.getScatterDimension();
  int64_t signal_stride = 1;
  for (int64_t i = 0; i < ctx.operands.size(); ++i) {
    ReductionOperandContext& operand = ctx.operands[i];
    const llvm::ArrayRef<int64_t> input_tile_shape =
        operand.input_tile.getType().getShape();
    const llvm::ArrayRef<int64_t> input_shape = operand.non_tiled_input_shape;
    if (input_shape.empty() || scatter_dim < 0 ||
        scatter_dim >= static_cast<int64_t>(input_shape.size()) ||
        input_tile_shape.size() != input_shape.size()) {
      return absl::InvalidArgumentError(
          "Invalid scatter dimension for reduce-scatter.");
    }
    if (ctx.world_size <= 0 || input_shape[scatter_dim] % ctx.world_size != 0) {
      return absl::InvalidArgumentError(
          "Reduce-scatter scatter dimension is not divisible by world size.");
    }
    const llvm::ArrayRef<int64_t> output_tile_shape =
        mlir::cast<mlir::RankedTensorType>(op.getType(i)).getShape();
    if (output_tile_shape.size() != input_shape.size()) {
      return absl::InvalidArgumentError(
          "Reduce-scatter output tile rank must match input rank.");
    }
    const int64_t per_rank_size = input_shape[scatter_dim] / ctx.world_size;
    if (input_tile_shape[scatter_dim] !=
            output_tile_shape[scatter_dim] * ctx.world_size ||
        per_rank_size % input_tile_shape[scatter_dim] != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Reduce-scatter input tile size along the scatter dimension (",
          input_tile_shape[scatter_dim],
          ") must equal output_tile_size * world_size (",
          output_tile_shape[scatter_dim] * ctx.world_size,
          ") and divide the per-rank size (", per_rank_size, ")."));
    }
    int64_t programs_per_rank = per_rank_size / input_tile_shape[scatter_dim];
    for (int64_t d = 0; d < static_cast<int64_t>(input_shape.size()); ++d) {
      if (d == scatter_dim) {
        continue;
      }
      programs_per_rank *=
          xla::CeilOfRatio(input_shape[d], input_tile_shape[d]);
    }
    signal_stride = std::max(signal_stride, programs_per_rank);
    operand.per_rank_size = per_rank_size;
    operand.output_tile_shape.assign(output_tile_shape.begin(),
                                     output_tile_shape.end());
  }
  if (signal_stride * ctx.world_size > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError("Too many reduce-scatter programs.");
  }
  ctx.scatter_dimension = scatter_dim;
  ctx.signal_stride = signal_stride;
  return ctx;
}

// Unmanaged kernel arguments of collectives that use the AllReduce argument
// layout: rank, signal value (except for reduce-scatter), signal buffers and
// one remote buffer per parameter.
absl::StatusOr<std::vector<Shape>> GetRemoteBufferUnmanagedKernelArguments(
    const HloComputation* computation,
    const HloCollectiveInstruction* collective) {
  const int32_t num_devices =
      collective->device_list()->num_devices_per_group();
  const bool has_invocation_count =
      collective->opcode() != HloOpcode::kReduceScatter;
  const int32_t num_metadata_args = has_invocation_count
                                        ? kNumCollectiveMetadataArgs
                                        : kNumCollectiveMetadataArgs - 1;
  std::vector<Shape> unmanaged_arguments;
  unmanaged_arguments.reserve(computation->num_parameters() +
                              num_metadata_args);

  // rank (and signal_value when present)
  unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));
  if (has_invocation_count) {
    unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));
  }
  // The shape for signal and scratch buffers does not really matter in the end
  // because this would just be a pointer. For documentation purposes we add
  // the correct shape which would be
  // - num_devices * num_blocks for the signal buffer.
  // - num_devices * shape of the parameter for scratch buffers.
  // Since number of blocks is not known in this context we use a constant.
  static constexpr int32_t kMaxBlocksPerGrid = 24;
  unmanaged_arguments.push_back(
      ShapeUtil::MakeShape(S32, {num_devices, kMaxBlocksPerGrid}));
  // Scratch buffers
  for (const HloInstruction* instr : computation->parameter_instructions()) {
    Shape shape =
        ShapeUtil::InsertDimensionAtIndex(instr->shape(), 0, num_devices);
    unmanaged_arguments.push_back(shape);
  }
  TF_RET_CHECK(unmanaged_arguments.size() ==
               computation->num_parameters() + num_metadata_args);
  return unmanaged_arguments;
}

mlir::LogicalResult PopulateReductionComputation(
    mlir::PatternRewriter& rewriter, mlir::Operation* op,
    ReductionComputationEmitter& computation_emitter) {
  // At the moment we expect only one operation in the reduction computation
  // to be relevant.
  auto& reduction_computation_region = op->getRegion(0);
  int num_ops_to_emit = 0;
  for (auto& block : reduction_computation_region.getBlocks()) {
    for (auto& block_op : block.without_terminator()) {
      if (llvm::dyn_cast<mlir::tensor::ExtractOp>(block_op) ||
          (llvm::dyn_cast<mlir::tensor::FromElementsOp>(block_op))) {
        // These ops are not relevant to the reduction and are just emitted so
        // that we have a valid stablehlo all reduce op.
        // We don't emit them, but they don't count towards our only one op in
        // the reduction computation requirement.
        continue;
      }

      if (block_op.getNumOperands() != 2) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "AllReduce computation must only contain binary operations.");
      }

      if (block_op.getNumResults() != 1) {
        return rewriter.notifyMatchFailure(
            op->getLoc(),
            "AllReduce computation must only contain operations with one "
            "result.");
      }

      computation_emitter = [&](mlir::ImplicitLocOpBuilder& builder,
                                xtile::TensorValue accumulator,
                                xtile::TensorValue next_tile) {
        // Emit a generic binary operation with the given operands.
        mlir::OperationState state(builder.getLoc(), block_op.getName());
        state.addOperands({accumulator, next_tile});
        state.addTypes({accumulator.getType()});
        mlir::Operation* new_op = builder.create(state);
        return mlir::cast<xtile::TensorValue>(new_op->getResult(0));
      };
      num_ops_to_emit++;
    }
  }

  if (num_ops_to_emit != 1) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        "AllReduce op must have exactly one relevant operation in order to "
        "be lowered to triton.");
  }

  return mlir::success();
}

// Emits code that reads the last barrier signal value posted by this block from
// its own slot in the local rank's signal buffer and returns `last + 1` as the
// signal value for this launch's first barrier. It replaces the host-provided
// invocation count argument (see `CollectiveCodegenConfig::device_sync_count`).
//
// Arguments:
// - `signal_buffers`: `!tt.ptr<i64>` table of the per-rank signal buffers.
// - `rank`: i32 rank of this device in the replica group.
// - `block_id`: i32 program id of this block.
// - `world_size`: number of ranks in the replica group.
// - `signal_slot`, `signal_stride`: See `BlockBarrierOp` for details.
//   If not set, we use the default indexing of
//   SignalBuffers[rank][block_id * world_size + rank].
//
// After a local CTA barrier, `BlockBarrierOp` writes the barrier signal value
// into this slot of `SignalBuffers[0..world_size]`, and no other rank or block
// ever writes to this slot of `SignalBuffers[rank]`.
// For two-shot `AllReduce`, the second barrier writes `signal_value + 1`, so
// the slot advances by 2 across launches and adding 1 at entry yields the next
// launch's first signal value. Signal buffers are allocated and zeroed once per
// executor, so the counter persists across executions (including CUDA graph
// replays and device-side loops) without any extra store or thread barrier.
mlir::Value EmitDeviceInvocationCount(mlir::ImplicitLocOpBuilder& b,
                                      mlir::Value signal_buffers,
                                      mlir::Value rank, mlir::Value block_id,
                                      int64_t world_size,
                                      mlir::Value signal_slot = nullptr,
                                      int64_t signal_stride = 0) {
  CHECK_EQ(signal_slot != nullptr, signal_stride > 0)
      << "signal_slot and signal_stride must be set together";
  const auto ptr_to_i32_type =
      ttir::PointerType::get(b.getI32Type(), kGlobalAddressSpace);
  const auto ptr_to_i64_type =
      ttir::PointerType::get(b.getI64Type(), kGlobalAddressSpace);
  mlir::Value signal_buffers_i64 =
      ttir::BitcastOp::create(b, ptr_to_i64_type, signal_buffers);
  // SignalBuffers[rank]
  mlir::Value local_signal_buffer_i64 = ttir::LoadOp::create(
      b, ttir::AddPtrOp::create(b, ptr_to_i64_type, signal_buffers_i64, rank),
      /*isVolatile=*/false);
  mlir::Value local_signal_buffer =
      ttir::IntToPtrOp::create(b, ptr_to_i32_type, local_signal_buffer_i64);
  mlir::Value world_size_op = arith::ConstantOp::create(
      b, b.getI32IntegerAttr(static_cast<int32_t>(world_size)));
  mlir::Value counter_index;
  if (signal_slot != nullptr) {
    // Computes:
    // counter_index = ((block_id - signal_slot * signal_stride +
    //                   rank * signal_stride) * world_size) + signal_slot
    // See LowerBlockBarrierOp in triton_xla_lower_block_barrier_pass.cc for
    // details on the asymmetric barrier indexing.
    mlir::Value stride_op = arith::ConstantOp::create(
        b, b.getI32IntegerAttr(static_cast<int32_t>(signal_stride)));
    mlir::Value slot_offset = arith::MulIOp::create(b, signal_slot, stride_op);
    mlir::Value base_block = arith::SubIOp::create(b, block_id, slot_offset);
    mlir::Value rank_offset = arith::MulIOp::create(b, rank, stride_op);
    mlir::Value target_block =
        arith::AddIOp::create(b, base_block, rank_offset);
    mlir::Value target_block_offset =
        arith::MulIOp::create(b, target_block, world_size_op);
    counter_index = arith::AddIOp::create(b, target_block_offset, signal_slot);
  } else {
    // SignalBuffers[rank][block_id * world_size + rank]
    counter_index = arith::AddIOp::create(
        b, arith::MulIOp::create(b, block_id, world_size_op), rank);
  }
  mlir::Value counter_ptr = ttir::AddPtrOp::create(
      b, ptr_to_i32_type, local_signal_buffer, counter_index);
  // Volatile to make sure that every launch reads the counter from memory.
  mlir::Value counter =
      ttir::LoadOp::create(b, counter_ptr, /*isVolatile=*/true);
  return arith::AddIOp::create(
      b, counter, arith::ConstantOp::create(b, b.getI32IntegerAttr(1)));
}

// Picks the active half (0 or 1) of a double buffer from the lowest bit of
// `signal_value`.
mlir::Value EmitDoubleBufferIndex(mlir::ImplicitLocOpBuilder& b,
                                  mlir::Value signal_value) {
  const mlir::Type i64_type = b.getI64Type();
  mlir::Value signal_value_i64 =
      arith::ExtSIOp::create(b, i64_type, signal_value);
  mlir::Value one =
      arith::ConstantOp::create(b, i64_type, b.getI64IntegerAttr(1));
  return arith::AndIOp::create(b, signal_value_i64, one);
}

// Returns the element offset of the active half (`buffer_index` in {0, 1}) of
// a double buffer that holds `num_elements` elements of `element_type` per
// half.
mlir::Value EmitDoubleBufferOffsetFromIndex(mlir::ImplicitLocOpBuilder& b,
                                            mlir::Value buffer_index,
                                            int64_t num_elements,
                                            PrimitiveType element_type) {
  const mlir::Type i64_type = b.getI64Type();
  // The allocated double buffer size in bytes is aligned to
  // kXlaAllocatedBufferAlignBytes. To get the offset to the second buffer, we
  // divide the buffer size by 2 and then divide by the element size to get
  // the number of elements in the buffer. Its important to do this to make
  // sure that start of the buffers are always aligned to 16 bytes.
  const int64_t element_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  const int64_t buffer_size = xla::RoundUpTo<uint64_t>(
      num_elements * element_size, kXlaAllocatedBufferAlignBytes);
  const int64_t elements_per_buffer = buffer_size / element_size;
  mlir::Value elements_per_buffer_op = arith::ConstantOp::create(
      b, i64_type, b.getI64IntegerAttr(elements_per_buffer));
  return arith::MulIOp::create(b, buffer_index, elements_per_buffer_op);
}

// Returns the element offset of the active half of a double buffer that holds
// `num_elements` elements of `element_type` per half. The lowest bit of
// `signal_value` selects the half.
mlir::Value EmitDoubleBufferOffset(mlir::ImplicitLocOpBuilder& b,
                                   mlir::Value signal_value,
                                   int64_t num_elements,
                                   PrimitiveType element_type) {
  mlir::Value buffer_index = EmitDoubleBufferIndex(b, signal_value);
  return EmitDoubleBufferOffsetFromIndex(b, buffer_index, num_elements,
                                         element_type);
}

// Emits instructions to get the pointer to the remote buffer of the given
// rank.
// We have a !tt.ptr<i64> pointing to the base of the remote
// buffers. We add the rank index to the base pointer and load to get to the
// base pointer of the remote buffer of the given rank. Then we add the buffer
// offset to get the pointer to the correct buffer inside (double buffering).
// Note that the returned pointer is of the storage type not the logical type.
mlir::Value EmitRemoteBufferPtr(mlir::ImplicitLocOpBuilder& b,
                                mlir::Value remote_buffers, mlir::Value rank,
                                mlir::Value buffer_offset,
                                mlir::Type elem_storage_type) {
  const auto ptr_to_i64_type =
      ttir::PointerType::get(b.getI64Type(), kGlobalAddressSpace);
  const auto ptr_to_elem_type =
      ttir::PointerType::get(elem_storage_type, kGlobalAddressSpace);
  mlir::Value remote_buf_ptr_addr =
      ttir::AddPtrOp::create(b, ptr_to_i64_type, remote_buffers, rank);
  mlir::Value remote_buf_i64 = ttir::LoadOp::create(b,                      //
                                                    remote_buf_ptr_addr,    //
                                                    /*isVolatile=*/false);  //
  mlir::Value remote_buf_ptr_base = ttir::IntToPtrOp::create(
      b, ptr_to_elem_type, remote_buf_i64,
      llvm::ArrayRef<mlir::NamedAttribute>{xtile::GetDivisibilityAttr(b)});
  return ttir::AddPtrOp::create(b, ptr_to_elem_type, remote_buf_ptr_base,
                                buffer_offset);
}

// Emits a barrier between this block and the matching blocks of the other
// ranks.
// Otherwise, remote ranks might start reading the data before it is ready.
void EmitBlockBarrier(
    mlir::ImplicitLocOpBuilder& b, mlir::Value signal_buffers, mlir::Value rank,
    mlir::Value signal_value, int64_t world_size,
    mlir::Value signal_slot = nullptr, int64_t signal_stride = 0,
    mtx::BarrierMode barrier_mode = mtx::BarrierMode::kSymmetric,
    bool emit_local_barrier = true) {
  if (emit_local_barrier) {
    mlir::triton::gpu::BarrierOp::create(b,
                                         mlir::triton::gpu::AddrSpace::Local);
  }
  mtx::BlockBarrierOp::create(
      b, signal_buffers, rank, signal_value, signal_slot,
      b.getI32IntegerAttr(static_cast<int32_t>(world_size)),
      signal_stride > 0
          ? b.getI32IntegerAttr(static_cast<int32_t>(signal_stride))
          : mlir::IntegerAttr(),
      mtx::BarrierModeAttr::get(b.getContext(), barrier_mode));
}

struct CollectiveOperandExtractInfo {
  xtile::TensorValue input_tile;
  xtile::ExtractTileOp input_extract;
  mlir::TypedValue<mlir::MemRefType> input_buffer;
  mlir::Value remote_buffers_arg;
  PrimitiveType element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID;
};

absl::StatusOr<CollectiveOperandExtractInfo> ExtractCollectiveOperand(
    mlir::Value operand, int64_t operand_index, absl::string_view op_name,
    xtile::EntryFuncOp entry_func, int64_t num_parameters,
    int32_t remote_buffers_start, bool require_single_use = true) {
  auto input_tile = mlir::cast<xtile::TensorValue>(operand);
  auto input_extract = llvm::dyn_cast_if_present<xtile::ExtractTileOp>(
      input_tile.getDefiningOp());
  if (!input_extract && input_tile.getDefiningOp() &&
      input_tile.getDefiningOp()->getNumOperands() > 0) {
    // Workaround(i1_to_i8_workaround): booleans are stored as i8 and cast to
    // i1 after ExtractTileOp.
    input_extract = llvm::dyn_cast_if_present<xtile::ExtractTileOp>(
        input_tile.getDefiningOp()->getOperand(0).getDefiningOp());
  }
  if (!input_extract) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s operand %d must be defined by xtile::ExtractTileOp.", op_name,
        operand_index));
  }

  // The collective operand must be read from a fusion parameter. Its argument
  // number selects the matching remote buffer argument.
  auto input_arg =
      mlir::dyn_cast<mlir::BlockArgument>(input_extract.getSource());
  if (!input_arg ||
      input_arg.getOwner()->getParentOp() != entry_func.getOperation() ||
      input_arg.getArgNumber() >= num_parameters) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s operand %d must be extracted from a fusion parameter.", op_name,
        operand_index));
  }
  if (require_single_use &&
      (!input_extract->hasOneUse() || !input_tile.hasOneUse())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s input tile %d must only be used by the %s op.",
                        op_name, operand_index, op_name));
  }

  PrimitiveType element_type = xla::ConvertMlirTypeToPrimitiveType(
      input_extract.getType().getElementType());
  if (element_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not convert %s operand %d element type to PrimitiveType.",
        op_name, operand_index));
  }

  CollectiveOperandExtractInfo info;
  info.input_tile = input_tile;
  info.input_extract = input_extract;
  info.input_buffer = input_extract.getSource();
  info.remote_buffers_arg =
      entry_func.getArgument(remote_buffers_start + input_arg.getArgNumber());
  info.element_type = element_type;
  return info;
}

class ReductionEmitter {
 protected:
  struct OperandState {
    mlir::Value remote_input_buffers_i64;
    mlir::Value buffer_offset;
    // Layout of the input tensor in minor-to-major order.
    llvm::SmallVector<int64_t> layout;
    mlir::Type elem_type;
    mlir::Type elem_storage_type;
    ttir::PointerType ptr_to_elem_type;
  };

  ReductionEmitter(ReductionEmitterContext& ctx,
                   mlir::PatternRewriter& rewriter)
      : base_ctx_(ctx),
        rewriter_(rewriter),
        builder_(base_ctx_.op->getLoc(), rewriter) {}

  ~ReductionEmitter() = default;

  absl::Status Initialize(bool is_two_shot = false) {
    CHECK(!initialized_);
    // NB: This must be done before any other IR is emitted so that we can bail
    // out in case it fails.
    // Otherwise, the IR is considered modified and we end up in an infinite
    // loop.
    if (mlir::failed(PopulateReductionComputation(
            rewriter_, base_ctx_.op, reduce_computation_emitter_))) {
      return absl::InternalError("Failed to populate reduction computation.");
    }
    // 1. Opaque arguments. They start after the input/output arguments.
    const int32_t start_idx = base_ctx_.num_input_output_args;
    device_rank_ = base_ctx_.xtile_entry_fn.getArgument(start_idx);
    if (!device_rank_.getType().isInteger(32)) {
      return absl::InvalidArgumentError(
          "Collective rank argument must be an i32.");
    }
    // When present (e.g. AllReduce), the invocation count argument at
    // `start_idx + 1` is unused: the signal value comes from a counter in
    // device memory (see Emit setup IR below).
    const int32_t buffers_start_idx =
        start_idx + (base_ctx_.has_invocation_count ? 2 : 1);
    // !tt.ptr<i64>
    signal_buffers_ = base_ctx_.xtile_entry_fn.getArgument(buffers_start_idx);

    // 2. Constants, types, and setup IR.
    ptr_to_i64_type_ =
        ttir::PointerType::get(builder_.getI64Type(), kGlobalAddressSpace);
    block_id_ = ttir::GetProgramIdOp::create(builder_, 0);
    signal_value_ =
        EmitDeviceInvocationCount(builder_, signal_buffers_, device_rank_,
                                  block_id_, base_ctx_.world_size);
    mlir::Value signal_value = signal_value_;
    if (is_two_shot) {
      // For two-shot signal value is always incremented by 2.
      // So we need to right shift it by 1 to get the buffer index.
      signal_value = arith::ShRSIOp::create(
          builder_, signal_value_.getType(), signal_value_,
          arith::ConstantOp::create(builder_, signal_value_.getType(),
                                    builder_.getI32IntegerAttr(1)));
    }
    mlir::Value buffer_index = EmitDoubleBufferIndex(builder_, signal_value);

    operand_states_.reserve(base_ctx_.operands.size());
    for (ReductionOperandContext& op_ctx : base_ctx_.operands) {
      OperandState state;
      state.elem_type = mlir::getElementTypeOrSelf(op_ctx.input_tile.getType());
      state.elem_storage_type = xtile::StorageType(state.elem_type);
      state.ptr_to_elem_type =
          ttir::PointerType::get(state.elem_storage_type, kGlobalAddressSpace);
      ABSL_ASSIGN_OR_RETURN(state.layout,
                       xtile::GetPermutationMinorToMajor(
                           op_ctx.input_extract.getSource().getType()));
      if (op_ctx.remote_input_buffers.getType() == ptr_to_i64_type_) {
        state.remote_input_buffers_i64 = op_ctx.remote_input_buffers;
      } else {
        state.remote_input_buffers_i64 = ttir::BitcastOp::create(
            builder_, ptr_to_i64_type_, op_ctx.remote_input_buffers);
      }
      state.buffer_offset = EmitDoubleBufferOffsetFromIndex(
          builder_, buffer_index, op_ctx.num_elements, op_ctx.element_type);
      operand_states_.push_back(std::move(state));
    }
    initialized_ = true;
    return absl::OkStatus();
  }

  // Returns the pointer to the active remote buffer of `operand_idx` for the
  // given rank. See EmitRemoteBufferPtr.
  mlir::Value GetRemoteBufferPtr(int64_t operand_idx, mlir::Value rank_idx) {
    CHECK(initialized_);
    const OperandState& state = operand_states_[operand_idx];
    return EmitRemoteBufferPtr(builder_, state.remote_input_buffers_i64,
                               rank_idx, state.buffer_offset,
                               state.elem_storage_type);
  }

  // Loads a tile of `operand_idx` from the remote buffer of the given rank.
  // Offsets must be global offsets ie, from the beginning of the remote buffer.
  // Shape must be exact shape of the tile to be loaded.
  xtile::TensorValue LoadTileForRank(int64_t operand_idx, mlir::Value rank_idx,
                                     mlir::ValueRange offsets,
                                     llvm::ArrayRef<int64_t> strides,
                                     llvm::ArrayRef<int64_t> shape) {
    CHECK(initialized_);
    const ReductionOperandContext& op_ctx = base_ctx_.operands[operand_idx];
    const OperandState& state = operand_states_[operand_idx];
    mlir::Value remote_buf_ptr = GetRemoteBufferPtr(operand_idx, rank_idx);
    auto [ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_,                      //
        remote_buf_ptr,                // The rank-specific base pointer
        op_ctx.non_tiled_input_shape,  // The full global shape
        state.layout,                  // The layout of the input tensor
        offsets,                       // The global base offsets of the tile
        // The full tile shape. This is the same as the input shape plus 1 for
        // every dimension that is reduced. Since we are not reducing any
        // dimensions, this is the same as the input shape.
        shape,                //
        strides,              //
        /*reduced_dims=*/{},  // Not reducing for gather
        shape                 //
    );
    // tensor<tile_shape, elem_storage_type>
    auto next_tile = mlir::cast<xtile::TensorValue>(
        ttir::LoadOp::create(builder_,                 //
                             ptrs,                     //
                             mask,                     //
                             /*other=*/mlir::Value(),  //
                             /*isVolatile=*/false)
            .getResult());
    // Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // See fusion emitter for more details.
    if (state.elem_storage_type != state.elem_type) {
      next_tile = mlir::cast<xtile::TensorValue>(
          arith::TruncIOp::create(
              builder_, next_tile.getType().clone(state.elem_type), next_tile)
              .getResult());
    }
    return next_tile;
  }

  // Overload for integer rank.
  xtile::TensorValue LoadTileForRank(int64_t operand_idx, int32_t rank,
                                     mlir::ValueRange offsets,
                                     llvm::ArrayRef<int64_t> strides,
                                     llvm::ArrayRef<int64_t> shape) {
    mlir::Value rank_idx = arith::ConstantOp::create(
        builder_, builder_.getI64Type(), builder_.getI64IntegerAttr(rank));
    return LoadTileForRank(operand_idx, rank_idx, offsets, strides, shape);
  }

  // Stores an entire tile of `operand_idx` to the symmetric buffer of
  // device_rank_.
  mlir::LogicalResult EmitCopyToSymmetric(int64_t operand_idx,
                                          mlir::Value tile_to_store,
                                          mlir::ValueRange offsets,
                                          llvm::ArrayRef<int64_t> strides,
                                          llvm::ArrayRef<int64_t> shape) {
    CHECK(initialized_);
    const ReductionOperandContext& op_ctx = base_ctx_.operands[operand_idx];
    const OperandState& state = operand_states_[operand_idx];
    mlir::Value remote_buf_ptr = GetRemoteBufferPtr(operand_idx, device_rank_);

    // Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // The parameter extraction casts the storage type to the logical type.
    // But for copying to the remote buffer we need to cast it back to the
    // storage type. Downstream passes should be able to optimize this away.
    mlir::Value storage_tile = tile_to_store;
    if (state.elem_storage_type != state.elem_type) {
      auto shaped_type = mlir::cast<mlir::ShapedType>(tile_to_store.getType());
      storage_tile = arith::ExtUIOp::create(
          builder_, shaped_type.clone(state.elem_storage_type), tile_to_store);
    }
    auto [ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_,                      //
        remote_buf_ptr,                // The rank-specific base pointer
        op_ctx.non_tiled_input_shape,  // The full global shape
        state.layout,                  // The layout of the input tensor
        offsets,                       // The global base offsets of the tile
        shape,                         // The full tile shape. Same as
                                       // input shape since no
                                       // dimensions are reduced.
        strides,                       //
        /*reduced_dims=*/{},           // Not reducing scatter.
        shape                          // The tile shape.
    );
    ttir::StoreOp::create(builder_, ptrs, storage_tile,
                          /*mask=*/mask);
    return mlir::success();
  }

  mlir::LogicalResult EmitSync(mlir::Value signal_value) {
    CHECK(initialized_);
    EmitBlockBarrier(builder_, signal_buffers_, device_rank_, signal_value,
                     base_ctx_.world_size);
    return mlir::success();
  }

  ReductionEmitterContext& base_ctx_;
  mlir::PatternRewriter& rewriter_;
  mlir::ImplicitLocOpBuilder builder_;

  ReductionComputationEmitter reduce_computation_emitter_{nullptr};

  mlir::Value block_id_;
  mlir::Value device_rank_;
  mlir::Value signal_value_;
  mlir::Value signal_buffers_;
  ttir::PointerType ptr_to_i64_type_;
  llvm::SmallVector<OperandState, 2> operand_states_;

  bool initialized_ = false;
};

class AllReduceEmitter : public ReductionEmitter {
 public:
  static mlir::LogicalResult Emit(AllReduceEmitterContext ctx,
                                  mlir::PatternRewriter& rewriter) {
    AllReduceEmitter emitter(ctx, rewriter);
    if (auto result = emitter.Initialize(); !result.ok()) {
      LOG(ERROR) << "Failed to initialize AllReduceEmitter: "
                 << result.message();
      return mlir::failure();
    }
    switch (emitter.ctx_.strategy) {
      case AllReduceStrategy::kOneShot:
        return emitter.EmitOneShot();
      case AllReduceStrategy::kTwoShot:
        return emitter.EmitTwoShot();
      case AllReduceStrategy::kMultimem:
        return emitter.rewriter_.notifyMatchFailure(
            emitter.ctx_.op->getLoc(),
            "Multimem all-reduce is not yet supported for codegeneration.");
    }
  }

 private:
  AllReduceEmitter(AllReduceEmitterContext& ctx,
                   mlir::PatternRewriter& rewriter)
      : ReductionEmitter(ctx, rewriter), ctx_(ctx) {}

  absl::Status Initialize() {
    ABSL_RETURN_IF_ERROR(ReductionEmitter::Initialize(
        /*is_two_shot=*/ctx_.strategy == AllReduceStrategy::kTwoShot));
    subtile_shapes_.reserve(ctx_.operands.size());
    for (int64_t i = 0; i < ctx_.operands.size(); ++i) {
      const ReductionOperandContext& operand = ctx_.operands[i];
      const OperandState& state = operand_states_[i];
      const llvm::ArrayRef<int64_t>& input_tile_shape_dims =
          operand.input_tile.getType().getShape();
      Shape input_tile_shape = ShapeUtil::MakeShapeWithDenseLayout(
          operand.element_type, input_tile_shape_dims, state.layout);
      // Subtile shape for one-shot is the same as the input tile shape.
      llvm::SmallVector<int64_t> subtile_shape(input_tile_shape_dims.begin(),
                                               input_tile_shape_dims.end());
      // For two-shot, divide the tile into num_devices tiles.
      if (ctx_.strategy == AllReduceStrategy::kTwoShot) {
        subtile_shape =
            GreedyPowerOfTwoTiles(input_tile_shape, ctx_.world_size);
        // Make sure subtile shape perfectly divides the input tile shape.
        // Crash Ok. This is an internal precondition which is always expected
        // to be true. Internal tile shape is 2^n / 2^m should be perfectly
        // divisible for n >= m.
        CHECK_EQ(
            Product(input_tile_shape.dimensions()) % Product(subtile_shape), 0)
            << "Input tile shape is not perfectly divisible by subtile shape."
            << "Input tile shape: " << input_tile_shape
            << "Subtile shape: " << absl::StrJoin(subtile_shape, ",");
      }
      subtile_shapes_.push_back(std::move(subtile_shape));
    }
    return absl::OkStatus();
  }

  // Returns the offsets of the sub-tile that the given rank is responsible for.
  // These are local offsets with respect to the tile.
  // For loading from the remote buffer, we need to add the global offsets.
  llvm::SmallVector<mlir::Value> RankToLocalOffsets(
      int64_t operand_idx, mlir::Value rank_idx,
      llvm::ArrayRef<int64_t> tile_dims, llvm::ArrayRef<int64_t> subtile_dims) {
    const OperandState& state = operand_states_[operand_idx];
    int32_t num_dims = tile_dims.size();
    // Calculate the number of subtiles along each dimension
    // e.g., Tile(1, 4, 32) / Subtile(1, 2, 16) -> D = [1, 2, 2]
    // We know that tile_dims[i] is divisible by subtile_dims[i].
    // See precondition during Initialize.
    llvm::SmallVector<int64_t> dimensions(num_dims, 0);
    for (int64_t i = 0; i < num_dims; ++i) {
      dimensions[i] = tile_dims[i] / subtile_dims[i];
    }
    if (!rank_idx.getType().isInteger(64)) {
      rank_idx =
          arith::ExtUIOp::create(builder_, builder_.getI64Type(), rank_idx);
    }
    // Decompose rank_idx based on layout minor-to-major.
    llvm::SmallVector<mlir::Value, 4> offsets(num_dims);
    for (int32_t i = 0; i < num_dims; ++i) {
      int layout_dim = state.layout[i];
      mlir::Value dimension = arith::ConstantOp::create(
          builder_, builder_.getI64Type(),
          builder_.getI64IntegerAttr(dimensions[layout_dim]));
      mlir::Value stride = arith::ConstantOp::create(
          builder_, builder_.getI64Type(),
          builder_.getI64IntegerAttr(subtile_dims[layout_dim]));
      offsets[layout_dim] = arith::MulIOp::create(
          builder_, stride,
          arith::RemSIOp::create(builder_, rank_idx, dimension));
      rank_idx = arith::DivSIOp::create(builder_, rank_idx, dimension);
    }
    return offsets;
  }

  // Calculates the offsets and shape of the sub-tile that the given rank is
  // responsible for. Only valid for two-shot.
  // Offsets are calculated as input extract + local offsets for supplied rank.
  // Note: The offsets are global offsets ie, from the beginning of the input
  // buffer. So for example if the num_elements is 1024 and the tiles are of
  // size 512 with 2 ranks. For two-shot, each ranks responsibility is 512/2 =
  // 256 per tile. Output shape is always [256]. Offsets are:
  //  - Rank0Tile0: [0]; Rank0Tile1: [512]
  //  - Rank1Tile0: [256]; Rank1Tile1: [768]
  llvm::SmallVector<mlir::Value> CalculateSubtileOffsets(int64_t operand_idx,
                                                         mlir::Value rank_idx) {
    ReductionOperandContext& operand = ctx_.operands[operand_idx];
    auto local_offsets = RankToLocalOffsets(
        operand_idx, rank_idx, operand.input_tile.getType().getShape(),
        subtile_shapes_[operand_idx]);
    // Global offsets
    llvm::SmallVector<mlir::Value> offsets = operand.input_extract.getOffsets();
    for (int i = 0; i < local_offsets.size(); ++i) {
      offsets[i] = arith::IndexCastOp::create(builder_, builder_.getI64Type(),
                                              offsets[i]);
      offsets[i] =
          arith::AddIOp::create(builder_, local_offsets[i], offsets[i]);
    }
    return offsets;
  }

  // Create a tensor of world_size rank_ids and loads it.
  // It then reshapes it to the number of sub-tiles in each dimension.
  //
  // Eg: tensor<0..7> with a tile size of 128,128 and subtile_shape of <32, 64>
  // implies the tensor is reshaped to 4, 2.
  // Then each dimension is broadcast to the subtile shape.
  // so tensor<4, 2> becomes tensor<4, 32, 2, 64>.
  // This is finally reshaped to the tile shape tensor<128, 128>.
  // The finally result is then a tensor of pointers to the remote buffers of
  // each rank. If a load is performed with this tensor, it will gather the
  // first element from the remote buffers of each rank based on the
  // responsibility of each rank.
  //
  // Returns: tensor<tile_shape x !ptr<element_type>>
  xtile::TensorValue CreateTensorOfRemoteBufferPtrs(
      int64_t operand_idx, mlir::RankedTensorType tile_type,
      llvm::ArrayRef<int64_t> subtile_shape, int64_t world_size) {
    const OperandState& state = operand_states_[operand_idx];
    const llvm::ArrayRef<int64_t> tile_shape = tile_type.getShape();
    const mlir::RankedTensorType tensor_of_world_size_ptr_of_ptrs =
        mlir::RankedTensorType::get({world_size}, ptr_to_i64_type_);
    mlir::Value remote_buffers =
        ttir::SplatOp::create(builder_, tensor_of_world_size_ptr_of_ptrs,
                              state.remote_input_buffers_i64);
    const mlir::Value rank_ids = ttir::MakeRangeOp::create(
        builder_,
        mlir::RankedTensorType::get({world_size}, builder_.getI32Type()), 0,
        world_size);
    const mlir::Value rank_ids_i64 = arith::ExtUIOp::create(
        builder_,
        mlir::RankedTensorType::get({world_size}, builder_.getI64Type()),
        rank_ids);
    // tensor<world_size x !ptr<i64>> where each pointer is the base address of
    // the remote buffer of rank i.
    remote_buffers =
        ttir::AddPtrOp::create(builder_, tensor_of_world_size_ptr_of_ptrs,
                               remote_buffers, rank_ids_i64);
    // Load the 64-bit addresses from the table
    // tensor<world_size x i64>
    remote_buffers = ttir::LoadOp::create(builder_,                 //
                                          remote_buffers,           //
                                          /*mask=*/mlir::Value(),   //
                                          /*other=*/mlir::Value(),  //
                                          /*isVolatile=*/false)
                         .getResult();
    // tensor<world_size x !ptr<elem_type>>
    const mlir::RankedTensorType tensor_of_world_size_ptrs =
        mlir::RankedTensorType::get({world_size}, state.ptr_to_elem_type);
    remote_buffers = ttir::IntToPtrOp::create(
        builder_, tensor_of_world_size_ptrs, remote_buffers,
        {xtile::GetDivisibilityAttr(builder_)});
    mlir::Value buffer_offsets = ttir::SplatOp::create(
        builder_,
        mlir::RankedTensorType::get({world_size}, builder_.getI64Type()),
        state.buffer_offset);
    // Add the buffer_offset to the base of each loaded pointer.
    remote_buffers = ttir::AddPtrOp::create(builder_, tensor_of_world_size_ptrs,
                                            remote_buffers, buffer_offsets);
    // Add a new dimension of size 1 to the tensor of pointers.
    // tensor<world_size x 1 x !ptr<elem_type>>
    remote_buffers =
        ttir::ExpandDimsOp::create(builder_, remote_buffers, /*axis=*/1);
    // Broadcast to RxNumElementsPerRank
    mlir::Type broadcasted_type = mlir::RankedTensorType::get(
        {world_size, Product(subtile_shape)}, state.ptr_to_elem_type);
    remote_buffers =
        ttir::BroadcastOp::create(builder_, broadcasted_type, remote_buffers);
    llvm::SmallVector<int64_t> physical_shape(tile_shape.size(), 0);
    bool requires_transpose = false;
    for (int i = 0; i < tile_shape.size(); ++i) {
      const auto idx = state.layout[state.layout.size() - i - 1];
      physical_shape[i] = tile_shape[idx];
      if (state.layout[i] != state.layout.size() - 1 - i) {
        requires_transpose = true;
      }
    }
    // Reshape to tile shape.
    remote_buffers = ttir::ReshapeOp::create(
        builder_,
        mlir::RankedTensorType::get(physical_shape, state.ptr_to_elem_type),
        remote_buffers);
    if (requires_transpose) {
      llvm::SmallVector<int32_t> permutation(state.layout.size());
      for (int i = 0; i < state.layout.size(); ++i) {
        permutation[state.layout[i]] = state.layout.size() - 1 - i;
      }
      remote_buffers = ttir::TransOp::create(
          builder_,
          mlir::RankedTensorType::get(tile_shape, state.ptr_to_elem_type),
          remote_buffers, permutation);
    }
    return mlir::cast<xtile::TensorValue>(remote_buffers);
  }

  // Emits instructions to load a tile from the remote buffers of all ranks
  // based on the responsibility of each rank.
  // So for a tile of size T with R ranks, each rank will be responsible for
  // T / R elements.
  // This method will arrange the pointers to size T such that
  // first T/R elements point to rank 0, next T/R elements point to rank 1, etc.
  // For each dimension.
  // And then perform a tensor<Tx!ptr<elem_type>> gathered load.
  absl::StatusOr<xtile::TensorValue> EmitGatherLoad(int64_t operand_idx) {
    ReductionOperandContext& operand = ctx_.operands[operand_idx];
    const OperandState& state = operand_states_[operand_idx];
    mlir::RankedTensorType tile_type = operand.input_tile.getType();
    const mlir::ArrayRef<int64_t> tile_shape = tile_type.getShape();
    // tensor<tile_shape x !ptr<elem_type>>
    // Each pointer within subtile i points to data in the remote buffer of
    // rank i.
    xtile::TensorValue remote_buffer_tensor = CreateTensorOfRemoteBufferPtrs(
        operand_idx, tile_type, subtile_shapes_[operand_idx], ctx_.world_size);
    auto [final_ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_,              //
        remote_buffer_tensor,  // The tensor of rank-specific base pointers
        operand.non_tiled_input_shape,             // The full global shape
        state.layout,                              //
        operand.input_extract.getOffsets(),        // The global base offsets
        operand.input_extract.getFullTileShape(),  //
        operand.input_extract.getStrides(),
        /*reduced_dims=*/{},  // Not reducing for gather
        tile_shape);
    // The final gather load tensor<tile_shape x elem_type>
    return mlir::cast<xtile::TensorValue>(
        ttir::LoadOp::create(builder_,                 //
                             final_ptrs,               //
                             mask,                     //
                             /*other=*/mlir::Value(),  //
                             /*isVolatile=*/false)
            .getResult());
  }

  mlir::LogicalResult EmitOneShot() {
    CHECK(initialized_);
    const int64_t num_operands = ctx_.operands.size();
    llvm::SmallVector<llvm::SmallVector<mlir::Value>, 2> all_offsets(
        num_operands);
    llvm::SmallVector<llvm::SmallVector<int64_t>, 2> all_strides(num_operands);
    llvm::SmallVector<bool, 2> is_scalar(num_operands, false);

    // 1. CopyPhase: Local tiles to the symmetric buffers for the current
    // device.
    for (int64_t i = 0; i < num_operands; ++i) {
      ReductionOperandContext& operand = ctx_.operands[i];
      OperandState& state = operand_states_[i];
      all_offsets[i].assign(operand.input_extract.getOffsets().begin(),
                            operand.input_extract.getOffsets().end());
      all_strides[i].assign(operand.input_extract.getStrides().begin(),
                            operand.input_extract.getStrides().end());
      is_scalar[i] = all_offsets[i].empty();
      // Lift scalar inputs to a tensor of shape {1}.
      if (is_scalar[i]) {
        operand.input_tile = xtile::Splat(builder_, operand.input_tile, {1});
        operand.non_tiled_input_shape = {1};
        state.layout = {0};
        subtile_shapes_[i] = {1};
        mlir::Value c0 = builder_.create<arith::ConstantIndexOp>(0);
        all_offsets[i] = {c0};
        all_strides[i] = {1};
      }
      if (mlir::failed(EmitCopyToSymmetric(
              i, operand.input_tile, all_offsets[i], all_strides[i],
              operand.input_tile.getType().getShape()))) {
        return rewriter_.notifyMatchFailure(ctx_.op,
                                            "Failed to emit copy to symmetric");
      }
    }
    // 2. Synchronization phase: Wait for all ranks to complete the copy.
    if (mlir::failed(EmitSync(signal_value_))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit sync for one-shot");
    }
    // 3. Reduce phase: Load tiles from all ranks and reduce them.
    llvm::SmallVector<mlir::Value, 2> results;
    results.reserve(num_operands);
    for (int64_t i = 0; i < num_operands; ++i) {
      ReductionOperandContext& operand = ctx_.operands[i];
      OperandState& state = operand_states_[i];
      llvm::ArrayRef<int64_t> shape = operand.input_tile.getType().getShape();
      xtile::TensorValue accumulator =
          LoadTileForRank(i, 0, all_offsets[i], all_strides[i], shape);
      for (int32_t rank = 1; rank < ctx_.world_size; ++rank) {
        xtile::TensorValue next_tile =
            LoadTileForRank(i, rank, all_offsets[i], all_strides[i], shape);
        accumulator =
            reduce_computation_emitter_(builder_, accumulator, next_tile);
      }
      mlir::Value result = accumulator;
      if (is_scalar[i]) {
        result = ttir::UnsplatOp::create(builder_, accumulator);
        result = mlir::tensor::FromElementsOp::create(
            builder_, mlir::RankedTensorType::get({}, state.elem_type), result);
      }
      results.push_back(result);
    }
    rewriter_.replaceOp(ctx_.op, results);
    return mlir::success();
  }

  mlir::LogicalResult EmitTwoShot() {
    CHECK(initialized_);
    const int64_t num_operands = ctx_.operands.size();
    // 1. CopyPhase: Local tiles to the symmetric buffers for the current
    // device.
    for (int64_t i = 0; i < num_operands; ++i) {
      ReductionOperandContext& operand = ctx_.operands[i];
      if (mlir::failed(EmitCopyToSymmetric(
              i, operand.input_tile, operand.input_extract.getOffsets(),
              operand.input_extract.getStrides(),
              operand.input_tile.getType().getShape()))) {
        return rewriter_.notifyMatchFailure(ctx_.op,
                                            "Failed to emit copy to symmetric");
      }
    }
    // 2. Shot1: Wait for all ranks to complete the copy.
    if (mlir::failed(EmitSync(signal_value_))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit sync for shot1");
    }
    // 3. Reduce phase:
    for (int64_t i = 0; i < num_operands; ++i) {
      ReductionOperandContext& operand = ctx_.operands[i];
      llvm::ArrayRef<int64_t> strides = operand.input_extract.getStrides();
      // 3.1 Accumulate what each rank is responsible for.
      llvm::SmallVector<mlir::Value> self_offsets =
          CalculateSubtileOffsets(i, device_rank_);
      xtile::TensorValue accumulator =
          LoadTileForRank(i, 0, self_offsets, strides, subtile_shapes_[i]);
      for (int rank = 1; rank < ctx_.world_size; ++rank) {
        xtile::TensorValue next_tile =
            LoadTileForRank(i, rank, self_offsets, strides, subtile_shapes_[i]);
        accumulator =
            reduce_computation_emitter_(builder_, accumulator, next_tile);
      }
      // 3.2 Copy reduced sub-tile back to local rank's remote buffer.
      if (mlir::failed(EmitCopyToSymmetric(i, accumulator, self_offsets,
                                           strides, subtile_shapes_[i]))) {
        return rewriter_.notifyMatchFailure(
            ctx_.op, "Failed to emit copy result to symmetric for two-shot");
      }
    }
    // 4. Shot2: Wait for all ranks to complete the reduce.
    mlir::Value next_signal_value = arith::AddIOp::create(
        builder_, signal_value_,
        arith::ConstantOp::create(builder_, builder_.getI32Type(),
                                  builder_.getI32IntegerAttr(1)));
    if (mlir::failed(EmitSync(next_signal_value))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit sync for shot2");
    }
    // 5. Gather from all ranks to output tiles.
    llvm::SmallVector<mlir::Value, 2> results;
    results.reserve(num_operands);
    for (int64_t i = 0; i < num_operands; ++i) {
      absl::StatusOr<xtile::TensorValue> gathered_tensor = EmitGatherLoad(i);
      if (!gathered_tensor.ok()) {
        return rewriter_.notifyMatchFailure(
            ctx_.op, absl::StrCat("Failed to emit gathered load: ",
                                  gathered_tensor.status().message()));
      }
      results.push_back(gathered_tensor.value());
    }
    rewriter_.replaceOp(ctx_.op, results);
    return mlir::success();
  }

  AllReduceEmitterContext& ctx_;
  // Calculated per-operand sub-tile shapes for the all reduce.
  // For 1-shot this is the same as the tile shape. Since each rank operates on
  // the entire tile.
  // For 2-shot: The sub-tile is the tile shape divided by the
  // number of devices. Since tile shape is a power of 2 and the number of
  // devices is a power of 2, the sub-tile shape will also be a power of 2 and
  // consequently work with the tiling infra.
  llvm::SmallVector<llvm::SmallVector<int64_t>, 2> subtile_shapes_;
};

class ReduceScatterEmitter : public ReductionEmitter {
 public:
  static mlir::LogicalResult Emit(ReduceScatterEmitterContext ctx,
                                  mlir::PatternRewriter& rewriter) {
    ReduceScatterEmitter emitter(ctx, rewriter);
    if (auto result = emitter.Initialize(); !result.ok()) {
      LOG(ERROR) << "Failed to initialize ReduceScatterEmitter: "
                 << result.message();
      return mlir::failure();
    }
    return emitter.EmitImpl();
  }

 private:
  ReduceScatterEmitter(ReduceScatterEmitterContext& ctx,
                       mlir::PatternRewriter& rewriter)
      : ReductionEmitter(ctx, rewriter), ctx_(ctx) {}

  // Pull-based one-shot reduce-scatter.
  //
  // The input is [world_size * shard_size, ...] and the output is
  // [shard_size, ...] (scatter dimension `d`). The tiling guarantees that the
  // input tile of every program (`output_tile[d] * world_size` rows starting at
  // `output_offset[d] * world_size`) lies within a single shard, so program
  // `pid` produces a tile destined to exactly one rank.
  //   1. Copy phase: every program copies its local input tile into the local
  //      rank's symmetric buffer. This replaces a host-launched D2D copy of the
  //      input into the scratch buffer.
  //   2. Sync phase: a producer-symmetric block barrier. Every program signals
  //      its own slot, and waits only for the (unique) program on every peer
  //      that staged the input tile this program reduces.
  //   3. Reduce phase: pull the output tile from the symmetric buffers of all
  //      ranks (at shard `rank`) and reduce them in rank order.
  mlir::LogicalResult EmitImpl() {
    CHECK(initialized_);
    const int64_t scatter_dim = ctx_.scatter_dimension;

    // 1. CopyPhase: local input tiles to the symmetric buffers of this rank.
    for (int64_t i = 0; i < ctx_.operands.size(); ++i) {
      ReductionOperandContext& operand = ctx_.operands[i];
      if (mlir::failed(EmitCopyToSymmetric(
              i, operand.input_tile, operand.input_extract.getOffsets(),
              operand.input_extract.getStrides(),
              operand.input_tile.getType().getShape()))) {
        return rewriter_.notifyMatchFailure(ctx_.op,
                                            "Failed to emit copy to symmetric");
      }
    }

    // 2. Synchronization phase: wait for the producers of our input tiles.
    mlir::Value signal_slot = arith::DivUIOp::create(
        builder_, block_id_,
        arith::ConstantOp::create(
            builder_, builder_.getI32IntegerAttr(
                          static_cast<int32_t>(ctx_.signal_stride))));
    EmitBlockBarrier(builder_, signal_buffers_, device_rank_, signal_value_,
                     ctx_.world_size, signal_slot, ctx_.signal_stride,
                     mtx::BarrierMode::kProducerSymmetric);

    // 3. Reduce phase: pull this rank's output tiles from every peer.
    //
    // Each block copies one input tile (step 1) and writes one output tile.
    // The input tile is R (world_size) times the output tile. Example with
    // R = 2, input f32[16] -> output f32[8] and two blocks, on rank 0:
    //   block 0: copies input 0..8 (for rank 0). Writes output 0..4.
    //   block 1: copies input 8..16 (for rank 1). Writes output 4..8.
    mlir::Value local_rank = arith::IndexCastOp::create(
        builder_, builder_.getIndexType(), device_rank_);
    mlir::Value copy_rank = arith::IndexCastOp::create(
        builder_, builder_.getIndexType(), signal_slot);
    llvm::SmallVector<mlir::Value, 2> results;
    results.reserve(ctx_.operands.size());
    for (int64_t i = 0; i < ctx_.operands.size(); ++i) {
      ReductionOperandContext& operand = ctx_.operands[i];
      llvm::SmallVector<mlir::Value> pull_offsets =
          operand.input_extract.getOffsets();
      llvm::SmallVector<int64_t> strides{operand.input_extract.getStrides()};
      mlir::Value per_rank_size =
          arith::ConstantIndexOp::create(builder_, operand.per_rank_size);
      mlir::Value copy_rank_offset =
          arith::MulIOp::create(builder_, copy_rank, per_rank_size);
      mlir::Value intra_rank_offset = arith::SubIOp::create(
          builder_, pull_offsets[scatter_dim], copy_rank_offset);
      mlir::Value subtile_size = arith::ConstantIndexOp::create(
          builder_, operand.output_tile_shape[scatter_dim]);
      mlir::Value subtile_offset =
          arith::MulIOp::create(builder_, copy_rank, subtile_size);
      mlir::Value output_offset =
          arith::AddIOp::create(builder_, intra_rank_offset, subtile_offset);
      mlir::Value local_rank_offset =
          arith::MulIOp::create(builder_, local_rank, per_rank_size);
      pull_offsets[scatter_dim] =
          arith::AddIOp::create(builder_, local_rank_offset, output_offset);

      // Accumulate in the element type (rounding after every step) to match the
      // numerics of NCCL and the Triton all-reduce kernel.
      auto load = [&](int32_t rank) -> xtile::TensorValue {
        return LoadTileForRank(i, rank, pull_offsets, strides,
                               operand.output_tile_shape);
      };
      xtile::TensorValue accumulator = load(0);
      for (int32_t rank = 1; rank < ctx_.world_size; ++rank) {
        accumulator =
            reduce_computation_emitter_(builder_, accumulator, load(rank));
      }
      results.push_back(accumulator);
    }
    rewriter_.replaceOp(ctx_.op, results);
    return mlir::success();
  }

  ReduceScatterEmitterContext& ctx_;
};

// Heuristic maximum warps per block across collective kernels.
// Triton expects a power of 2 for num_warps.
constexpr int64_t kMaxWarpsPerBlock = 16;

BlockLevelFusionConfig CreateBlockLevelFusionConfig(
    int64_t num_warps,
    absl::Span<const llvm::SmallVector<int64_t>> output_tiles) {
  BlockLevelFusionConfig config;
  config.set_num_warps(num_warps);
  config.set_num_ctas(1);    // No block-level clustering.
  config.set_num_stages(1);  // No pipelining of loops.
  for (const llvm::SmallVector<int64_t>& tile_sizes : output_tiles) {
    config.add_output_tiles()->mutable_sizes()->Assign(tile_sizes.begin(),
                                                       tile_sizes.end());
  }
  return config;
}

BlockLevelFusionConfig CreateBlockLevelFusionConfig(
    int64_t num_warps, const llvm::SmallVector<int64_t>& tile_sizes) {
  return CreateBlockLevelFusionConfig(
      num_warps, absl::MakeConstSpan(&tile_sizes, /*size=*/1));
}

// Tiles one rank's part of a collective output across `blocks_per_rank`
// programs. `collective_dim` is first divided by `num_devices` (R).
//
// All-gather: output f32[16] on 2 ranks -> one rank's part is f32[8].
//
// Reduce-scatter: output f32[8] on 2 ranks. Each block copies one input tile
// and writes one output tile. The input tile is R times the output tile.
// With two blocks on rank 0:
//   block 0: copies input 0..8 (for rank 0). Reduces and writes output 0..4.
//   block 1: copies input 8..16 (for rank 1). Reduces and writes output 4..8.
absl::StatusOr<llvm::SmallVector<int64_t>> GetCollectiveOutputTile(
    const Shape& output_shape, int64_t collective_dim, int64_t num_devices,
    int32_t blocks_per_rank) {
  TF_RET_CHECK(output_shape.dimensions(collective_dim) % num_devices == 0)
      << "Collective output " << output_shape.ToString()
      << " must be divisible by " << num_devices << " along dimension "
      << collective_dim;
  const int64_t per_rank_size =
      output_shape.dimensions(collective_dim) / num_devices;
  Shape per_rank_shape = output_shape;
  per_rank_shape.set_dimensions(collective_dim, per_rank_size);
  llvm::SmallVector<int64_t> tile_sizes =
      GreedyPowerOfTwoTiles(per_rank_shape, blocks_per_rank);
  tile_sizes[collective_dim] = std::min(
      tile_sizes[collective_dim], static_cast<int64_t>(llvm::bit_floor(
                                      static_cast<uint64_t>(per_rank_size))));
  return tile_sizes;
}

// Returns the output shapes of `instr` as a span (a 1-element span for
// array-shaped instructions, or `tuple_shapes()` for tuple-shaped ones).
absl::Span<const Shape> GetOutputShapes(const HloInstruction* instr) {
  return instr->shape().IsTuple()
             ? absl::MakeConstSpan(instr->shape().tuple_shapes())
             : absl::MakeConstSpan(&instr->shape(), 1);
}

// Computes one output tile per output of an all-gather or reduce-scatter
// `instr` (see GetCollectiveOutputTile). Each output is tiled independently
// using `num_blocks / num_devices` programs per rank; for variadic collectives,
// smaller outputs get padded with masked-out blocks by
// `GetIndexWiseVariadicSchedule`.
absl::StatusOr<std::vector<llvm::SmallVector<int64_t>>>
GetCollectiveOutputTiles(const HloInstruction* instr, int64_t collective_dim,
                         int64_t num_devices, int32_t num_blocks) {
  const int32_t blocks_per_rank =
      std::max<int32_t>(1, num_blocks / static_cast<int32_t>(num_devices));
  const absl::Span<const Shape> output_shapes = GetOutputShapes(instr);
  std::vector<llvm::SmallVector<int64_t>> tiles;
  tiles.reserve(output_shapes.size());
  for (const Shape& output_shape : output_shapes) {
    ABSL_ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> tile_sizes,
                     GetCollectiveOutputTile(output_shape, collective_dim,
                                             num_devices, blocks_per_rank));
    tiles.push_back(std::move(tile_sizes));
  }
  return tiles;
}

absl::StatusOr<BlockLevelFusionConfig> GetBlockLevelConfigForAllReduce(
    const GpuTopology& gpu_topology, const HloAllReduceInstruction* instr,
    const DeviceAssignment* device_assignment) {
  // Heuristic maximum blocks per grid after benchmarking. Power of 2 so blocks
  // are consumed cleanly in powers of 2 from major to minor dimensions.
  static constexpr int64_t kAllReduceMaxBlocksPerGrid = 32;

  ABSL_ASSIGN_OR_RETURN(const AllReduceInfo all_reduce_info,
                   BuildAllReduceInfo(/*is_multimem_enabled=*/false,
                                      gpu_topology, instr, device_assignment));

  const int64_t elements_per_warp =
      WarpSize(gpu_topology.gpu_target_config().device_description) *
      se::gpu::kNumElementsPerThread;
  const int64_t elements_per_rank =
      all_reduce_info.num_elements /
      (all_reduce_info.all_reduce_strategy == AllReduceStrategy::kTwoShot
           ? all_reduce_info.num_devices
           : 1);
  const int64_t total_warps = CeilOfRatio(elements_per_rank, elements_per_warp);
  const int64_t num_warps = std::min<int64_t>(
      kMaxWarpsPerBlock, llvm::bit_ceil(static_cast<uint64_t>(total_warps)));
  const int64_t num_blocks =
      std::min(kAllReduceMaxBlocksPerGrid, CeilOfRatio(total_warps, num_warps));

  llvm::SmallVector<int64_t> tile_sizes =
      GreedyPowerOfTwoTiles(instr->shape(), num_blocks);
  const int64_t linear_tile_size = Product(tile_sizes);
  if (all_reduce_info.all_reduce_strategy == AllReduceStrategy::kTwoShot &&
      linear_tile_size % all_reduce_info.num_devices != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Two-shot all-reduce linear_tile_size(%d) %% num_devices(%d) != 0. "
        "Codegen will not be supported.",
        linear_tile_size, all_reduce_info.num_devices));
  }

  return CreateBlockLevelFusionConfig(num_warps, tile_sizes);
}

absl::StatusOr<BlockLevelFusionConfig> GetBlockLevelConfigForAllGather(
    const GpuTopology& gpu_topology, const HloAllGatherInstruction* instr,
    const DeviceAssignment* device_assignment) {
  ABSL_ASSIGN_OR_RETURN(const AllGatherInfo all_gather_info,
                   BuildAllGatherInfo(gpu_topology, instr, device_assignment));

  const int64_t elements_per_warp =
      WarpSize(gpu_topology.gpu_target_config().device_description) *
      se::gpu::kNumElementsPerThread;
  // All-gather always uses a one-shot strategy: every rank reads its peers'
  // slices directly, so the work is proportional to the full output element
  // count with no rank-based partitioning.
  const int64_t total_warps =
      CeilOfRatio(all_gather_info.max_output_elements, elements_per_warp);
  const int64_t num_warps = std::min<int64_t>(
      kMaxWarpsPerBlock, llvm::bit_ceil(static_cast<uint64_t>(total_warps)));
  const int64_t num_blocks =
      std::min(kAllGatherMaxBlocksPerGrid, CeilOfRatio(total_warps, num_warps));

  ABSL_ASSIGN_OR_RETURN(
      std::vector<llvm::SmallVector<int64_t>> output_tiles,
      GetCollectiveOutputTiles(instr, instr->all_gather_dimension(),
                               all_gather_info.num_devices, num_blocks));
  return CreateBlockLevelFusionConfig(num_warps, output_tiles);
}

absl::StatusOr<BlockLevelFusionConfig> GetBlockLevelConfigForReduceScatter(
    const GpuTopology& gpu_topology, const HloReduceScatterInstruction* instr,
    const DeviceAssignment* device_assignment) {
  ABSL_ASSIGN_OR_RETURN(
      const ReduceScatterInfo reduce_scatter_info,
      BuildReduceScatterInfo(gpu_topology, instr, device_assignment));

  const int64_t num_devices = reduce_scatter_info.num_devices;
  const int64_t elements_per_warp =
      WarpSize(gpu_topology.gpu_target_config().device_description) *
      se::gpu::kNumElementsPerThread;
  const int64_t elements_per_rank_tile =
      CeilOfRatio(reduce_scatter_info.num_output_elements, num_devices);
  const int64_t warps_per_rank_tile =
      CeilOfRatio(elements_per_rank_tile, elements_per_warp);
  const int64_t num_warps = std::min<int64_t>(
      kMaxWarpsPerBlock,
      llvm::bit_ceil(static_cast<uint64_t>(warps_per_rank_tile)));
  const int64_t blocks_per_rank_tile =
      CeilOfRatio(warps_per_rank_tile, num_warps);
  const int64_t num_blocks =
      std::min(kReduceScatterMaxBlocksPerGrid,
               num_devices * static_cast<int64_t>(llvm::bit_floor(
                                 static_cast<uint64_t>(blocks_per_rank_tile))));

  ABSL_ASSIGN_OR_RETURN(std::vector<llvm::SmallVector<int64_t>> output_tiles,
                   GetCollectiveOutputTiles(instr, instr->scatter_dimension(),
                                            num_devices, num_blocks));
  return CreateBlockLevelFusionConfig(num_warps, output_tiles);
}

absl::StatusOr<BlockLevelFusionConfig> GetBlockLevelConfigForAllToAll(
    const GpuTopology& gpu_topology, const HloAllToAllInstruction* instr,
    const DeviceAssignment* device_assignment) {
  static constexpr int64_t kAllToAllMaxBlocksPerGrid = 64;

  ABSL_ASSIGN_OR_RETURN(const AllToAllInfo all_to_all_info,
                   BuildAllToAllInfo(gpu_topology, instr, device_assignment));

  const int64_t num_devices = all_to_all_info.num_devices;
  const int64_t elements_per_warp =
      WarpSize(gpu_topology.gpu_target_config().device_description) *
      se::gpu::kNumElementsPerThread;
  const int64_t elements_per_rank =
      CeilOfRatio(all_to_all_info.num_elements, num_devices);
  const int64_t warps_per_rank =
      CeilOfRatio(elements_per_rank, elements_per_warp);
  const int64_t num_warps = std::min<int64_t>(
      kMaxWarpsPerBlock, llvm::bit_ceil(static_cast<uint64_t>(warps_per_rank)));
  const int64_t blocks_per_rank = CeilOfRatio(warps_per_rank, num_warps);
  const int64_t num_blocks =
      std::min(kAllToAllMaxBlocksPerGrid, num_devices * blocks_per_rank);

  const Shape& output_shape = instr->shape();
  const int64_t split_dim = *instr->split_dimension();
  const int64_t per_rank_size =
      output_shape.dimensions(split_dim) / num_devices;
  Shape per_rank_shape = output_shape;
  per_rank_shape.set_dimensions(split_dim, per_rank_size);
  llvm::SmallVector<int64_t> tile_sizes = GreedyPowerOfTwoTiles(
      per_rank_shape, std::max<int32_t>(1, num_blocks / num_devices));

  return CreateBlockLevelFusionConfig(num_warps, tile_sizes);
}

// Per-operand state of an all-gather rewrite.
struct AllGatherOperandContext {
  // Tile of the local input that feeds the all-gather. For booleans this is
  // the i8 -> i1 cast of `input_extract` (see i1_to_i8_workaround).
  mlir::Value input_tile;
  xtile::ExtractTileOp input_extract;
  // Local input buffer read by `input_extract`.
  mlir::TypedValue<mlir::MemRefType> input_buffer;
  // !tt.ptr<i64> table of the gathered parameter's per-rank remote buffers.
  mlir::Value remote_buffers_arg;
  int64_t split_dim = -1;
  PrimitiveType element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID;
  llvm::ArrayRef<int64_t> per_rank_shape;
  llvm::ArrayRef<int64_t> tile_sizes;
};

struct AllGatherRewriteContext {
  mlir::stablehlo::AllGatherOp op;
  llvm::SmallVector<AllGatherOperandContext, 2> operands;
  mlir::Value rank_arg;
  mlir::Value signal_buffers_arg;
  int64_t world_size = 0;
  int64_t gather_dim = 0;
  // Number of consecutive programs that read from the same peer. See
  // EmitPeerReplicaId.
  int64_t signal_stride = 0;
};

// Returns the number of programs needed to cover one rank's slice of an
// operand, i.e. the product of the per-dimension tile counts.
int64_t ProgramsPerRank(const AllGatherOperandContext& operand) {
  int64_t programs = 1;
  for (auto [dim_size, tile_size] :
       llvm::zip(operand.per_rank_shape, operand.tile_sizes)) {
    programs *= xla::CeilOfRatio(dim_size, tile_size);
  }
  return programs;
}

absl::StatusOr<AllGatherOperandContext> BuildAllGatherOperandContext(
    mlir::stablehlo::AllGatherOp op, int64_t operand_index,
    xtile::EntryFuncOp entry_func, int64_t num_parameters,
    int32_t metadata_args_start, int64_t world_size, int64_t gather_dim) {
  ABSL_ASSIGN_OR_RETURN(
      CollectiveOperandExtractInfo operand_info,
      ExtractCollectiveOperand(
          op.getOperand(operand_index), operand_index, "AllGather", entry_func,
          num_parameters, metadata_args_start + kNumCollectiveMetadataArgs));
  AllGatherOperandContext ctx;
  ctx.input_tile = operand_info.input_tile;
  ctx.input_extract = operand_info.input_extract;
  ctx.input_buffer = operand_info.input_buffer;
  ctx.remote_buffers_arg = operand_info.remote_buffers_arg;
  ctx.element_type = operand_info.element_type;
  ctx.per_rank_shape = ctx.input_buffer.getType().getShape();
  ctx.tile_sizes = ctx.input_extract.getFullTileShape();

  // Every tile must be gathered from a single rank (see EmitPeerReplicaId).
  const int64_t per_rank_size = ctx.per_rank_shape[gather_dim];
  const int64_t tile_size = ctx.tile_sizes[gather_dim];
  if (per_rank_size % tile_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "AllGather tile size (%d) must divide the per-rank size (%d) along the "
        "gather dimension.",
        tile_size, per_rank_size));
  }

  // Find a dimension to split the tile size across `world_size` to balance
  // the in-kernel D2D subtile copy work across participating ranks.
  // We prefer splitting along `gather_dim`, but fallback to the innermost
  // cleanly divisible dimension. If none are found, split_dim remains -1
  // (unsplit).
  if (ctx.tile_sizes[gather_dim] >= world_size &&
      ctx.tile_sizes[gather_dim] % world_size == 0) {
    ctx.split_dim = gather_dim;
  } else {
    for (int64_t d = static_cast<int64_t>(ctx.tile_sizes.size()) - 1; d >= 0;
         --d) {
      if (ctx.tile_sizes[d] >= world_size &&
          ctx.tile_sizes[d] % world_size == 0) {
        ctx.split_dim = d;
        break;
      }
    }
  }
  return ctx;
}

absl::StatusOr<AllGatherRewriteContext> BuildAllGatherContext(
    mlir::stablehlo::AllGatherOp op) {
  if (op.getOperands().empty()) {
    return absl::InvalidArgumentError(
        "AllGather op must have at least one operand.");
  }
  if (op.getNumOperands() != op.getNumResults()) {
    return absl::InvalidArgumentError(
        "AllGather op must have one result per operand.");
  }
  auto entry_func = op->getParentOfType<xtile::EntryFuncOp>();
  if (!entry_func) {
    return absl::InvalidArgumentError(
        "AllGather op must be in an XTile entry function.");
  }
  // The peer rank is derived from the program id (see EmitPeerReplicaId). This
  // requires one tile per program, i.e. no tile loop around the op.
  if (op->getParentOp() != entry_func.getOperation()) {
    return absl::InvalidArgumentError(
        "AllGather op must be directly nested in the XTile entry function.");
  }
  // Same layout as AllReduce: the collective metadata arguments followed by one
  // remote buffer argument per fusion parameter.
  auto num_opaque_attr =
      entry_func->getAttrOfType<mlir::IntegerAttr>("num_opaque_args");
  if (!num_opaque_attr ||
      num_opaque_attr.getInt() <= kNumCollectiveMetadataArgs) {
    return absl::InvalidArgumentError(
        "EntryFuncOp must have the collective metadata arguments followed by "
        "the remote buffer arguments.");
  }
  const int64_t num_parameters =
      num_opaque_attr.getInt() - kNumCollectiveMetadataArgs;
  const int32_t total_args = entry_func.getNumArguments();
  const int32_t metadata_args_start =
      total_args - num_opaque_attr.getInt() - kNumTileIndexArgs;

  ABSL_ASSIGN_OR_RETURN(auto replica_groups,
                   xla::ConvertReplicaGroups(op.getReplicaGroups(), op));
  if (replica_groups->replica_groups().empty()) {
    return absl::InvalidArgumentError(
        "AllGather replica groups must not be empty.");
  }

  AllGatherRewriteContext ctx;
  ctx.op = op;
  ctx.world_size = replica_groups->num_devices_per_group();
  ctx.gather_dim = op.getAllGatherDim();
  // Layout: opaque[0]=rank, opaque[1]=invocation count (unused, the signal
  // value comes from a counter in device memory), opaque[2]=signal_buffers,
  // opaque[3 + i]=remote buffers of parameter i (same as AllReduce).
  ctx.rank_arg = entry_func.getArgument(metadata_args_start);
  ctx.signal_buffers_arg = entry_func.getArgument(metadata_args_start + 2);

  for (int64_t i = 0; i < op.getNumOperands(); ++i) {
    ABSL_ASSIGN_OR_RETURN(AllGatherOperandContext operand,
                     BuildAllGatherOperandContext(
                         op, i, entry_func, num_parameters, metadata_args_start,
                         ctx.world_size, ctx.gather_dim));
    ctx.operands.push_back(std::move(operand));
  }

  if (op.getNumOperands() == 1) {
    // Single operand: programs enumerate the output tiles in row-major order,
    // so `signal_stride` consecutive programs cover one rank's slice along
    // `gather_dim`.
    const AllGatherOperandContext& operand = ctx.operands.front();
    ctx.signal_stride = operand.per_rank_shape[ctx.gather_dim] /
                        operand.tile_sizes[ctx.gather_dim];
    for (int64_t j = ctx.gather_dim + 1;
         j < static_cast<int64_t>(operand.per_rank_shape.size()); ++j) {
      ctx.signal_stride *=
          xla::CeilOfRatio(operand.per_rank_shape[j], operand.tile_sizes[j]);
    }
    return ctx;
  }

  // Each program handles one peer's tile. `signal_stride` is the max number of
  // programs per peer across operands; the barrier and `EmitPeerReplicaId` step
  // by `signal_stride` to find which peer a program belongs to.
  int64_t signal_stride = 0;
  for (const AllGatherOperandContext& operand : ctx.operands) {
    signal_stride = std::max(signal_stride, ProgramsPerRank(operand));
  }
  ctx.signal_stride = signal_stride;
  return ctx;
}

// Returns the rank whose output slice contains this program's tile. Programs
// enumerate output tiles in row-major order (same assumption as
// `signal_stride` and the asymmetric barrier), so each run of `signal_stride`
// consecutive programs covers one rank's slice along `gather_dim` /
// `split_dim`.
template <typename ContextT>
mlir::Value EmitPeerReplicaId(mlir::ImplicitLocOpBuilder& b,
                              const ContextT& ctx, mlir::Value block_id) {
  mlir::Value stride_op = arith::ConstantOp::create(
      b, b.getI32IntegerAttr(static_cast<int32_t>(ctx.signal_stride)));
  mlir::Value world_size_op = arith::ConstantOp::create(
      b, b.getI32IntegerAttr(static_cast<int32_t>(ctx.world_size)));
  mlir::Value tile_group = arith::DivUIOp::create(b, block_id, stride_op);
  return arith::RemUIOp::create(b, tile_group, world_size_op);
}

// Returns the active slot of `rank`'s remote buffer as a memref with the type
// of the local input buffer.
template <typename ContextT>
mlir::Value EmitRemoteBufferMemref(mlir::ImplicitLocOpBuilder& b,
                                   const ContextT& ctx, mlir::Value rank,
                                   mlir::Value buffer_offset) {
  const mlir::MemRefType memref_type = ctx.input_buffer.getType();
  mlir::Value ptr =
      EmitRemoteBufferPtr(b, ctx.remote_buffers_arg, rank, buffer_offset,
                          memref_type.getElementType());
  return mtx::PtrToMemrefOp::create(b, memref_type, ptr);
}

// Copies this program's slice of `operand`'s local input into this rank's
// scratch buffer. If `split_dim != -1`, the tile is split across `world_size`
// ranks so each peer copies `1 / world_size` of it; otherwise (`split_dim ==
// -1`) the whole tile is copied.
void CopySubtileToScratch(mlir::ImplicitLocOpBuilder& b,
                          const AllGatherRewriteContext& ctx,
                          const AllGatherOperandContext& operand,
                          mlir::Value peer_replica_id,
                          mlir::Value& peer_replica_idx,
                          mlir::Value buffer_offset) {
  // Op handles are value types; MLIR accessors are not const-qualified.
  xtile::ExtractTileOp input_extract = operand.input_extract;
  llvm::SmallVector<mlir::Value> stage_offsets(
      input_extract.getOffsets().begin(), input_extract.getOffsets().end());
  llvm::SmallVector<int64_t> subtile_shape(operand.tile_sizes.begin(),
                                           operand.tile_sizes.end());
  if (operand.split_dim != -1) {
    const int64_t subtile_size =
        operand.tile_sizes[operand.split_dim] / ctx.world_size;
    mlir::Value subtile_size_val =
        arith::ConstantIndexOp::create(b, subtile_size);
    if (!peer_replica_idx) {
      peer_replica_idx =
          arith::IndexCastOp::create(b, b.getIndexType(), peer_replica_id);
    }
    mlir::Value subtile_off =
        arith::MulIOp::create(b, peer_replica_idx, subtile_size_val);
    stage_offsets[operand.split_dim] =
        arith::AddIOp::create(b, stage_offsets[operand.split_dim], subtile_off);
    subtile_shape[operand.split_dim] = subtile_size;
  }

  auto subtile_tensor_type = mlir::RankedTensorType::get(
      subtile_shape, input_extract.getType().getElementType());
  auto subtile = xtile::ExtractTileOp::create(
      b, subtile_tensor_type, operand.input_buffer, stage_offsets,
      subtile_shape, input_extract.getStrides());

  mlir::Value own_scratch_slot =
      EmitRemoteBufferMemref(b, operand, ctx.rank_arg, buffer_offset);
  xtile::InsertTileOp::create(b, subtile, own_scratch_slot, stage_offsets,
                              subtile_shape, input_extract.getStrides());
}

// Emits the tile of `operand` read from the peer's remote buffer: a clone of
// the local `input_extract` (and of the i1 cast, if any) that reads from
// `pull_scratch_slot` instead.
mlir::Value EmitPullTileFromPeer(mlir::PatternRewriter& rewriter,
                                 const AllGatherOperandContext& operand,
                                 mlir::Value pull_scratch_slot) {
  xtile::ExtractTileOp input_extract = operand.input_extract;
  mlir::IRMapping mapping;
  mapping.map(operand.input_buffer, pull_scratch_slot);
  mlir::Operation* peer_extract = rewriter.clone(*input_extract, mapping);
  if (operand.input_tile.getDefiningOp() == input_extract) {
    return peer_extract->getResult(0);
  }
  // Workaround(i1_to_i8_workaround): re-apply the i8 -> i1 cast.
  mapping.map(input_extract.getResult(), peer_extract->getResult(0));
  mlir::Operation* peer_cast =
      rewriter.clone(*operand.input_tile.getDefiningOp(), mapping);
  return peer_cast->getResult(0);
}

struct AllToAllRewriteContext {
  mlir::stablehlo::AllToAllOp op;
  xtile::ExtractTileOp input_extract;
  // Local input buffer read by `input_extract`.
  mlir::TypedValue<mlir::MemRefType> input_buffer;
  mlir::Value rank_arg;
  mlir::Value signal_buffers_arg;
  // !tt.ptr<i64> table of the per-rank remote scratch buffers.
  mlir::Value remote_buffers_arg;
  int64_t world_size = 0;
  int64_t split_dim = 0;
  int64_t shard_size = 0;
  int64_t signal_stride = 0;
  AllToAllStrategy strategy = AllToAllStrategy::kOneShot;
  PrimitiveType element_type = PrimitiveType::PRIMITIVE_TYPE_INVALID;
  llvm::ArrayRef<int64_t> input_shape;
  llvm::ArrayRef<int64_t> tile_sizes;
};

absl::StatusOr<AllToAllRewriteContext> BuildAllToAllContext(
    mlir::stablehlo::AllToAllOp op) {
  if (op.getOperands().size() != 1) {
    return absl::InvalidArgumentError(
        "AllToAll op must have exactly one operand.");
  }
  if (op.getSplitDimension() != op.getConcatDimension()) {
    return absl::InvalidArgumentError(
        "AllToAll op split_dimension must equal concat_dimension.");
  }
  auto entry_func = op->getParentOfType<xtile::EntryFuncOp>();
  if (!entry_func) {
    return absl::InvalidArgumentError(
        "AllToAll op must be in an XTile entry function.");
  }
  if (op->getParentOp() != entry_func.getOperation()) {
    return absl::InvalidArgumentError(
        "AllToAll op must be directly nested in the XTile entry function.");
  }
  auto num_opaque_attr =
      entry_func->getAttrOfType<mlir::IntegerAttr>("num_opaque_args");
  if (!num_opaque_attr ||
      num_opaque_attr.getInt() <= kNumCollectiveMetadataArgs) {
    return absl::InvalidArgumentError(
        "EntryFuncOp must have the collective metadata arguments followed by "
        "the remote buffer arguments.");
  }

  // stablehlo.all_to_all requires the split dimension of the operand to be
  // divisible by split_count. When this is not the case, the XTile fusion
  // emitter wraps the operand and result in unrealized_conversion_cast with a
  // padded type.
  // TODO(patrios): Use xtile.all_to_all to avoid this cast workaround.
  mlir::Value input_tile = op.getOperand(0);
  if (auto cast_in =
          input_tile.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (!input_tile.hasOneUse() || cast_in.getInputs().size() != 1) {
      return absl::InvalidArgumentError(
          "Expected single-input UnrealizedConversionCastOp wrapper around "
          "AllToAll operand.");
    }
    input_tile = cast_in.getInputs()[0];
  }

  const int64_t num_parameters =
      num_opaque_attr.getInt() - kNumCollectiveMetadataArgs;
  int32_t total_args = entry_func.getNumArguments();
  int32_t metadata_args_start =
      total_args - num_opaque_attr.getInt() - kNumTileIndexArgs;
  ABSL_ASSIGN_OR_RETURN(
      CollectiveOperandExtractInfo extract_info,
      ExtractCollectiveOperand(input_tile, /*operand_index=*/0, "AllToAll",
                               entry_func, num_parameters,
                               metadata_args_start + kNumCollectiveMetadataArgs,
                               /*require_single_use=*/true));

  ABSL_ASSIGN_OR_RETURN(auto replica_groups,
                   xla::ConvertReplicaGroups(op.getReplicaGroups(), op));
  if (replica_groups->replica_groups().empty()) {
    return absl::InvalidArgumentError(
        "AllToAll replica groups must not be empty.");
  }

  AllToAllRewriteContext ctx;
  ctx.op = op;
  ctx.input_extract = extract_info.input_extract;
  ctx.input_buffer = extract_info.input_buffer;
  ctx.world_size = replica_groups->num_devices_per_group();
  ctx.split_dim = op.getSplitDimension();
  ctx.element_type = extract_info.element_type;
  ctx.input_shape = ctx.input_buffer.getType().getShape();
  ctx.tile_sizes = extract_info.input_extract.getFullTileShape();
  if (ctx.world_size <= 0 ||
      ctx.input_shape[ctx.split_dim] % ctx.world_size != 0) {
    return absl::InvalidArgumentError(
        "AllToAll split dimension must be divisible by world size.");
  }
  std::optional<xtile::CollectiveStrategy> strategy =
      xtile::GetCollectiveStrategy(op);
  if (!strategy.has_value()) {
    return absl::InvalidArgumentError(
        "AllToAll op must have a collective_strategy attribute.");
  }
  switch (*strategy) {
    case xtile::CollectiveStrategy::kOneShot:
      ctx.strategy = AllToAllStrategy::kOneShot;
      break;
    case xtile::CollectiveStrategy::kTwoShot:
      ctx.strategy = AllToAllStrategy::kTwoShot;
      break;
  }
  ctx.shard_size = ctx.input_shape[ctx.split_dim] / ctx.world_size;
  const int64_t tile_size = ctx.tile_sizes[ctx.split_dim];
  if (ctx.shard_size % tile_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "AllToAll tile size (%d) must divide the per-rank shard size (%d) "
        "along the split dimension.",
        tile_size, ctx.shard_size));
  }

  ctx.rank_arg = entry_func.getArgument(metadata_args_start);
  ctx.signal_buffers_arg = entry_func.getArgument(metadata_args_start + 2);
  ctx.remote_buffers_arg = extract_info.remote_buffers_arg;

  ctx.signal_stride = ctx.shard_size / tile_size;
  for (int64_t j = ctx.split_dim + 1;
       j < static_cast<int64_t>(ctx.input_shape.size()); ++j) {
    ctx.signal_stride *=
        xla::CeilOfRatio(ctx.input_shape[j], ctx.tile_sizes[j]);
  }
  return ctx;
}

// Emits:
//   %shard_size = arith.constant <ctx.shard_size> : index
//   %replica_idx = arith.index_cast %replica_id : i32 to index
//   %shard_off = arith.muli %replica_idx, %shard_size : index
//   %split_offset = arith.addi %intra_shard_offset, %shard_off : index
mlir::Value EmitShardSplitOffset(mlir::ImplicitLocOpBuilder& b,
                                 AllToAllRewriteContext& ctx,
                                 mlir::Value replica_id) {
  mlir::Value shard_size_val =
      arith::ConstantIndexOp::create(b, ctx.shard_size);
  mlir::Value replica_idx =
      arith::IndexCastOp::create(b, b.getIndexType(), replica_id);
  mlir::Value shard_off = arith::MulIOp::create(b, replica_idx, shard_size_val);
  return arith::AddIOp::create(b, ctx.input_extract.getOffsets()[ctx.split_dim],
                               shard_off);
}

// Pushes this program's tile from the target peer's shard of the local input
// buffer into this rank's shard of the peer's remote scratch buffer.
// Needed for push-based algorithms like AllToAll.
void CopyTileToScratch(mlir::ImplicitLocOpBuilder& b,
                       AllToAllRewriteContext& ctx, mlir::Value peer_replica_id,
                       mlir::Value peer_split_offset,
                       mlir::Value buffer_offset) {
  llvm::SmallVector<mlir::Value> read_offsets(
      ctx.input_extract.getOffsets().begin(),
      ctx.input_extract.getOffsets().end());
  read_offsets[ctx.split_dim] = peer_split_offset;

  auto tile_tensor_type = mlir::RankedTensorType::get(
      ctx.tile_sizes, ctx.input_extract.getType().getElementType());
  auto tile = xtile::ExtractTileOp::create(
      b, tile_tensor_type, ctx.input_buffer, read_offsets, ctx.tile_sizes,
      ctx.input_extract.getStrides());

  llvm::SmallVector<mlir::Value> write_offsets(
      ctx.input_extract.getOffsets().begin(),
      ctx.input_extract.getOffsets().end());
  write_offsets[ctx.split_dim] = EmitShardSplitOffset(b, ctx, ctx.rank_arg);

  mlir::Value peer_scratch_slot =
      EmitRemoteBufferMemref(b, ctx, peer_replica_id, buffer_offset);
  xtile::InsertTileOp::create(b, tile, peer_scratch_slot, write_offsets,
                              ctx.tile_sizes, ctx.input_extract.getStrides());
}

// Flat shapes of one reduce-scatter operand and its output (see
// FlattenReduceScatterFusion).
struct FlatReduceScatterShapes {
  Shape input;
  Shape output;
};

// Returns the flattened input `[R * R, ...]` and output `[R, ...]` shapes for
// a reduce-scatter operand with `num_output_elements` output elements on
// `num_devices` ranks.
absl::StatusOr<FlatReduceScatterShapes> GetFlatReduceScatterShapes(
    PrimitiveType element_type, int64_t num_devices,
    int64_t num_output_elements) {
  TF_RET_CHECK(num_devices > 0);
  TF_RET_CHECK(num_output_elements % num_devices == 0)
      << "Reduce-scatter output element count " << num_output_elements
      << " is not divisible by the number of devices " << num_devices;
  const int64_t inner_elements = num_output_elements / num_devices;
  // If `inner_elements` isn't a power of 2 (e.g., 12288), a 2D `[R, 12288]`
  // shape rounds the inner tile up to 16384 and can underfill the grid.
  // Factor out `B = gcd(inner_elements, max_blocks_per_rank)` into a middle
  // dimension `[R, B, inner_elements / B]` so `GreedyPowerOfTwoTiles` can split
  // along `B` first.
  const int64_t max_blocks_per_rank =
      std::max<int64_t>(1, kReduceScatterMaxBlocksPerGrid / num_devices);
  const int64_t inner_blocks =
      llvm::has_single_bit(static_cast<uint64_t>(inner_elements))
          ? 1
          : std::gcd(inner_elements, max_blocks_per_rank);
  if (inner_blocks > 1) {
    return FlatReduceScatterShapes{
        ShapeUtil::MakeShapeWithDenseLayout(
            element_type,
            {num_devices * num_devices, inner_blocks,
             inner_elements / inner_blocks},
            {2, 1, 0}),
        ShapeUtil::MakeShapeWithDenseLayout(
            element_type,
            {num_devices, inner_blocks, inner_elements / inner_blocks},
            {2, 1, 0})};
  }
  return FlatReduceScatterShapes{
      ShapeUtil::MakeShapeWithDenseLayout(
          element_type, {num_devices * num_devices, inner_elements}, {1, 0}),
      ShapeUtil::MakeShapeWithDenseLayout(
          element_type, {num_devices, inner_elements}, {1, 0})};
}

// Validates that `fused_computation` is a single-operand or canonicalized
// variadic reduce-scatter fusion whose operands are direct fusion parameters
// and satisfies `IsReduceScatterFlattenable`. Returns the mutable
// `HloReduceScatterInstruction*`.
absl::StatusOr<HloReduceScatterInstruction*>
ValidateReduceScatterFusionForFlattening(HloComputation* fused_computation) {
  HloInstruction* old_root = fused_computation->root_instruction();
  auto* rs = DynCast<HloReduceScatterInstruction>(
      GetMutableCollectiveHero(fused_computation));
  TF_RET_CHECK(rs != nullptr)
      << "Expected a reduce-scatter in fusion computation.";
  const int64_t num_operands = rs->operand_count();
  TF_RET_CHECK(num_operands >= 1);
  if (num_operands > 1) {
    TF_RET_CHECK(rs->shape().IsTuple() &&
                 rs->shape().tuple_shapes().size() == num_operands)
        << "Expected one output per reduce-scatter operand.";
    TF_RET_CHECK(old_root->opcode() == HloOpcode::kTuple &&
                 old_root->operand_count() == num_operands)
        << "Expected a tuple of get-tuple-elements root for variadic "
           "reduce-scatter.";
    for (int64_t k = 0; k < num_operands; ++k) {
      const HloInstruction* gte = old_root->operand(k);
      TF_RET_CHECK(gte->opcode() == HloOpcode::kGetTupleElement &&
                   gte->operand(0) == rs && gte->tuple_index() == k)
          << "Expected root tuple operand " << k
          << " to be get-tuple-element(rs, " << k << ").";
    }
  } else {
    TF_RET_CHECK(old_root == rs && rs->shape().IsArray())
        << "Expected a reduce-scatter fusion root.";
  }
  TF_RET_CHECK(fused_computation->num_parameters() == num_operands)
      << "Expected one parameter per reduce-scatter operand.";
  for (int64_t k = 0; k < num_operands; ++k) {
    TF_RET_CHECK(rs->operand(k) == fused_computation->parameter_instruction(k))
        << "Expected operand " << k << " to be parameter " << k;
  }
  TF_RET_CHECK(IsReduceScatterFlattenable(rs))
      << "Reduce-scatter is not flattenable.";
  return rs;
}

// Bitcasts `fusion_instr`'s inputs to `param_shapes` and replaces `rs` inside
// the fusion with a reduce-scatter on `param_shapes` and `new_fusion_shape`
// with `scatter_dimension=0`.
absl::Status ApplyFlatShapesToRs(HloFusionInstruction* fusion_instr,
                                 HloReduceScatterInstruction* rs,
                                 absl::Span<const Shape> param_shapes,
                                 const Shape& new_fusion_shape) {
  HloComputation* parent = fusion_instr->parent();
  HloComputation* fused_computation =
      fusion_instr->fused_instructions_computation();
  HloInstruction* old_root = fused_computation->root_instruction();
  const int64_t num_operands = rs->operand_count();

  std::vector<HloInstruction*> params;
  params.reserve(num_operands);
  for (int64_t k = 0; k < num_operands; ++k) {
    HloInstruction* param = fused_computation->parameter_instruction(k);
    *param->mutable_shape() = param_shapes[k];
    params.push_back(param);
  }
  HloInstruction* new_rs =
      fused_computation->AddInstruction(HloInstruction::CreateReduceScatter(
          new_fusion_shape, params, rs->to_apply(), rs->device_list(),
          rs->constrain_layout(), rs->channel_id(), rs->use_global_device_ids(),
          /*scatter_dimension=*/0));
  new_rs->CopyBackendConfigFrom(rs);
  new_rs->set_metadata(rs->metadata());
  new_rs->set_frontend_attributes(rs->frontend_attributes());

  HloInstruction* new_root = new_rs;
  std::vector<HloInstruction*> old_gtes;
  if (num_operands > 1) {
    old_gtes.assign(old_root->operands().begin(), old_root->operands().end());
    std::vector<HloInstruction*> gtes;
    gtes.reserve(num_operands);
    for (int64_t k = 0; k < num_operands; ++k) {
      gtes.push_back(fused_computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_rs, k)));
    }
    new_root =
        fused_computation->AddInstruction(HloInstruction::CreateTuple(gtes));
    new_root->add_frontend_attribute(kDisableTupleSimplifierAttribute, "true");
  }
  fused_computation->set_root_instruction(new_root,
                                          /*accept_different_shape=*/true);
  const std::string rs_name(rs->name());
  if (num_operands > 1) {
    ABSL_RETURN_IF_ERROR(fused_computation->RemoveInstruction(old_root));
    for (HloInstruction* old_gte : old_gtes) {
      ABSL_RETURN_IF_ERROR(fused_computation->RemoveInstruction(old_gte));
    }
  }
  ABSL_RETURN_IF_ERROR(fused_computation->RemoveInstruction(rs));
  // Keep the original name so that the flattened reduce-scatter can still be
  // matched to the original HLO (e.g. in profiles).
  new_rs->SetAndSanitizeName(rs_name);

  for (int64_t k = 0; k < num_operands; ++k) {
    HloInstruction* bitcast =
        parent->AddInstruction(HloInstruction::CreateBitcast(
            param_shapes[k], fusion_instr->mutable_operand(k)));
    ABSL_RETURN_IF_ERROR(fusion_instr->ReplaceOperandWithDifferentShape(k, bitcast));
  }
  *fusion_instr->mutable_shape() = new_fusion_shape;
  return absl::OkStatus();
}

// Bitcasts the outputs of `fusion_instr` back to `original_output_shapes` in
// its parent computation.
absl::Status RestoreOriginalShapeForRs(
    HloFusionInstruction* fusion_instr,
    absl::Span<const Shape> original_output_shapes) {
  HloComputation* parent = fusion_instr->parent();
  const int64_t num_operands = original_output_shapes.size();
  if (num_operands == 1) {
    HloInstruction* bitcast = parent->AddInstruction(
        HloInstruction::CreateBitcast(original_output_shapes[0], fusion_instr));
    return fusion_instr->ReplaceAllUsesWithDifferentShape(bitcast);
  }

  const std::vector<HloInstruction*> old_users(fusion_instr->users().begin(),
                                               fusion_instr->users().end());
  std::vector<HloInstruction*> original_outputs;
  original_outputs.reserve(num_operands);
  for (int64_t k = 0; k < num_operands; ++k) {
    HloInstruction* gte = parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(fusion_instr, k));
    original_outputs.push_back(parent->AddInstruction(
        HloInstruction::CreateBitcast(original_output_shapes[k], gte)));
  }
  HloInstruction* original_tuple =
      parent->AddInstruction(HloInstruction::CreateTuple(original_outputs));
  return fusion_instr->ReplaceAllUsesWithDifferentShape(old_users,
                                                        original_tuple);
}

template <typename ComputationT>
auto FindCollectiveHero(ComputationT* computation)
    -> decltype(computation->root_instruction()) {
  if (computation == nullptr) {
    return nullptr;
  }
  auto is_collective = [](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kAllReduce ||
           instr->opcode() == HloOpcode::kAllGather ||
           instr->opcode() == HloOpcode::kAllToAll ||
           instr->opcode() == HloOpcode::kReduceScatter;
  };
  if (auto* root = computation->root_instruction(); is_collective(root)) {
    return root;
  }
  for (auto* instr : computation->instructions()) {
    if (is_collective(instr)) {
      return instr;
    }
  }
  return nullptr;
}

}  // namespace

llvm::SmallVector<int64_t> GreedyPowerOfTwoTiles(const Shape& output_shape,
                                                 int32_t num_blocks) {
  CHECK_GT(num_blocks, 0) << "num_blocks must be positive. Was " << num_blocks;
  // Rank fits in int32_t.
  const auto rank = static_cast<int32_t>(output_shape.dimensions().size());
  const llvm::ArrayRef<const int64_t> minor_to_major =
      LayoutUtil::MinorToMajor(output_shape);
  llvm::SmallVector<int64_t, 4> tile_sizes(rank);
  // NB: Unsigned because llvm::bit_<> functions expect unsigned.
  uint64_t remaining_blocks = num_blocks;
  // Iterate from most major to most minor to keep memory contiguous for each
  // block.
  for (int32_t i = rank - 1; i >= 0; --i) {
    const auto dim = static_cast<int32_t>(minor_to_major[i]);
    const uint64_t dim_size = output_shape.dimensions(dim);
    // Largest power-of-two k <= std::min(dim_size, remaining_blocks).
    const uint64_t k = std::max(
        uint64_t{1}, llvm::bit_floor(std::min(remaining_blocks, dim_size)));
    // Round up number of tiles in this dimension to power of two.
    tile_sizes[dim] =
        static_cast<int64_t>(llvm::bit_ceil(xla::CeilOfRatio(dim_size, k)));
    const uint64_t blocks_used =
        xla::CeilOfRatio(dim_size, static_cast<uint64_t>(tile_sizes[dim]));
    remaining_blocks = xla::FloorOfRatio(remaining_blocks, blocks_used);
  }
  return tile_sizes;
}

const HloInstruction* GetCollectiveHero(const HloComputation* computation) {
  return FindCollectiveHero(computation);
}

HloInstruction* GetMutableCollectiveHero(HloComputation* computation) {
  return FindCollectiveHero(computation);
}

absl::StatusOr<BlockLevelFusionConfig> GetCollectiveBlockLevelFusionConfig(
    const GpuTopology& gpu_topology, const HloFusionInstruction* fusion_instr,
    const DeviceAssignment* device_assignment) {
  const HloInstruction* instr =
      GetCollectiveHero(fusion_instr->fused_instructions_computation());
  if (instr == nullptr) {
    return absl::UnimplementedError(absl::StrFormat(
        "No collective hero found for Triton codegen: %s",
        fusion_instr->fused_instructions_computation()->ToString()));
  }
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr->backend_config<GpuBackendConfig>());
  if (!IsTritonCollectiveKernel(
          gpu_config.collective_backend_config().kernel_strategy())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Instruction was not annotated with Triton strategy but still called "
        "for Triton codegen. %s",
        instr->ToString()));
  }
  if (!gpu_topology.has_gpu_target_config()) {
    return absl::InvalidArgumentError(
        "GpuTopology must have a target config to build "
        "BlockLevelFusionConfig.");
  }
  if (instr->shape().IsTuple()) {
    const HloInstruction* root = fusion_instr->fused_expression_root();
    TF_RET_CHECK(root->opcode() == HloOpcode::kTuple &&
                 root->operand_count() == instr->shape().tuple_shapes().size())
        << "Expected a tuple fusion root for variadic collective: "
        << fusion_instr->fused_instructions_computation()->ToString();
  }

  if (const HloAllReduceInstruction* all_reduce =
          DynCast<HloAllReduceInstruction>(instr)) {
    return GetBlockLevelConfigForAllReduce(gpu_topology, all_reduce,
                                           device_assignment);
  }

  if (const HloAllGatherInstruction* all_gather =
          DynCast<HloAllGatherInstruction>(instr)) {
    return GetBlockLevelConfigForAllGather(gpu_topology, all_gather,
                                           device_assignment);
  }

  if (const HloAllToAllInstruction* all_to_all =
          DynCast<HloAllToAllInstruction>(instr)) {
    return GetBlockLevelConfigForAllToAll(gpu_topology, all_to_all,
                                          device_assignment);
  }

  if (const HloReduceScatterInstruction* reduce_scatter =
          DynCast<HloReduceScatterInstruction>(instr)) {
    return GetBlockLevelConfigForReduceScatter(gpu_topology, reduce_scatter,
                                               device_assignment);
  }

  return absl::UnimplementedError(
      absl::StrFormat("Unsupported collective opcode for Triton codegen: %s",
                      HloOpcodeString(instr->opcode())));
}

absl::Status TrySetGpuBackendConfigForCollective(
    const GpuTopology& gpu_topology, HloFusionInstruction* fusion_instr,
    const DeviceAssignment* device_assignment) {
  ABSL_ASSIGN_OR_RETURN(BlockLevelFusionConfig block_config,
                   GetCollectiveBlockLevelFusionConfig(
                       gpu_topology, fusion_instr, device_assignment));
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                   fusion_instr->backend_config<GpuBackendConfig>());
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      kTritonCollectiveFusionKind);
  *gpu_backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() = std::move(block_config);
  ABSL_RETURN_IF_ERROR(
      fusion_instr->set_backend_config(std::move(gpu_backend_config)));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Shape>> GetCollectiveUnmanagedKernelArguments(
    const HloFusionInstruction* fusion) {
  const HloComputation* computation = fusion->fused_instructions_computation();
  const HloInstruction* collective = GetCollectiveHero(computation);
  if (collective == nullptr) {
    return std::vector<Shape>();
  }
  return GetRemoteBufferUnmanagedKernelArguments(
      computation, Cast<HloCollectiveInstruction>(collective));
}

absl::StatusOr<int32_t> AddCollectiveMetadataArguments(
    llvm::SmallVector<mlir::Type>& fn_arg_types, mlir::ImplicitLocOpBuilder& b,
    const HloComputation* hlo_computation) {
  const HloInstruction* collective = GetCollectiveHero(hlo_computation);
  const bool has_invocation_count =
      collective == nullptr ||
      collective->opcode() != HloOpcode::kReduceScatter;
  // rank: i32
  fn_arg_types.push_back(b.getI32Type());
  if (has_invocation_count) {
    // signal_value: i32
    fn_arg_types.push_back(b.getI32Type());
  }
  // signal_buffers: !tt.ptr<i64>
  fn_arg_types.push_back(
      ttir::PointerType::get(b.getI64Type(), kGlobalAddressSpace));

  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    (void)p;
    // Also add the remote/scratch buffers for collectives.
    // !tt.ptr<i64>
    fn_arg_types.push_back(
        ttir::PointerType::get(b.getI64Type(), kGlobalAddressSpace));
  }
  const int32_t num_metadata_args = has_invocation_count
                                        ? kNumCollectiveMetadataArgs
                                        : kNumCollectiveMetadataArgs - 1;
  return hlo_computation->num_parameters() + num_metadata_args;
}

mlir::LogicalResult RewriteAllReduce(mlir::stablehlo::AllReduceOp op,
                                     mlir::PatternRewriter& rewriter) {
  const mlir::Location loc = op->getLoc();
  absl::StatusOr<AllReduceEmitterContext> maybe_context =
      CreateAllReduceEmitterContext(op);
  if (!maybe_context.ok()) {
    VLOG(3) << "Failed to create AllReduceEmitterContext: "
            << maybe_context.status().message();
    return rewriter.notifyMatchFailure(
        loc, absl::StrCat("Failed to create AllReduceEmitterContext: ",
                          maybe_context.status().message()));
  }
  VLOG(3) << "AllReduceEmitter::Emit using strategy: "
          << maybe_context->strategy;
  return AllReduceEmitter::Emit(*std::move(maybe_context), rewriter);
}

mlir::LogicalResult RewriteReduceScatter(xtile::ReduceScatterOp op,
                                         mlir::PatternRewriter& rewriter) {
  const mlir::Location loc = op->getLoc();
  absl::StatusOr<ReduceScatterEmitterContext> maybe_context =
      CreateReduceScatterEmitterContext(op);
  if (!maybe_context.ok()) {
    VLOG(3) << "Failed to create ReduceScatterEmitterContext: "
            << maybe_context.status().message();
    return rewriter.notifyMatchFailure(
        loc, absl::StrCat("Failed to create ReduceScatterEmitterContext: ",
                          maybe_context.status().message()));
  }
  return ReduceScatterEmitter::Emit(*std::move(maybe_context), rewriter);
}

absl::StatusOr<SymmetricMemoryType> GetSymmetricMemoryType(
    const GpuTopology& gpu_topology, const HloInstruction& instr,
    const DeviceAssignment* device_assignment) {
  const HloInstruction* collective = &instr;
  if (instr.opcode() == HloOpcode::kFusion) {
    collective = GetCollectiveHero(instr.fused_instructions_computation());
    TF_RET_CHECK(collective != nullptr)
        << "No collective hero found in fusion: " << instr.ToString();
  }
  ABSL_ASSIGN_OR_RETURN(
      const bool is_collective_single_host,
      IsCollectiveSingleHost(gpu_topology, *collective, device_assignment));
  ABSL_ASSIGN_OR_RETURN(const bool is_cross_host_possible,
                   IsCrossHostCollectiveKernelPossible(
                       gpu_topology, *collective, device_assignment));
  const bool is_lsa_possible = IsLsaPossible(gpu_topology);

  if (!is_collective_single_host) {
    if (is_cross_host_possible && is_lsa_possible) {
      return SymmetricMemoryType::kLoadStoreAccessible;
    }
    return absl::UnimplementedError(
        "Cross-host symmetric memory collectives are not supported.");
  }
  return is_lsa_possible ? SymmetricMemoryType::kLoadStoreAccessible
                         : SymmetricMemoryType::kXlaRendezvous;
}

absl::StatusOr<CollectiveKernelSpec> CreateCollectiveKernelSpec(
    const GpuTopology& gpu_topology, const HloInstruction* instr,
    const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type,
    const DeviceAssignment* device_assignment) {
  const HloInstruction* collective = instr;
  if (instr->opcode() == HloOpcode::kFusion) {
    collective = GetCollectiveHero(instr->fused_instructions_computation());
    TF_RET_CHECK(collective != nullptr)
        << "No collective hero found in fusion: " << instr->ToString();
  }
  switch (collective->opcode()) {
    case HloOpcode::kAllReduce:
      return CreateAllReduceKernelSpec(
          gpu_topology, Cast<HloAllReduceInstruction>(collective),
          launch_dimensions, scratch_memory_type, device_assignment);
    case HloOpcode::kAllGather:
      return CreateAllGatherKernelSpec(collective, launch_dimensions,
                                       scratch_memory_type);
    case HloOpcode::kAllToAll:
      return CreateAllToAllKernelSpec(collective, launch_dimensions,
                                      scratch_memory_type);
    case HloOpcode::kReduceScatter:
      return CreateReduceScatterKernelSpec(collective, launch_dimensions,
                                           scratch_memory_type);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("CollectiveKernelSpec creation not implemented for "
                          "opcode: %s",
                          HloOpcodeString(collective->opcode())));
  }
}

mlir::LogicalResult RewriteAllGather(mlir::stablehlo::AllGatherOp op,
                                     mlir::PatternRewriter& rewriter) {
  absl::StatusOr<AllGatherRewriteContext> maybe_ctx = BuildAllGatherContext(op);
  if (!maybe_ctx.ok()) {
    return rewriter.notifyMatchFailure(op, maybe_ctx.status().message());
  }
  AllGatherRewriteContext& ctx = *maybe_ctx;

  // All local tiles (and their offsets) are defined before `op`, so the
  // rewrite is emitted right before it.
  mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  builder.setInsertionPoint(op);

  mlir::Value block_id = ttir::GetProgramIdOp::create(builder, 0);
  mlir::Value peer_replica_id = EmitPeerReplicaId(builder, ctx, block_id);
  mlir::Value signal_value = EmitDeviceInvocationCount(
      builder, ctx.signal_buffers_arg, ctx.rank_arg, block_id, ctx.world_size,
      /*signal_slot=*/peer_replica_id, ctx.signal_stride);

  // Copy this program's subtile of every local tile to this rank's remote
  // buffers.
  mlir::Value buffer_index = EmitDoubleBufferIndex(builder, signal_value);
  mlir::Value peer_replica_idx;
  llvm::SmallVector<mlir::Value, 2> buffer_offsets;
  buffer_offsets.reserve(ctx.operands.size());
  for (const AllGatherOperandContext& operand : ctx.operands) {
    buffer_offsets.push_back(EmitDoubleBufferOffsetFromIndex(
        builder, buffer_index, Product(operand.per_rank_shape),
        operand.element_type));
    CopySubtileToScratch(builder, ctx, operand, peer_replica_id,
                         peer_replica_idx, buffer_offsets.back());
  }
  // Wait until the peer has staged the whole tile of every operand. A single
  // barrier covers all operands because every program stages all of them.
  EmitBlockBarrier(builder, ctx.signal_buffers_arg, ctx.rank_arg, signal_value,
                   ctx.world_size, /*signal_slot=*/peer_replica_id,
                   ctx.signal_stride, mtx::BarrierMode::kConsumerSymmetric);
  // Pull the tiles from the peer's remote buffers.
  llvm::SmallVector<mlir::Value, 2> peer_tiles;
  peer_tiles.reserve(ctx.operands.size());
  for (const auto& [operand, buffer_offset] :
       llvm::zip(ctx.operands, buffer_offsets)) {
    mlir::Value pull_scratch_slot = EmitRemoteBufferMemref(
        builder, operand, peer_replica_id, buffer_offset);
    peer_tiles.push_back(
        EmitPullTileFromPeer(rewriter, operand, pull_scratch_slot));
  }
  rewriter.replaceOp(op, peer_tiles);
  // The local tiles were only used by `op`.
  for (const AllGatherOperandContext& operand : ctx.operands) {
    if (operand.input_tile.getDefiningOp() != operand.input_extract) {
      rewriter.eraseOp(operand.input_tile.getDefiningOp());
    }
    rewriter.eraseOp(operand.input_extract);
  }

  return mlir::success();
}

mlir::LogicalResult RewriteAllToAll(mlir::stablehlo::AllToAllOp op,
                                    mlir::PatternRewriter& rewriter) {
  absl::StatusOr<AllToAllRewriteContext> maybe_ctx = BuildAllToAllContext(op);
  if (!maybe_ctx.ok()) {
    return rewriter.notifyMatchFailure(op, maybe_ctx.status().message());
  }
  AllToAllRewriteContext& ctx = *maybe_ctx;

  mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  builder.setInsertionPoint(ctx.input_extract);

  mlir::Value block_id = ttir::GetProgramIdOp::create(builder, 0);
  mlir::Value peer_replica_id = EmitPeerReplicaId(builder, ctx, block_id);
  mlir::Value peer_split_offset =
      EmitShardSplitOffset(builder, ctx, peer_replica_id);
  mlir::Value signal_value = EmitDeviceInvocationCount(
      builder, ctx.signal_buffers_arg, ctx.rank_arg, block_id, ctx.world_size);

  mlir::Value buffer_offset;
  if (ctx.strategy == AllToAllStrategy::kOneShot) {
    buffer_offset = EmitDoubleBufferOffset(
        builder, signal_value, Product(ctx.input_shape), ctx.element_type);
    // Push this program's tile of the local input tensor directly into
    // `peer_replica_id`'s remote scratch buffer at shard `rank_arg`.
    CopyTileToScratch(builder, ctx, peer_replica_id, peer_split_offset,
                      buffer_offset);
    // Wait until `peer_replica_id` has pushed its corresponding tile into this
    // rank's scratch buffer.
    EmitBlockBarrier(builder, ctx.signal_buffers_arg, ctx.rank_arg,
                     signal_value, ctx.world_size,
                     /*signal_slot=*/peer_replica_id, ctx.signal_stride,
                     mtx::BarrierMode::kPairwiseSymmetric);
  } else {
    // TODO(patrios): Instead of using scratch buffer, mark AllToAll output
    // buffer as a collective operation during compilation and write directly to
    // it. Without this right now the only benefit of two shot is lower memory
    // usage.
    buffer_offset = arith::ConstantOp::create(builder, builder.getI64Type(),
                                              builder.getI64IntegerAttr(0));
    // Synchronize with `peer_replica_id` before writing into its
    // single-buffered remote scratch buffer.
    EmitBlockBarrier(builder, ctx.signal_buffers_arg, ctx.rank_arg,
                     signal_value, ctx.world_size,
                     /*signal_slot=*/peer_replica_id, ctx.signal_stride,
                     mtx::BarrierMode::kPairwiseSymmetric,
                     /*emit_local_barrier=*/false);
    CopyTileToScratch(builder, ctx, peer_replica_id, peer_split_offset,
                      buffer_offset);
    mlir::Value exit_signal_value = arith::AddIOp::create(
        builder, signal_value,
        arith::ConstantOp::create(builder, builder.getI32IntegerAttr(1)));
    EmitBlockBarrier(builder, ctx.signal_buffers_arg, ctx.rank_arg,
                     exit_signal_value, ctx.world_size,
                     /*signal_slot=*/peer_replica_id, ctx.signal_stride,
                     mtx::BarrierMode::kPairwiseSymmetric);
  }

  // Read the tile pushed by `peer_replica_id` from this rank's own local
  // scratch buffer: offset along `split_dim` is `intra_shard_offset +
  // peer_replica_id * shard_size`, matching `output_insert`.
  mlir::Value own_scratch_slot =
      EmitRemoteBufferMemref(builder, ctx, ctx.rank_arg, buffer_offset);
  rewriter.modifyOpInPlace(ctx.input_extract, [&]() {
    ctx.input_extract.getSourceMutable().assign(own_scratch_slot);
    ctx.input_extract.getOffsetsMutable()[ctx.split_dim].assign(
        peer_split_offset);
  });
  rewriter.replaceOp(op, op.getOperand(0));
  return mlir::success();
}

bool ShouldFlattenReduceScatter(
    const HloReduceScatterInstruction* absl_nonnull rs) {
  const int64_t num_devices = rs->device_list()->num_devices_per_group();
  if (num_devices <= 0) {
    return false;
  }
  const int64_t scatter_dim = rs->scatter_dimension();
  for (const Shape& output_shape : GetOutputShapes(rs)) {
    if (scatter_dim < 0 ||
        scatter_dim >= static_cast<int64_t>(output_shape.dimensions().size()) ||
        output_shape.dimensions(scatter_dim) % num_devices != 0) {
      return true;
    }
    const int64_t per_peer_elements =
        ShapeUtil::ElementsIn(output_shape) / num_devices;
    if (per_peer_elements <= 0 ||
        !llvm::has_single_bit(static_cast<uint64_t>(per_peer_elements))) {
      return true;
    }
  }
  return false;
}

absl::Status FlattenReduceScatterFusion(
    HloFusionInstruction* absl_nonnull fusion_instr) {
  HloComputation* fused_computation =
      fusion_instr->fused_instructions_computation();
  ABSL_ASSIGN_OR_RETURN(HloReduceScatterInstruction * rs,
                   ValidateReduceScatterFusionForFlattening(fused_computation));
  const int64_t num_operands = rs->operand_count();
  const int64_t num_devices = rs->device_list()->num_devices_per_group();

  std::vector<Shape> original_output_shapes;
  std::vector<Shape> param_shapes;
  std::vector<Shape> output_shapes;
  original_output_shapes.reserve(num_operands);
  param_shapes.reserve(num_operands);
  output_shapes.reserve(num_operands);
  for (const Shape& output_shape : GetOutputShapes(rs)) {
    ABSL_ASSIGN_OR_RETURN(
        FlatReduceScatterShapes flat_shapes,
        GetFlatReduceScatterShapes(output_shape.element_type(), num_devices,
                                   ShapeUtil::ElementsIn(output_shape)));
    original_output_shapes.push_back(output_shape);
    param_shapes.push_back(std::move(flat_shapes.input));
    output_shapes.push_back(std::move(flat_shapes.output));
  }

  const Shape new_fusion_shape = num_operands == 1
                                     ? output_shapes.front()
                                     : ShapeUtil::MakeTupleShape(output_shapes);
  ABSL_RETURN_IF_ERROR(
      ApplyFlatShapesToRs(fusion_instr, rs, param_shapes, new_fusion_shape));
  VLOG(3) << "Flattened reduce-scatter fusion " << fusion_instr->name()
          << " to " << new_fusion_shape.ToString();
  return RestoreOriginalShapeForRs(fusion_instr, original_output_shapes);
}

}  // namespace xla::gpu
