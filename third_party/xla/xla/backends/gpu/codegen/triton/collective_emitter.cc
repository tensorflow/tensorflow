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

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/codegen/emitters/ir/xla_ops.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace xla::gpu {
namespace {

using ::xla::se::gpu::AllReduceStrategy;
using xtile::TensorValue;

namespace ttir = ::mlir::triton;
namespace mtx = ::mlir::triton::xla;
namespace arith = ::mlir::arith;

using ReductionComputationEmitter = absl::AnyInvocable<xtile::TensorValue(
    mlir::ImplicitLocOpBuilder&, xtile::TensorValue, xtile::TensorValue)>;

// The main memory space on a device (HBM).
static constexpr auto kGlobalAddressSpace =
    static_cast<std::underlying_type_t<mlir::NVVM::NVVMMemorySpace>>(
        mlir::NVVM::NVVMMemorySpace::Global);

// Metadata arguments for the collective emitter.
// device_rank, signal_value, signal_buffers.
static constexpr int32_t kNumCollectiveMetadataArgs = 3;
static constexpr int32_t kNumTileIndexArgs = 1;

struct AllReduceInfo {
  ReductionKind reduction_kind;
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
  AllReduceStrategy all_reduce_strategy;
};

// Common context for all reduce emitters.
struct AllReduceEmitterContext {
  mlir::stablehlo::AllReduceOp op;
  int32_t num_input_output_args;
  int32_t num_scratch_buffers;
  // The entry function of the all reduce op.
  xtile::EntryFuncOp xtile_entry_fn;
  // The input tile to all reduce.
  xtile::TensorValue input_tile;
  // The extract tile op that produced the input tile.
  xtile::ExtractTileOp input_extract;
  // The entire shape of the input to all reduce.
  llvm::SmallVector<int64_t, 4> non_tiled_input_shape;
  AllReduceStrategy strategy;
  // Total number of elements in the input to all reduce.
  int64_t num_elements;
};

absl::StatusOr<AllReduceEmitterContext> CreateAllReduceEmitterContext(
    mlir::stablehlo::AllReduceOp op) {
  AllReduceEmitterContext ctx;
  if (op.getOperands().size() != 1) {
    return absl::InvalidArgumentError(
        "AllReduce op must have exactly one operand in order to be lowered "
        "to triton.");
  }
  ctx.xtile_entry_fn = op->getParentOfType<xtile::EntryFuncOp>();
  if (!ctx.xtile_entry_fn) {
    return absl::InvalidArgumentError(
        "AllReduce op must be in an XTile entry function in order to be "
        "lowered to triton.");
  }
  // Variadics are not supported yet so we can fix inputs to 1.
  // Which means 2 arguments for input/output one for scratch buffers and
  // 3 metadata arguments. Plus 1 for the tile index for a total of 7.
  ctx.num_input_output_args = op->getNumOperands() * 2;
  ctx.num_scratch_buffers = op->getNumOperands();
  const int32_t expected_num_args =
      ctx.num_input_output_args + ctx.num_scratch_buffers +
      kNumCollectiveMetadataArgs + kNumTileIndexArgs;
  if (ctx.xtile_entry_fn.getNumArguments() != expected_num_args) {
    return absl::InvalidArgumentError(
        absl::StrCat("AllReduce op must have ", expected_num_args,
                     " arguments in order to "
                     "be lowered to triton, but it has ",
                     ctx.xtile_entry_fn.getNumArguments()));
  }
  ctx.input_tile = mlir::cast<xtile::TensorValue>(op->getOperand(0));
  // We assume the input to all reduce is an xtile::ExtractTileOp, or that the
  // parent of the input is an xtile::ExtractTileOp (edge case for booleans).
  ctx.input_extract =
      llvm::dyn_cast<xtile::ExtractTileOp>(ctx.input_tile.getDefiningOp());
  if (!ctx.input_extract &&
      ctx.input_tile.getDefiningOp()->getNumOperands() > 0) {
    // Workaround(i1_to_i8_workaround).
    // Go one place up this is an edge case for booleans
    // Booleans are stored as i8 and then casted to i1 so the tile we get is
    // after the cast. To get the extract tile we need to go one step up.
    ctx.input_extract = llvm::dyn_cast<xtile::ExtractTileOp>(
        ctx.input_tile.getDefiningOp()->getOperand(0).getDefiningOp());
  }
  if (!ctx.input_extract) {
    return absl::InvalidArgumentError(
        "AllReduce op must have an extract tile op as operand in order to be "
        "lowered to triton.");
  }
  // The source for the extract is the non-tiled input (memref).
  ctx.non_tiled_input_shape = llvm::SmallVector<int64_t, 4>(
      ctx.input_extract.getSource().getType().getShape());
  ctx.num_elements = Product(ctx.non_tiled_input_shape);
  int64_t input_byte_size =
      ctx.num_elements *
      llvm::divideCeil(mlir::cast<mlir::ShapedType>(ctx.input_tile.getType())
                           .getElementTypeBitWidth(),
                       8);
  ctx.strategy = GetAllReduceStrategy(input_byte_size,
                                      /*is_multimem_enabled=*/false);
  ctx.op = op;
  return ctx;
}

// Returns the AllReduceInfo for the given all-reduce instruction if the
// instruction is supported by the codegen.
std::optional<AllReduceInfo> MaybeBuildAllReduceInfo(
    const HloAllReduceInstruction* all_reduce) {
  if (!all_reduce->GetModule()
           ->config()
           .debug_options()
           .xla_gpu_unsupported_use_all_reduce_one_shot_kernel()) {
    return std::nullopt;
  }
  if (all_reduce->device_list().replica_groups().empty()) {
    VLOG(1) << "Replica groups are empty for " << all_reduce->name()
            << ". Codegen will not be supported.";
    return std::nullopt;
  }
  const int64_t num_devices = all_reduce->device_list().num_devices_per_group();
  const std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(all_reduce->called_computations().front());
  if (!reduction_kind.has_value()) {
    return std::nullopt;
  }
  // TODO(b/383125489): Support variadic all-reduce.
  if (all_reduce->operand_count() > 1) {
    return std::nullopt;
  }
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_reduce->operand(0)->shape());
  const PrimitiveType element_type =
      all_reduce->operand(0)->shape().element_type();
  const int64_t byte_size =
      num_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  // NB: We do not codegen multimem kernels for now.
  const AllReduceStrategy all_reduce_strategy =
      GetAllReduceStrategy(byte_size, /*is_multimem_enabled=*/false);
  if (!IsAllReduceKernelSupported(num_devices, num_elements, element_type,
                                  reduction_kind.value(),
                                  all_reduce_strategy)) {
    return std::nullopt;
  }
  return AllReduceInfo{
      /* .reduction_kind= */ reduction_kind.value(),
      /* .num_devices= */ num_devices,
      /* .num_elements= */ num_elements,
      /* .element_type= */ element_type,
      /* .all_reduce_strategy= */ all_reduce_strategy,
  };
}

// The logic here is very naive and assumes a monotonic layout
// where only the last dimension is used as a tiling dimension.
absl::StatusOr<std::optional<BlockLevelFusionConfig>>
GetBlockLevelFusionConfigForAllReduce(
    const se::DeviceDescription& device_info,
    const HloAllReduceInstruction* all_reduce) {
  const std::optional<AllReduceInfo> all_reduce_info =
      MaybeBuildAllReduceInfo(all_reduce);
  if (!all_reduce_info.has_value()) {
    VLOG(3) << "AllReduceInfo is not available for " << all_reduce->name()
            << ". Codegen will not be supported.";
    return std::nullopt;
  }
  const Shape& output_shape = all_reduce->shape();
  const LaunchDimensions launch_dims = AllReduceLaunchDimensions(
      all_reduce_info->num_elements, all_reduce_info->num_devices,
      all_reduce_info->all_reduce_strategy);
  BlockLevelFusionConfig block_level_config;
  block_level_config.set_num_warps(launch_dims.num_threads_per_block() /
                                   WarpSize(device_info));
  block_level_config.set_num_ctas(1);    // No block-level clustering.
  block_level_config.set_num_stages(1);  // No pipelining of loops.
  Tile* output_tile = block_level_config.add_output_tiles();
  const int64_t rank = output_shape.dimensions().size();

  // Tile sizes are rolled up to power of 2 because this is what triton expects
  // and consequently the tiling infra.
  for (int i = 0; i < rank - 1; ++i) {
    output_tile->add_sizes(llvm::PowerOf2Ceil(output_shape.dimensions(i)));
  }
  // The last dimension is divided amongst blocks.
  if (rank > 0) {
    const int64_t tile_size =
        CeilOfRatio(output_shape.dimensions(rank - 1),
                    absl::implicit_cast<int64_t>(launch_dims.num_blocks()));
    const int64_t last_dimension = llvm::PowerOf2Ceil(tile_size);
    output_tile->add_sizes(last_dimension);
    if (all_reduce_info->all_reduce_strategy == AllReduceStrategy::kTwoShot &&
        last_dimension % all_reduce_info->num_devices != 0) {
      VLOG(3) << "Last dimension of output tile '" << last_dimension
              << "' is not divisible by number of devices '"
              << all_reduce_info->num_devices << "' for " << all_reduce->name()
              << ". Codegen will not be supported.";
      return std::nullopt;
    }
  }
  return block_level_config;
}

absl::StatusOr<std::vector<Shape>> GetAllReduceUnmanagedKernelArguments(
    const HloComputation* computation,
    const HloAllReduceInstruction* all_reduce) {
  const int32_t num_devices = all_reduce->device_list().num_devices_per_group();
  std::vector<Shape> unmanaged_arguments;
  unmanaged_arguments.reserve(computation->num_parameters() +
                              kNumCollectiveMetadataArgs);

  // rank and signal_value
  unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));
  unmanaged_arguments.push_back(ShapeUtil::MakeShape(S32, {}));
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
               computation->num_parameters() + kNumCollectiveMetadataArgs);
  return unmanaged_arguments;
}

mlir::LogicalResult PopulateReductionComputation(
    mlir::PatternRewriter& rewriter, mlir::stablehlo::AllReduceOp op,
    ReductionComputationEmitter& computation_emitter) {
  // At the moment we expect only one operation in the reduction computation
  // to be relevant.
  auto& reduction_computation_region = op.getComputation();
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

class AllReduceEmitter {
 public:
  static mlir::LogicalResult Emit(AllReduceEmitterContext ctx,
                                  mlir::PatternRewriter& rewriter) {
    AllReduceEmitter emitter(std::move(ctx), rewriter);
    if (mlir::failed(emitter.Initialize())) {
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
  AllReduceEmitter(AllReduceEmitterContext ctx, mlir::PatternRewriter& rewriter)
      : ctx_(std::move(ctx)),
        rewriter_(rewriter),
        builder_(ctx_.op->getLoc(), rewriter) {}

  mlir::LogicalResult Initialize() {
    CHECK(!initialized_);
    // NB: This must be done before any other IR is emitted so that we can bail
    // out in case it fails.
    // Otherwise, the IR is considered modified and we end up in an infinite
    // loop.
    if (mlir::failed(PopulateReductionComputation(
            rewriter_, ctx_.op, reduce_computation_emitter_))) {
      return mlir::failure();
    }
    // 1. Opaque arguments. They start after the input/output arguments.
    const int32_t start_idx = ctx_.num_input_output_args;
    device_rank_ = ctx_.xtile_entry_fn.getArgument(start_idx);
    CHECK(device_rank_.getType().isInteger(32));
    signal_value_ = ctx_.xtile_entry_fn.getArgument(start_idx + 1);
    CHECK(signal_value_.getType().isInteger(32));
    // !tt.ptr<!tt.ptr<i32>>
    signal_buffers_ = ctx_.xtile_entry_fn.getArgument(start_idx + 2);
    // !tt.ptr<!tt.ptr<i64>>
    remote_input_buffers_ = ctx_.xtile_entry_fn.getArgument(start_idx + 3);

    // 2. Constants and types.
    world_size_ = ctx_.op.getReplicaGroups().getShapedType().getDimSize(1);

    elem_type_ = mlir::getElementTypeOrSelf(ctx_.input_tile.getType());
    elem_storage_type_ = xtile::StorageType(elem_type_);
    ptr_to_i64_type_ =
        ttir::PointerType::get(builder_.getI64Type(), kGlobalAddressSpace);
    ptr_to_elem_type_ =
        ttir::PointerType::get(elem_storage_type_, kGlobalAddressSpace);
    remote_memref_type_ = mlir::MemRefType::get(
        ctx_.input_extract.getSource().getType().getShape(),
        elem_storage_type_);

    // 3. Emit setup IR.
    remote_input_buffers_i64_ = ttir::BitcastOp::create(
        builder_, ptr_to_i64_type_, remote_input_buffers_);

    const mlir::Type i64_type = builder_.getI64Type();
    // Check if last bit of signal_value is 0 or 1.
    mlir::Value signal_value = signal_value_;
    // For two-shot signal value is always incremented by 2.
    // So we need to right shift it by 1 to get the buffer index.
    if (ctx_.strategy == AllReduceStrategy::kTwoShot) {
      signal_value = arith::ShRSIOp::create(
          builder_, signal_value.getType(), signal_value,
          arith::ConstantOp::create(builder_, signal_value.getType(),
                                    builder_.getI32IntegerAttr(1)));
    }
    mlir::Value buffer_index = arith::AndIOp::create(
        builder_, i64_type,
        arith::ExtSIOp::create(builder_, i64_type, signal_value),  // i32->i64
        arith::ConstantOp::create(builder_, i64_type,
                                  builder_.getI64IntegerAttr(1)));
    buffer_offset_ = arith::MulIOp::create(
        builder_, i64_type, buffer_index,
        arith::ConstantOp::create(
            builder_, i64_type, builder_.getI64IntegerAttr(ctx_.num_elements)));

    initialized_ = true;
    return mlir::success();
  }

  // Emits instructions to get the pointer to the remote buffer of the given
  // rank.
  // We have a !tt.ptr<!tt.ptr<i64>> pointing to the base of the remote
  // buffers. We add the rank index to the base pointer and load to get to the
  // base pointer of the remote buffer of the given rank. Then we add the buffer
  // offset to get the pointer to the correct buffer inside (double buffering).
  // Finally, we convert the pointer to a memref for xtile.ExtractTileOp.
  mlir::Value GetRemoteBufferMemref(mlir::Value rank_idx) {
    CHECK(initialized_);
    mlir::Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
        builder_, ptr_to_i64_type_, remote_input_buffers_i64_, rank_idx);
    mlir::Value remote_buf_i64 =
        ttir::LoadOp::create(builder_,                      //
                             remote_buf_ptr_addr,           //
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             /*isVolatile=*/false);         //
    mlir::Value remote_buf_ptr_base =
        ttir::IntToPtrOp::create(builder_, ptr_to_elem_type_, remote_buf_i64,
                                 llvm::ArrayRef<mlir::NamedAttribute>{
                                     xtile::GetDivisibilityAttr(builder_)});
    mlir::Value remote_buf_ptr = ttir::AddPtrOp::create(
        builder_, ptr_to_elem_type_, remote_buf_ptr_base, buffer_offset_);
    return mtx::PtrToMemrefOp::create(builder_, remote_memref_type_,
                                      remote_buf_ptr);
  }

  // Loads a tile from the remote buffer of the given rank.
  // Offsets must be global offsets ie, from the beginning of the remote buffer.
  xtile::TensorValue LoadTileForRank(mlir::Value rank_idx,
                                     mlir::ValueRange offsets,
                                     llvm::ArrayRef<int64_t> shape) {
    mlir::Value remote_buf_memref = GetRemoteBufferMemref(rank_idx);
    auto tensor_type = mlir::RankedTensorType::get(shape, elem_storage_type_);
    xtile::TensorValue next_tile = xtile::ExtractTileOp::create(
        builder_, tensor_type, remote_buf_memref, offsets, shape,
        ctx_.input_extract.getStrides());
    // # Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // See fusion emitter for more details.
    if (elem_storage_type_ != elem_type_) {
      next_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, next_tile, elem_type_));
    }
    return next_tile;
  }

  // Overload for integer rank.
  xtile::TensorValue LoadTileForRank(int32_t rank, mlir::ValueRange offsets,
                                     llvm::ArrayRef<int64_t> sub_tile_shape) {
    mlir::Value rank_idx = arith::ConstantOp::create(
        builder_, builder_.getI64Type(), builder_.getI64IntegerAttr(rank));
    return LoadTileForRank(rank_idx, offsets, sub_tile_shape);
  }

  mlir::LogicalResult EmitCopyToSymmetric() {
    CHECK(initialized_);
    mlir::Value remote_buf_memref = GetRemoteBufferMemref(device_rank_);

    // Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // The parameter extraction casts the storage type to the logical type.
    // But for copying to the remote buffer we need to cast it back to the
    // storage type. Downstream passes should be able to optimize this away.
    mlir::Value storage_tile = ctx_.input_tile;
    if (elem_storage_type_ != elem_type_) {
      storage_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, ctx_.input_tile, elem_storage_type_));
    }
    xtile::InsertTileOp::create(
        builder_, storage_tile, remote_buf_memref,
        ctx_.input_extract.getOffsets(),
        ctx_.input_extract.getTile().getType().getShape(),
        ctx_.input_extract.getStrides());
    return mlir::success();
  }

  mlir::LogicalResult EmitSync(mlir::Value signal_value) {
    CHECK(initialized_);
    // All threads in the block should have completed their writes before we
    // proceed with a block barrier.
    // Otherwise, remote ranks might start reading the data before it is ready.
    mlir::triton::gpu::LocalBarrierOp::create(builder_);
    mtx::BlockBarrierOp::create(builder_, signal_buffers_, device_rank_,
                                signal_value,
                                builder_.getI32IntegerAttr(world_size_));
    return mlir::success();
  }

  // Calculates the offsets and shape of the sub-tile that the given rank is
  // responsible for. Only valid for two-shot.
  // - The last dimension of the shape is divided amongst ranks.
  //   So each rank has a shape of [x, y, z / world_size].
  // - The offsets are the same as input tile except the last dimension which
  //   is offset by the last_tile_size_per_rank * rank_idx.
  //   So this becomes [x, y, z / world_size * rank_idx] for rank_idx.
  // Note: The offsets are global offsets ie, from the beginning of the input
  // buffer. So for example if the num_elements is 1024 and the tiles are of
  // size 512 with 2 ranks. For two-shot, each ranks responsibility is 512/2 =
  // 256 per tile. Output shape is always [256]. Offsets are:
  //  - Rank0Tile0: [0]; Rank0Tile1: [512]
  //  - Rank1Tile0: [256]; Rank1Tile1: [768]
  std::pair<llvm::SmallVector<mlir::Value>, llvm::SmallVector<int64_t>>
  CalculateSubtileOffsetsAndShape(mlir::Value rank_idx) {
    llvm::SmallVector<int64_t> sub_tile_shape =
        llvm::to_vector(ctx_.input_tile.getType().getShape());
    const int32_t last_dim = sub_tile_shape.size() - 1;
    // We assume the tiled dimension is the last one.
    // Crash OK.
    // Precondition checked during GetBlockLevelFusionConfigForAllReduce.
    CHECK_EQ(sub_tile_shape[last_dim] % world_size_, 0)
        << "Tiled dimension not divisible by world size.";
    sub_tile_shape[last_dim] /= world_size_;
    const mlir::Type i64_type = builder_.getI64Type();
    const mlir::Value sub_tile_offset_i64 = arith::MulIOp::create(
        builder_, i64_type,
        arith::ExtSIOp::create(builder_, i64_type, rank_idx),
        arith::ConstantOp::create(
            builder_, i64_type,
            builder_.getI64IntegerAttr(sub_tile_shape[last_dim])));
    const mlir::Value sub_tile_offset = arith::IndexCastOp::create(
        builder_, builder_.getIndexType(), sub_tile_offset_i64);
    llvm::SmallVector<mlir::Value> sub_tile_offsets =
        llvm::to_vector(ctx_.input_extract.getOffsets());
    sub_tile_offsets[last_dim] = arith::AddIOp::create(
        builder_, sub_tile_offsets[last_dim], sub_tile_offset);
    return std::make_pair(sub_tile_offsets, sub_tile_shape);
  }

  // Emits instructions to load a tile from the remote buffers of all ranks
  // based on the responsibility of each rank.
  // So for a tile of size T with R ranks, each rank will be responsible for
  // T / R elements.
  // This method will arrange the pointers to size T such that
  // first T/R elements point to rank 0, next T/R elements point to rank 1, etc.
  // And then perform a tensor<Tx!ptr<elem_type>> gathered load.
  absl::StatusOr<xtile::TensorValue> EmitGatherLoad() {
    mlir::RankedTensorType tile_type = ctx_.input_tile.getType();
    const mlir::ArrayRef<int64_t> tile_shape = tile_type.getShape();
    llvm::SmallVector<int64_t> subtile_shape(tile_shape.begin(),
                                             tile_shape.end());
    subtile_shape.back() /= world_size_;
    // tensor<tile_shape x !ptr<elem_type>>
    auto tensor_of_ptrs_type =
        mlir::RankedTensorType::get(tile_shape, ptr_to_elem_type_);
    // tensor<tile_shape x !ptr<i64>>
    auto tensor_of_i64_ptrs_type =
        mlir::RankedTensorType::get(tile_shape, ptr_to_i64_type_);
    const mlir::Type i32_type = builder_.getI32Type();
    TF_ASSIGN_OR_RETURN(const llvm::SmallVector<int64_t> layout,
                        xtile::GetPermutationMinorToMajor(
                            ctx_.input_extract.getSource().getType()));
    // Create 1D range [0, full_shape.back)
    mlir::Value range = ttir::MakeRangeOp::create(
        builder_, mlir::RankedTensorType::get({tile_shape.back()}, i32_type), 0,
        tile_shape.back());
    // Broadcast range to full_shape (to match rank_id for all elements)
    // Eg for world_size = 2, and tile_shape = [2, 4], we will have
    // iota_list_dim = [0, 1, 2, 3], [0, 1, 2, 3]
    mlir::Value iota_last_dim = triton::ExpandAndBroadcastValue(
        builder_, range, tile_shape.size() - 1,
        mlir::RankedTensorType::get(tile_shape, i32_type));
    mlir::Value sub_tile_size_const = arith::ConstantOp::create(
        builder_, i32_type, builder_.getI32IntegerAttr(subtile_shape.back()));
    mlir::Value sub_tile_size_splat = ttir::SplatOp::create(
        builder_, iota_last_dim.getType(), sub_tile_size_const);
    // rank_ids = [0, 0, 1, 1] (Following from above example)
    // This determines which elements are to be loaded from which rank.
    // Eg: [0, 0, 1, 1] means first two elements from rank 0, next two elements
    // from rank 1.
    mlir::Value rank_ids =
        arith::DivSIOp::create(builder_, iota_last_dim, sub_tile_size_splat);
    // Gather from all ranks.
    // peer_ptr_addrs = remote_input_buffers_i64_ + rank_ids
    mlir::Value base_table_splat = ttir::SplatOp::create(
        builder_, tensor_of_i64_ptrs_type, remote_input_buffers_i64_);
    mlir::Value peer_ptr_addrs = ttir::AddPtrOp::create(
        builder_, tensor_of_i64_ptrs_type, base_table_splat, rank_ids);

    // Load the 64-bit addresses from the table
    mlir::Value peer_base_i64 =
        ttir::LoadOp::create(builder_,                      //
                             peer_ptr_addrs,                //
                             /*mask=*/mlir::Value(),        //
                             /*other=*/mlir::Value(),       //
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             /*isVolatile=*/false);
    // Create tensor of pointers: tensor<tile_shape x !ptr<elem_type>>
    mlir::Value peer_ptrs =
        ttir::IntToPtrOp::create(builder_, tensor_of_ptrs_type, peer_base_i64);
    // Add the shared buffer offset (e.g. for double buffering)
    // FinalAddress = PeerBase + SharedBufferOffset + LocalOffset
    mlir::Value buffer_offsets = ttir::SplatOp::create(
        builder_,
        mlir::RankedTensorType::get(tile_shape, builder_.getI64Type()),
        buffer_offset_);
    // tensor<tile_shape x !ptr<elem_type>>
    mlir::Value peer_ptrs_with_buf_offset = ttir::AddPtrOp::create(
        builder_, tensor_of_ptrs_type, peer_ptrs, buffer_offsets);
    auto [final_ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_,                   //
        peer_ptrs_with_buf_offset,  // The tensor of rank-specific base pointers
        ctx_.non_tiled_input_shape,       // The full global shape
        layout,                           //
        ctx_.input_extract.getOffsets(),  // The global base offsets of the tile
        ctx_.input_extract.getFullTileShape(),  // The full tile shape
        ctx_.input_extract.getStrides(),
        /*reduced_dims=*/{},  // Not reducing for gather
        tile_shape);
    // The final gather load tensor<tile_shape x elem_type>
    return mlir::cast<xtile::TensorValue>(
        ttir::LoadOp::create(builder_,                      //
                             final_ptrs,                    //
                             mask,                          //
                             /*other=*/mlir::Value(),       //
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             /*isVolatile=*/false)
            .getResult());
  }

  mlir::LogicalResult EmitOneShot() {
    CHECK(initialized_);
    // 1. CopyPhase: Local tile to the symmetric buffer for the current device.
    if (mlir::failed(EmitCopyToSymmetric())) {
      return mlir::failure();
    }
    // 2. Synchronization phase: Wait for all ranks to complete the copy.
    if (mlir::failed(EmitSync(signal_value_))) {
      return mlir::failure();
    }
    // 3. Reduce phase: Load tiles from all ranks and reduce them.
    mlir::ValueRange offsets = ctx_.input_extract.getOffsets();
    llvm::ArrayRef<int64_t> shape =
        mlir::cast<mlir::ShapedType>(ctx_.input_tile.getType()).getShape();
    xtile::TensorValue accumulator = LoadTileForRank(0, offsets, shape);
    for (int rank = 1; rank < world_size_; ++rank) {
      xtile::TensorValue next_tile = LoadTileForRank(rank, offsets, shape);
      accumulator =
          reduce_computation_emitter_(builder_, accumulator, next_tile);
    }
    rewriter_.replaceOp(ctx_.op, accumulator.getDefiningOp());
    return mlir::success();
  }

  mlir::LogicalResult EmitTwoShot() {
    CHECK(initialized_);
    // 1. CopyPhase: Local tile to the symmetric buffer for the current device.
    if (mlir::failed(EmitCopyToSymmetric())) {
      return mlir::failure();
    }
    // 2. Shot1: Wait for all ranks to complete the copy.
    if (mlir::failed(EmitSync(signal_value_))) {
      return mlir::failure();
    }
    // 3. Reduce phase:
    // 3.1 Accumulate what each rank is responsible for.
    auto [self_offsets, self_shape] =
        CalculateSubtileOffsetsAndShape(device_rank_);
    xtile::TensorValue accumulator =
        LoadTileForRank(0, self_offsets, self_shape);
    for (int rank = 1; rank < world_size_; ++rank) {
      xtile::TensorValue next_tile =
          LoadTileForRank(rank, self_offsets, self_shape);
      accumulator =
          reduce_computation_emitter_(builder_, accumulator, next_tile);
    }
    // 3.2 Copy reduced sub-tile back to local rank's remote buffer.
    mlir::Value remote_buf_memref = GetRemoteBufferMemref(device_rank_);
    mlir::Value storage_tile = accumulator;
    if (elem_storage_type_ != elem_type_) {
      storage_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, accumulator, elem_storage_type_));
    }
    xtile::InsertTileOp::create(builder_, storage_tile, remote_buf_memref,
                                self_offsets, self_shape,
                                ctx_.input_extract.getStrides());
    // 4. Shot2: Wait for all ranks to complete the reduce.
    mlir::Value next_signal_value = arith::AddIOp::create(
        builder_, signal_value_,
        arith::ConstantOp::create(builder_, builder_.getI32Type(),
                                  builder_.getI32IntegerAttr(1)));
    if (mlir::failed(EmitSync(next_signal_value))) {
      return mlir::failure();
    }
    // 5. Gather from all ranks to output tile.
    absl::StatusOr<xtile::TensorValue> gathered_tensor = EmitGatherLoad();
    if (!gathered_tensor.ok()) {
      VLOG(3) << "Failed to emit gathered load: " << gathered_tensor.status();
      return mlir::failure();
    }
    rewriter_.replaceOp(ctx_.op, gathered_tensor.value());
    return mlir::success();
  }

  AllReduceEmitterContext ctx_;
  mlir::PatternRewriter& rewriter_;
  mlir::ImplicitLocOpBuilder builder_;

  ReductionComputationEmitter reduce_computation_emitter_{nullptr};

  mlir::Value device_rank_;
  mlir::Value signal_value_;
  mlir::Value signal_buffers_;
  mlir::Value remote_input_buffers_;

  mlir::Value remote_input_buffers_i64_;
  mlir::Value buffer_offset_;

  int64_t world_size_;

  mlir::Type elem_type_;
  mlir::Type elem_storage_type_;
  ttir::PointerType ptr_to_i64_type_;
  ttir::PointerType ptr_to_elem_type_;
  mlir::MemRefType remote_memref_type_;

  bool initialized_ = false;
};

}  // namespace

absl::StatusOr<std::optional<BlockLevelFusionConfig>>
GetCollectiveBlockLevelFusionConfig(const se::DeviceDescription& device_info,
                                    const HloFusionInstruction* fusion_instr) {
  const HloInstruction* root = fusion_instr->fused_expression_root();
  switch (root->opcode()) {
    case HloOpcode::kAllReduceStart:
      return GetBlockLevelFusionConfigForAllReduce(
          device_info, Cast<HloAllReduceInstruction>(root));
    default:
      return std::nullopt;
  }
}

absl::StatusOr<bool> TrySetGpuBackendConfigForCollective(
    const se::DeviceDescription& device_info,
    HloFusionInstruction* fusion_instr) {
  TF_ASSIGN_OR_RETURN(
      const std::optional<BlockLevelFusionConfig> block_config,
      GetCollectiveBlockLevelFusionConfig(device_info, fusion_instr));
  if (!block_config.has_value()) {
    VLOG(3) << "No block level fusion config calculated for collective: "
            << fusion_instr->ToString()
            << ". Not using Triton collective fusion.";
    return false;
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      fusion_instr->backend_config<GpuBackendConfig>());
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      kTritonCollectiveFusionKind);
  *gpu_backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() = *std::move(block_config);
  TF_RETURN_IF_ERROR(
      fusion_instr->set_backend_config(std::move(gpu_backend_config)));
  return true;
}

absl::StatusOr<std::vector<Shape>> GetCollectiveUnmanagedKernelArguments(
    const HloFusionInstruction* fusion) {
  const HloComputation* computation = fusion->fused_instructions_computation();
  const HloInstruction* root = computation->root_instruction();
  switch (root->opcode()) {
    case HloOpcode::kAllReduceStart:
      return GetAllReduceUnmanagedKernelArguments(
          computation, Cast<HloAllReduceInstruction>(root));
    default:
      return std::vector<Shape>();
  }
}

absl::StatusOr<int32_t> AddCollectiveMetadataArguments(
    llvm::SmallVector<mlir::Type>& fn_arg_types, mlir::ImplicitLocOpBuilder& b,
    const HloComputation* hlo_computation) {
  // rank: i32
  fn_arg_types.push_back(b.getI32Type());
  // signal_value: i32
  fn_arg_types.push_back(b.getI32Type());
  // signal_buffers: !tt.ptr<!tt.ptr<i32>>
  fn_arg_types.push_back(ttir::PointerType::get(
      ttir::PointerType::get(b.getI32Type(), kGlobalAddressSpace),
      kGlobalAddressSpace));
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    PrimitiveType type = p->shape().element_type();
    mlir::Type ir_type;
    if (type == U16) {
      ir_type = b.getI16Type();
    } else if (type == S4) {
      ir_type = b.getI4Type();
    } else {
      TF_ASSIGN_OR_RETURN(ir_type, xtile::PrimitiveTypeToMlirType(b, type));
    }
    // Also add the remote/scratch buffers for collectives.
    // !tt.ptr<!tt.ptr<type>>
    fn_arg_types.push_back(ttir::PointerType::get(
        ttir::PointerType::get(xtile::StorageType(ir_type),
                               kGlobalAddressSpace),
        kGlobalAddressSpace));
  }
  // num_metadata_args =
  return hlo_computation->num_parameters() + kNumCollectiveMetadataArgs;
}

mlir::LogicalResult RewriteAllReduce(mlir::stablehlo::AllReduceOp op,
                                     mlir::PatternRewriter& rewriter) {
  const mlir::Location loc = op->getLoc();
  absl::StatusOr<AllReduceEmitterContext> maybe_context =
      CreateAllReduceEmitterContext(op);
  if (!maybe_context.ok()) {
    return rewriter.notifyMatchFailure(
        loc, absl::StrCat("Failed to create AllReduceEmitterContext: ",
                          maybe_context.status().message()));
  }
  VLOG(3) << "AllReduceEmitter::Emit using strategy: "
          << maybe_context->strategy;
  return AllReduceEmitter::Emit(maybe_context.value(), rewriter);
}

}  // namespace xla::gpu
