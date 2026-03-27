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
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
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
#include "xla/layout_util.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
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
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace xla::gpu {
namespace {

using ::xla::se::gpu::AllReduceStrategy;

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
  int32_t num_input_output_args{0};
  int32_t num_scratch_buffers{0};
  // The entry function of the all reduce op.
  xtile::EntryFuncOp xtile_entry_fn;
  // The input tile to all reduce.
  xtile::TensorValue input_tile;
  // The extract tile op that produced the input tile.
  xtile::ExtractTileOp input_extract;
  // The entire shape of the input to all reduce.
  llvm::SmallVector<int64_t, 4> non_tiled_input_shape;
  PrimitiveType element_type;
  // The total number of devices in the all reduce.
  int64_t world_size{0};
  AllReduceStrategy strategy;
  // Total number of elements in the input to all reduce.
  int64_t num_elements{0};
};

absl::StatusOr<AllReduceEmitterContext> CreateAllReduceEmitterContext(
    mlir::stablehlo::AllReduceOp op) {
  AllReduceEmitterContext ctx;
  if (op.getOperands().size() != 1) {
    return absl::InvalidArgumentError(
        "AllReduce op must have exactly one operand in order to be lowered "
        "to triton.");
  }
  // operand(0) is xtile.extract op.
  mlir::Type element_type =
      mlir::cast<mlir::ShapedType>(op.getOperand(0).getType()).getElementType();
  ctx.element_type = xla::ConvertMlirTypeToPrimitiveType(element_type);
  if (ctx.element_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) {
    std::string type_string;
    llvm::raw_string_ostream stream(type_string);
    op.getOperand(0).print(stream);
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not convert operand type to a valid PrimitiveType."
        "Operand Type: %s",
        type_string));
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
  ctx.world_size = op.getReplicaGroups().getShapedType().getDimSize(1);
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
    VLOG(1)
        << "Skipping all-reduce codegen because "
           "xla_gpu_unsupported_use_all_reduce_one_shot_kernel is disabled.";
    return std::nullopt;
  }
  if (all_reduce->device_list()->replica_groups().empty()) {
    VLOG(1) << "Replica groups are empty for " << all_reduce->name()
            << ". Codegen will not be supported.";
    return std::nullopt;
  }
  const int64_t num_devices =
      all_reduce->device_list()->num_devices_per_group();
  if (!llvm::has_single_bit(static_cast<uint64_t>(num_devices))) {
    VLOG(1) << "Number of devices is not a power of 2 for "
            << all_reduce->name() << ". Codegen will not be supported.";
    return std::nullopt;
  }
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
  const int64_t max_supported_all_reduce_size_bytes =
      GetMaxSupportedAllReduceSizeBytes(all_reduce_strategy);
  if (byte_size > max_supported_all_reduce_size_bytes) {
    VLOG(3) << "Codegen forall-reduce is only supported for small inputs."
            << max_supported_all_reduce_size_bytes << " <" << byte_size;
    return std::nullopt;
  }
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
  const auto compute_capability = device_info.cuda_compute_capability();
  if (!compute_capability.IsAtLeastHopper()) {
    VLOG(3) << "Collective codegen requires compute capability of at least "
               "9.0. Got "
            << compute_capability.ToString()
            << ". Codegen will not be supported.";
    return std::nullopt;
  }
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
  block_level_config.set_num_warps(xla::CeilOfRatio(
      static_cast<int64_t>(launch_dims.num_threads_per_block()),
      WarpSize(device_info)));
  block_level_config.set_num_ctas(1);    // No block-level clustering.
  block_level_config.set_num_stages(1);  // No pipelining of loops.
  Tile* output_tile = block_level_config.add_output_tiles();
  const llvm::SmallVector<int64_t> tile_sizes =
      GreedyPowerOfTwoTiles(output_shape, launch_dims.num_blocks());
  output_tile->mutable_sizes()->Assign(tile_sizes.begin(), tile_sizes.end());
  const int64_t linear_tile_size = Product(tile_sizes);
  if (all_reduce_info->all_reduce_strategy == AllReduceStrategy::kTwoShot &&
      linear_tile_size % all_reduce_info->num_devices != 0) {
    VLOG(3) << "Two-shot all-reduce linear_tile_size(" << linear_tile_size
            << ") % num_devices(" << all_reduce_info->num_devices
            << ") != 0. Codegen will not be supported.";
    return std::nullopt;
  }
  VLOG(3) << "Block level fusion config for " << all_reduce->name() << ": "
          << block_level_config;
  return block_level_config;
}

absl::StatusOr<std::vector<Shape>> GetAllReduceUnmanagedKernelArguments(
    const HloComputation* computation,
    const HloAllReduceInstruction* all_reduce) {
  const int32_t num_devices =
      all_reduce->device_list()->num_devices_per_group();
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
  AllReduceEmitter(AllReduceEmitterContext ctx, mlir::PatternRewriter& rewriter)
      : ctx_(std::move(ctx)),
        rewriter_(rewriter),
        builder_(ctx_.op->getLoc(), rewriter) {}

  absl::Status Initialize() {
    CHECK(!initialized_);
    // NB: This must be done before any other IR is emitted so that we can bail
    // out in case it fails.
    // Otherwise, the IR is considered modified and we end up in an infinite
    // loop.
    if (mlir::failed(PopulateReductionComputation(
            rewriter_, ctx_.op, reduce_computation_emitter_))) {
      return absl::InternalError("Failed to populate reduction computation.");
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
    elem_type_ = mlir::getElementTypeOrSelf(ctx_.input_tile.getType());
    elem_storage_type_ = xtile::StorageType(elem_type_);
    ptr_to_i64_type_ =
        ttir::PointerType::get(builder_.getI64Type(), kGlobalAddressSpace);
    ptr_to_elem_type_ =
        ttir::PointerType::get(elem_storage_type_, kGlobalAddressSpace);
    TF_ASSIGN_OR_RETURN(layout_, xtile::GetPermutationMinorToMajor(
                                     ctx_.input_extract.getSource().getType()));

    const llvm::ArrayRef<int64_t>& input_tile_shape_dims =
        ctx_.input_tile.getType().getShape();
    Shape input_tile_shape = ShapeUtil::MakeShapeWithDenseLayout(
        ctx_.element_type, input_tile_shape_dims, layout_);
    // Subtile shape for one-shot is the same as the input tile shape.
    subtile_shape_ = {input_tile_shape_dims.begin(),
                      input_tile_shape_dims.end()};
    // For two-shot, divide the tile into num_devices tiles.
    if (ctx_.strategy == AllReduceStrategy::kTwoShot) {
      subtile_shape_ = GreedyPowerOfTwoTiles(input_tile_shape, ctx_.world_size);
      // Make sure subtile shape perfectly divides the input tile shape.
      // Crash Ok. This is an internal precondition which is always expected to
      // be true. Internal tile shape is 2^n / 2^m should be perfectly divisible
      // for n >= m.
      CHECK_EQ(Product(input_tile_shape.dimensions()) % Product(subtile_shape_),
               0)
          << "Input tile shape is not perfectly divisible by subtile shape."
          << "Input tile shape: " << input_tile_shape
          << "Subtile shape: " << absl::StrJoin(subtile_shape_, ",");
    }
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
    // The allocated double buffer size in bytes is aligned to
    // kXlaAllocatedBufferAlignBytes. To get the offset to the second buffer, we
    // divide the buffer size by 2 and then divide by the element size to get
    // the number of elements in the buffer. Its important to do this to make
    // sure that start of the buffers are always aligned to 16 bytes.
    const int64_t buffer_size = xla::RoundUpTo<uint64_t>(
        ctx_.num_elements *
            ShapeUtil::ByteSizeOfPrimitiveType(ctx_.element_type),
        kXlaAllocatedBufferAlignBytes);
    const int64_t elements_per_buffer =
        buffer_size / ShapeUtil::ByteSizeOfPrimitiveType(ctx_.element_type);
    buffer_offset_ = arith::MulIOp::create(
        builder_, i64_type, buffer_index,
        arith::ConstantOp::create(
            builder_, i64_type,
            builder_.getI64IntegerAttr(elements_per_buffer)));
    initialized_ = true;
    return absl::OkStatus();
  }

  // Emits instructions to get the pointer to the remote buffer of the given
  // rank.
  // We have a !tt.ptr<!tt.ptr<i64>> pointing to the base of the remote
  // buffers. We add the rank index to the base pointer and load to get to the
  // base pointer of the remote buffer of the given rank. Then we add the buffer
  // offset to get the pointer to the correct buffer inside (double buffering).
  // Note that the returned pointer is of the storage type not the logical type.
  mlir::Value GetRemoteBufferPtr(mlir::Value rank_idx) {
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
    return remote_buf_ptr;
  }

  // Loads a tile from the remote buffer of the given rank.
  // Offsets must be global offsets ie, from the beginning of the remote buffer.
  // Shape must be exact shape of the tile to be loaded.
  // For 1-shot this is the entire tile shape. For two-shot this is the subtile
  // shape.
  xtile::TensorValue LoadTileForRank(mlir::Value rank_idx,
                                     mlir::ValueRange offsets,
                                     llvm::ArrayRef<int64_t> strides,
                                     llvm::ArrayRef<int64_t> shape) {
    CHECK(initialized_);
    mlir::Value remote_buf_ptr = GetRemoteBufferPtr(rank_idx);
    auto [ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_,        //
        remote_buf_ptr,  // The tensor of rank-specific base pointers
        ctx_.non_tiled_input_shape,  // The full global shape
        layout_,                     // The layout of the input tensor
        offsets,                     // The global base offsets of the tile
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
        ttir::LoadOp::create(builder_,                      //
                             ptrs,                          //
                             mask,                          //
                             /*other=*/mlir::Value(),       //
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             /*isVolatile=*/false)
            .getResult());
    // Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // See fusion emitter for more details.
    if (elem_storage_type_ != elem_type_) {
      next_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, next_tile, elem_type_));
    }
    return next_tile;
  }

  // Overload for integer rank.
  xtile::TensorValue LoadTileForRank(int32_t rank, mlir::ValueRange offsets,
                                     llvm::ArrayRef<int64_t> strides,
                                     llvm::ArrayRef<int64_t> sub_tile_shape) {
    mlir::Value rank_idx = arith::ConstantOp::create(
        builder_, builder_.getI64Type(), builder_.getI64IntegerAttr(rank));
    return LoadTileForRank(rank_idx, offsets, strides, sub_tile_shape);
  }

  // Stores an entire tile to the symmetric buffer of device_rank_.
  mlir::LogicalResult EmitCopyToSymmetric(mlir::Value tile_to_store,
                                          mlir::ValueRange offsets,
                                          llvm::ArrayRef<int64_t> strides,
                                          llvm::ArrayRef<int64_t> shape) {
    CHECK(initialized_);
    mlir::Value remote_buf_ptr = GetRemoteBufferPtr(device_rank_);

    // Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // The parameter extraction casts the storage type to the logical type.
    // But for copying to the remote buffer we need to cast it back to the
    // storage type. Downstream passes should be able to optimize this away.
    mlir::Value storage_tile = tile_to_store;
    if (elem_storage_type_ != elem_type_) {
      storage_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder_, tile_to_store, elem_storage_type_));
    }
    auto [ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_,        //
        remote_buf_ptr,  // The tensor of rank-specific base pointers
        ctx_.non_tiled_input_shape,  // The full global shape
        layout_,                     // The layout of the input tensor
        offsets,                     // The global base offsets of the tile
        shape,                       // The full tile shape. Same as
                                     // input shape since no
                                     // dimensions are reduced.
        strides,                     //
        /*reduced_dims=*/{},         // Not reducing scatter.
        shape                        // The tile shape.
    );
    ttir::StoreOp::create(builder_, ptrs, storage_tile,
                          /*mask=*/mask, ttir::CacheModifier::NONE,
                          ttir::EvictionPolicy::NORMAL);
    return mlir::success();
  }

  mlir::LogicalResult EmitSync(mlir::Value signal_value) {
    CHECK(initialized_);
    // All threads in the block should have completed their writes before we
    // proceed with a block barrier.
    // Otherwise, remote ranks might start reading the data before it is ready.
    mlir::triton::gpu::BarrierOp::create(builder_,
                                         mlir::triton::gpu::AddrSpace::Local);
    mtx::BlockBarrierOp::create(builder_, signal_buffers_, device_rank_,
                                signal_value,
                                builder_.getI32IntegerAttr(ctx_.world_size));
    return mlir::success();
  }

  // Returns the offsets of the sub-tile that the given rank is responsible for.
  // These are local offsets with respect to the tile.
  // For loading from the remote buffer, we need to add the global offsets.
  llvm::SmallVector<mlir::Value> RankToLocalOffsets(
      mlir::Value rank_idx, llvm::ArrayRef<int64_t> tile_dims,
      llvm::ArrayRef<int64_t> subtile_dims) {
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
      int layout_dim = layout_[i];
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
  llvm::SmallVector<mlir::Value> CalculateSubtileOffsets(mlir::Value rank_idx) {
    auto local_offsets = RankToLocalOffsets(
        rank_idx, ctx_.input_tile.getType().getShape(), subtile_shape_);
    // Global offsets
    llvm::SmallVector<mlir::Value> offsets = ctx_.input_extract.getOffsets();
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
      mlir::RankedTensorType tile_type, llvm::ArrayRef<int64_t> subtile_shape,
      int64_t world_size) {
    const llvm::ArrayRef<int64_t> tile_shape = tile_type.getShape();
    const mlir::RankedTensorType tensor_of_world_size_ptr_of_ptrs =
        mlir::RankedTensorType::get({world_size}, ptr_to_i64_type_);
    mlir::Value remote_buffers = ttir::SplatOp::create(
        builder_, tensor_of_world_size_ptr_of_ptrs, remote_input_buffers_i64_);
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
    remote_buffers = ttir::LoadOp::create(builder_,                      //
                                          remote_buffers,                //
                                          /*mask=*/mlir::Value(),        //
                                          /*other=*/mlir::Value(),       //
                                          ttir::CacheModifier::NONE,     //
                                          ttir::EvictionPolicy::NORMAL,  //
                                          /*isVolatile=*/false)
                         .getResult();
    // tensor<world_size x !ptr<elem_type>>
    const mlir::RankedTensorType tensor_of_world_size_ptrs =
        mlir::RankedTensorType::get({world_size}, ptr_to_elem_type_);
    remote_buffers = ttir::IntToPtrOp::create(
        builder_, tensor_of_world_size_ptrs, remote_buffers,
        {xtile::GetDivisibilityAttr(builder_)});
    mlir::Value buffer_offsets = ttir::SplatOp::create(
        builder_,
        mlir::RankedTensorType::get({world_size}, builder_.getI64Type()),
        buffer_offset_);
    // Add the buffer_offset to the base of each loaded pointer.
    remote_buffers = ttir::AddPtrOp::create(builder_, tensor_of_world_size_ptrs,
                                            remote_buffers, buffer_offsets);
    // Add a new dimension of size 1 to the tensor of pointers.
    // tensor<world_size x 1 x !ptr<elem_type>>
    remote_buffers =
        ttir::ExpandDimsOp::create(builder_, remote_buffers, /*axis=*/1);
    // Broadcast to RxNumElementsPerRank
    mlir::Type broadcasted_type = mlir::RankedTensorType::get(
        {world_size, Product(subtile_shape)}, ptr_to_elem_type_);
    remote_buffers =
        ttir::BroadcastOp::create(builder_, broadcasted_type, remote_buffers);
    llvm::SmallVector<int64_t> physical_shape(tile_shape.size(), 0);
    bool requires_transpose = false;
    for (int i = 0; i < tile_shape.size(); ++i) {
      const auto idx = layout_[layout_.size() - i - 1];
      physical_shape[i] = tile_shape[idx];
      if (layout_[i] != layout_.size() - 1 - i) {
        requires_transpose = true;
      }
    }
    // Reshape to tile shape.
    remote_buffers = ttir::ReshapeOp::create(
        builder_,
        mlir::RankedTensorType::get(physical_shape, ptr_to_elem_type_),
        remote_buffers);
    if (requires_transpose) {
      llvm::SmallVector<int32_t> permutation(layout_.size());
      for (int i = 0; i < layout_.size(); ++i) {
        permutation[layout_[i]] = layout_.size() - 1 - i;
      }
      remote_buffers = ttir::TransOp::create(
          builder_, mlir::RankedTensorType::get(tile_shape, ptr_to_elem_type_),
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
  absl::StatusOr<xtile::TensorValue> EmitGatherLoad() {
    mlir::RankedTensorType tile_type = ctx_.input_tile.getType();
    const mlir::ArrayRef<int64_t> tile_shape = tile_type.getShape();
    // tensor<tile_shape x !ptr<elem_type>>
    // Each pointer within subtile i points to data in the remote buffer of
    // rank i.
    xtile::TensorValue remote_buffer_tensor = CreateTensorOfRemoteBufferPtrs(
        tile_type, subtile_shape_, ctx_.world_size);
    auto [final_ptrs, mask] = triton::CreateTensorOfPointersAndMask(
        builder_,              //
        remote_buffer_tensor,  // The tensor of rank-specific base pointers
        ctx_.non_tiled_input_shape,       // The full global shape
        layout_,                          //
        ctx_.input_extract.getOffsets(),  // The global base offsets of the tile
        ctx_.input_extract.getFullTileShape(),  //
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
    // Lift scalar inputs to a tensor of shape {1}.
    llvm::SmallVector<mlir::Value> offsets = ctx_.input_extract.getOffsets();
    llvm::SmallVector<int64_t> strides{ctx_.input_extract.getStrides()};
    const bool is_scalar = offsets.empty();
    if (is_scalar) {
      ctx_.input_tile = xtile::Splat(builder_, ctx_.input_tile, {1});
      ctx_.non_tiled_input_shape = {1};
      layout_ = {0};
      subtile_shape_ = {1};
      mlir::Value c0 = builder_.create<arith::ConstantIndexOp>(0);
      offsets = {c0};
      strides = {1};
    }
    // 1. CopyPhase: Local tile to the symmetric buffer for the current device.
    if (mlir::failed(
            EmitCopyToSymmetric(ctx_.input_tile, offsets, strides,
                                ctx_.input_tile.getType().getShape()))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit copy to symmetric");
    }
    // 2. Synchronization phase: Wait for all ranks to complete the copy.
    if (mlir::failed(EmitSync(signal_value_))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit sync for one-shot");
    }
    // 3. Reduce phase: Load tiles from all ranks and reduce them.
    llvm::ArrayRef<int64_t> shape = ctx_.input_tile.getType().getShape();
    xtile::TensorValue accumulator =
        LoadTileForRank(0, offsets, strides, shape);
    for (int32_t rank = 1; rank < ctx_.world_size; ++rank) {
      xtile::TensorValue next_tile =
          LoadTileForRank(rank, offsets, strides, shape);
      accumulator =
          reduce_computation_emitter_(builder_, accumulator, next_tile);
    }
    mlir::Value result = accumulator;
    if (is_scalar) {
      result = ttir::UnsplatOp::create(builder_, accumulator);
      result = mlir::tensor::FromElementsOp::create(
          builder_, mlir::RankedTensorType::get({}, elem_type_), result);
    }
    rewriter_.replaceOp(ctx_.op, result);
    return mlir::success();
  }

  mlir::LogicalResult EmitTwoShot() {
    CHECK(initialized_);
    // 1. CopyPhase: Local tile to the symmetric buffer for the current device.
    llvm::ArrayRef<int64_t> strides = ctx_.input_extract.getStrides();
    if (mlir::failed(EmitCopyToSymmetric(
            ctx_.input_tile, ctx_.input_extract.getOffsets(), strides,
            ctx_.input_tile.getType().getShape()))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit copy to symmetric");
    }
    // 2. Shot1: Wait for all ranks to complete the copy.
    if (mlir::failed(EmitSync(signal_value_))) {
      return rewriter_.notifyMatchFailure(ctx_.op,
                                          "Failed to emit sync for shot1");
    }
    // 3. Reduce phase:
    // 3.1 Accumulate what each rank is responsible for.
    llvm::SmallVector<mlir::Value> self_offsets =
        CalculateSubtileOffsets(device_rank_);
    xtile::TensorValue accumulator =
        LoadTileForRank(0, self_offsets, strides, subtile_shape_);
    for (int rank = 1; rank < ctx_.world_size; ++rank) {
      xtile::TensorValue next_tile =
          LoadTileForRank(rank, self_offsets, strides, subtile_shape_);
      accumulator =
          reduce_computation_emitter_(builder_, accumulator, next_tile);
    }
    // 3.2 Copy reduced sub-tile back to local rank's remote buffer.
    if (mlir::failed(EmitCopyToSymmetric(accumulator, self_offsets, strides,
                                         subtile_shape_))) {
      return rewriter_.notifyMatchFailure(
          ctx_.op, "Failed to emit copy result to symmetric for two-shot");
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
    // 5. Gather from all ranks to output tile.
    absl::StatusOr<xtile::TensorValue> gathered_tensor = EmitGatherLoad();
    if (!gathered_tensor.ok()) {
      return rewriter_.notifyMatchFailure(
          ctx_.op, absl::StrCat("Failed to emit gathered load: ",
                                gathered_tensor.status().message()));
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

  // Layout of the input tensor in minor-to-major order.
  llvm::SmallVector<int64_t> layout_;
  // Calculated sub-tile shape for the all reduce.
  // For 1-shot this is the same as the tile shape. Since each rank operates on
  // the entire tile.
  // For 2-shot: The sub-tile is the tile shape divided by the
  // number of devices. Since tile shape is a power of 2 and the number of
  // devices is a power of 2, the sub-tile shape will also be a power of 2  and
  // consequently work with the tiling infra.
  llvm::SmallVector<int64_t> subtile_shape_;

  mlir::Type elem_type_;
  mlir::Type elem_storage_type_;
  ttir::PointerType ptr_to_i64_type_;
  ttir::PointerType ptr_to_elem_type_;

  bool initialized_ = false;
};

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
    VLOG(3) << "Failed to create AllReduceEmitterContext: "
            << maybe_context.status().message();
    return rewriter.notifyMatchFailure(
        loc, absl::StrCat("Failed to create AllReduceEmitterContext: ",
                          maybe_context.status().message()));
  }
  VLOG(3) << "AllReduceEmitter::Emit using strategy: "
          << maybe_context->strategy;
  return AllReduceEmitter::Emit(maybe_context.value(), rewriter);
}

}  // namespace xla::gpu
