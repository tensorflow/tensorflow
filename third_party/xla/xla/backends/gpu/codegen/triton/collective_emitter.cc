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
#include <functional>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
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

namespace xla::gpu {
namespace {

using ::mlir::Value;
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

struct AllReduceInfo {
  ReductionKind reduction_kind;
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
  AllReduceStrategy all_reduce_strategy;
};

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
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_reduce->operand(0)->shape());
  const PrimitiveType element_type =
      all_reduce->operand(0)->shape().element_type();
  // NB: We do not codegen multimem kernels for now.
  const AllReduceStrategy all_reduce_strategy =
      GetAllReduceStrategy(num_elements, /*is_multimem_enabled=*/false);
  // TODO(b/383125489): Support variadic all-reduce.
  if (all_reduce->operand_count() > 1) {
    return std::nullopt;
  }
  const int64_t byte_size =
      num_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  // TODO(b/457333991): Support twoShot for codegen.
  if (byte_size >
      GetMaxSupportedAllReduceSizeBytes(AllReduceStrategy::kOneShot)) {
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
  const std::optional<AllReduceInfo> all_reduce_info =
      MaybeBuildAllReduceInfo(all_reduce);
  if (!all_reduce_info.has_value()) {
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
    output_tile->add_sizes(llvm::PowerOf2Ceil(tile_size));
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

mlir::LogicalResult populateReductionComputation(
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

// TODO(b/474276020)
mlir::LogicalResult RewriteAllReduce(mlir::stablehlo::AllReduceOp op,
                                     mlir::PatternRewriter& rewriter) {
  if (op.getOperands().size() != 1) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        "AllReduce op must have exactly one operand in order to be lowered "
        "to triton.");
  }

  // Find the entry function and get the arguments from there.
  auto xtile_entry_fn = op->getParentOfType<xtile::EntryFuncOp>();

  if (!xtile_entry_fn) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        "AllReduce op must be in an XTile entry function in order to be "
        "lowered to triton.");
  }

  // Variadics are not supported yet so we can fix inputs to 1.
  // Which means 2 arguments for input/output one for scratch buffers and 3
  // metadata arguments. Plus 1 for the tile index for a total of 7.
  const int32_t num_input_output_args = op.getNumOperands() * 2;
  const int32_t num_scratch_buffers = op.getNumOperands();
  static constexpr int32_t kNumTileIndexArgs = 1;
  if (xtile_entry_fn.getNumArguments() !=
      (num_input_output_args + num_scratch_buffers +
       kNumCollectiveMetadataArgs + kNumTileIndexArgs)) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        absl::StrCat("AllReduce op must have ",
                     num_input_output_args + num_scratch_buffers +
                         kNumCollectiveMetadataArgs + kNumTileIndexArgs,
                     " arguments in order to "
                     "be lowered to triton, but it has ",
                     xtile_entry_fn.getNumArguments()));
  }

  // Opaque arguments start after the input/output arguments.
  const int32_t start_idx = num_input_output_args;
  mlir::Value device_rank = xtile_entry_fn.getArgument(start_idx);
  CHECK(device_rank.getType().isInteger(32));
  mlir::Value signal_value = xtile_entry_fn.getArgument(start_idx + 1);
  CHECK(signal_value.getType().isInteger(32));
  // !tt.ptr<!tt.ptr<i32>>
  mlir::Value signal_buffers = xtile_entry_fn.getArgument(start_idx + 2);
  // !tt.ptr<!tt.ptr<i64>>
  mlir::Value remote_input_buffers = xtile_entry_fn.getArgument(start_idx + 3);

  // We assume the input to all reduce is an xtile::ExtractTileOp, or that the
  // parent of the input is an xtile::ExtractTileOp (edge case for booleans).
  auto input_tile = op.getOperand(0);
  auto input_tile_op =
      llvm::dyn_cast<xtile::ExtractTileOp>(input_tile.getDefiningOp());

  if (!input_tile_op && input_tile.getDefiningOp()->getNumOperands() > 0) {
    // Go one place up - this is an edge case for booleans
    // Workaround(i1_to_i8_workaround).
    input_tile_op = llvm::dyn_cast<xtile::ExtractTileOp>(
        input_tile.getDefiningOp()->getOperand(0).getDefiningOp());
  }
  if (!input_tile_op) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        "AllReduce op must have an extract tile op as operand in order to be "
        "lowered to triton.");
  }

  auto non_tiled_input_shape = input_tile_op.getSource().getType().getShape();

  const int64_t num_elements = std::accumulate(
      non_tiled_input_shape.begin(), non_tiled_input_shape.end(), /*init=*/1,
      std::multiplies<int64_t>());

  ReductionComputationEmitter reduce_computation_emitter = nullptr;

  // We have to perform this verification before emitting any triton specific
  // mlir so that we can fall back if the computation is invalid.
  if (mlir::failed(populateReductionComputation(rewriter, op,
                                                reduce_computation_emitter))) {
    return mlir::failure();
  }

  mlir::ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  // 1. Scatter phase: Copy local tile to the remote buffer of the current
  // rank.
  const auto ptr_to_i64_type =
      ttir::PointerType::get(builder.getI64Type(), kGlobalAddressSpace);
  auto remote_input_buffers_i64 =
      ttir::BitcastOp::create(builder, ptr_to_i64_type, remote_input_buffers);

  const mlir::Type i64_type = builder.getI64Type();
  const mlir::Type elem_type = mlir::getElementTypeOrSelf(input_tile.getType());
  const mlir::Type elem_storage_type = xtile::StorageType(elem_type);
  const auto ptr_to_elem_type =
      ttir::PointerType::get(elem_storage_type, kGlobalAddressSpace);
  constexpr int32_t kBitsPerByte = 8;
  const int64_t remote_buffer_size =
      num_elements * (elem_storage_type.getIntOrFloatBitWidth() / kBitsPerByte);
  Value buffer_index = arith::AndIOp::create(
      builder, i64_type,
      arith::ExtSIOp::create(builder, i64_type, signal_value),
      arith::ConstantOp::create(builder, i64_type,
                                builder.getI64IntegerAttr(1)));
  Value buffer_offset = arith::MulIOp::create(
      builder, i64_type, buffer_index,
      arith::ConstantOp::create(builder, i64_type,
                                builder.getI64IntegerAttr(remote_buffer_size)));
  // Helper function to get the buffer pointer for a given signal value.
  const auto get_buffer_ptr = [&](mlir::Value buffer_ptr_base) -> mlir::Value {
    return ttir::AddPtrOp::create(builder, ptr_to_elem_type, buffer_ptr_base,
                                  buffer_offset);
  };

  mlir::ArrayRef<int64_t> remote_shape = non_tiled_input_shape;
  const mlir::MemRefType remote_memref_type =
      mlir::MemRefType::get(remote_shape, elem_storage_type);
  // Scoped to reuse variable names during reduction phase.
  {
    Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
        builder, ptr_to_i64_type, remote_input_buffers_i64, device_rank);
    Value remote_buf_i64 =
        ttir::LoadOp::create(builder, remote_buf_ptr_addr,
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             false);                        // isVolatile
    Value remote_buf_ptr_base =
        ttir::IntToPtrOp::create(builder, ptr_to_elem_type, remote_buf_i64,
                                 llvm::ArrayRef<mlir::NamedAttribute>{
                                     xtile::GetDivisibilityAttr(builder)});
    Value remote_buf_ptr = get_buffer_ptr(remote_buf_ptr_base);
    mlir::Value remote_buf_memref =
        mtx::PtrToMemrefOp::create(builder, remote_memref_type, remote_buf_ptr);
    // Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // The parameter extraction casts the storage type to the logical type.
    // But for copying to the remote buffer we need to cast it back to the
    // storage type. Downstream passes should be able to optimize this away.
    mlir::Value storage_tile = input_tile;
    if (elem_storage_type != elem_type) {
      storage_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder, input_tile, elem_storage_type));
    }
    xtile::InsertTileOp::create(builder, storage_tile, remote_buf_memref,
                                input_tile_op.getOffsets(),
                                input_tile_op.getTile().getType().getShape(),
                                input_tile_op.getStrides());
  }

  // 2. Synchronization phase: Wait for all ranks to complete the scatter.
  int64_t world_size = op.getReplicaGroups().getShapedType().getDimSize(1);
  mtx::BlockBarrierOp::create(builder, signal_buffers, device_rank,
                              signal_value,
                              builder.getI32IntegerAttr(world_size));

  // 3. Reduce phase: Load tiles from all ranks and reduce them.
  const auto load_tile_for_rank = [&](int64_t rank) {
    Value rank_idx = arith::ConstantOp::create(builder, builder.getI64Type(),
                                               builder.getI64IntegerAttr(rank));
    Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
        builder, ptr_to_i64_type, remote_input_buffers_i64, rank_idx);
    Value remote_buf_i64 =
        ttir::LoadOp::create(builder, remote_buf_ptr_addr,
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             false);                        // isVolatile
    Value remote_buf_ptr_base =
        ttir::IntToPtrOp::create(builder, ptr_to_elem_type, remote_buf_i64);
    Value remote_buf_ptr = get_buffer_ptr(remote_buf_ptr_base);
    Value remote_buf_memref =
        mtx::PtrToMemrefOp::create(builder, remote_memref_type, remote_buf_ptr);

    auto tensor_type = mlir::RankedTensorType::get(
        input_tile_op.getTile().getType().getShape(), elem_storage_type);

    xtile::TensorValue next_tile = xtile::ExtractTileOp::create(
        builder, tensor_type, remote_buf_memref, input_tile_op.getOffsets(),
        input_tile_op.getTile().getType().getShape(),
        input_tile_op.getStrides());
    // # Workaround(i1_to_i8_workaround) as in fusion_emitter.
    // See fusion emitter for more details.
    if (elem_storage_type != elem_type) {
      next_tile = mlir::cast<xtile::TensorValue>(
          xtile::Cast(builder, next_tile, elem_type));
    }
    return next_tile;
  };

  xtile::TensorValue accumulator = load_tile_for_rank(0);

  for (int rank = 1; rank < world_size; ++rank) {
    xtile::TensorValue next_tile = load_tile_for_rank(rank);

    accumulator = reduce_computation_emitter(builder, accumulator, next_tile);
  }

  rewriter.replaceOp(op, accumulator.getDefiningOp());
  return mlir::success();
}

}  // namespace xla::gpu
