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
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/mlir_hlo/mhlo/utils/type_conversion.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace mlir_converter {
namespace {

using llvm::SmallVector;
using llvm::SmallVectorImpl;
using mlir::Block;
using mlir::FloatType;
using mlir::ImplicitLocOpBuilder;
using mlir::IRMapping;
using mlir::Location;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::AndIOp;
using mlir::arith::CmpIOp;
using mlir::arith::CmpIPredicate;
using mlir::arith::ConstantIndexOp;
using mlir::arith::ConstantOp;
using mlir::scf::IfOp;
using mlir::scf::YieldOp;

namespace arith = ::mlir::arith;
namespace mhlo = ::mlir::mhlo;
namespace scf = ::mlir::scf;

// HLO opcodes that we never support.
static auto& kUnsupportedOps =
    *new absl::flat_hash_set<HloOpcode>{HloOpcode::kAddDependency,
                                        HloOpcode::kAfterAll,
                                        HloOpcode::kAllGather,
                                        HloOpcode::kAllGatherDone,
                                        HloOpcode::kAllGatherStart,
                                        HloOpcode::kAllReduce,
                                        HloOpcode::kAllReduceDone,
                                        HloOpcode::kAllReduceStart,
                                        HloOpcode::kAllToAll,
                                        HloOpcode::kAsyncDone,
                                        HloOpcode::kAsyncStart,
                                        HloOpcode::kAsyncUpdate,
                                        HloOpcode::kBatchNormGrad,
                                        HloOpcode::kBatchNormInference,
                                        HloOpcode::kBatchNormTraining,
                                        HloOpcode::kCholesky,
                                        HloOpcode::kCollectivePermute,
                                        HloOpcode::kCollectivePermuteDone,
                                        HloOpcode::kCollectivePermuteStart,
                                        HloOpcode::kCopyDone,
                                        HloOpcode::kCopyStart,
                                        HloOpcode::kCustomCall,
                                        HloOpcode::kDomain,
                                        HloOpcode::kDynamicReshape,
                                        HloOpcode::kFft,
                                        HloOpcode::kFusion,
                                        HloOpcode::kGetDimensionSize,
                                        HloOpcode::kOptimizationBarrier,
                                        HloOpcode::kInfeed,
                                        HloOpcode::kOutfeed,
                                        HloOpcode::kPartitionId,
                                        HloOpcode::kRecv,
                                        HloOpcode::kRecvDone,
                                        HloOpcode::kReduceScatter,
                                        HloOpcode::kReplicaId,
                                        HloOpcode::kRng,
                                        HloOpcode::kRngBitGenerator,
                                        HloOpcode::kRngGetAndUpdateState,
                                        HloOpcode::kScatter,
                                        HloOpcode::kSelectAndScatter,
                                        HloOpcode::kSend,
                                        HloOpcode::kSendDone,
                                        HloOpcode::kSetDimensionSize,
                                        HloOpcode::kSort,
                                        HloOpcode::kTopK,
                                        HloOpcode::kTriangularSolve,
                                        HloOpcode::kWhile,
                                        HloOpcode::kConditional,
                                        HloOpcode::kStochasticConvert,
                                        HloOpcode::kCall};

bool IsUnsupportedGather(const HloInstruction* instr) {
  // We assume gather simplifier ran, so we don't need to support all gather
  // forms.
  if (instr->opcode() != HloOpcode::kGather) return false;

  auto* gather = Cast<HloGatherInstruction>(instr);
  const auto& dims = gather->gather_dimension_numbers();
  // We allow XLA:GPU's "canonical" gather (2D indices). And the form preferred
  // by the algebraic simplifier if the second index dimension is degenerate (1D
  // indices with implicit second dimension).
  int indices_rank = gather->operand(1)->shape().rank();
  if (dims.index_vector_dim() != 1 || !dims.collapsed_slice_dims().empty() ||
      indices_rank == 0 || indices_rank > 2) {
    return true;
  }

  for (auto [index, val] : llvm::enumerate(dims.start_index_map())) {
    if (index != val) return true;
  }
  for (auto [index, val] : llvm::enumerate(dims.offset_dims())) {
    if (index + 1 != val) return true;
  }
  return false;
}

absl::StatusOr<Value> GetSingleOperandValue(
    const OperandProvider& operand_provider, const HloInstruction* instr,
    int operand_index, ValueRange indices) {
  TF_ASSIGN_OR_RETURN(auto operand,
                      operand_provider(instr, operand_index, indices));
  TF_RET_CHECK(operand.size() == 1) << "Expected operand to be a single value.";
  return operand.front();
}

absl::StatusOr<SmallVector<Value>> EmitReduce(
    const HloInstruction* instr, ValueRange indices,
    const OperandProvider& operand_provider,
    const CallTargetProvider& call_target_provider, ImplicitLocOpBuilder& b) {
  auto* mlir_context = b.getContext();
  HloInstructionIndexing indexing =
      ComputeOutputToInputIndexing(instr, 0, mlir_context);
  const auto& indexing_map = *indexing.indexing_maps[0].begin();

  SmallVector<Value> init_values;
  for (int i = instr->operand_count() / 2; i < instr->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(init_values.emplace_back(),
                        GetSingleOperandValue(operand_provider, instr, i, {}));
  }

  auto body =
      [&](ValueRange iter_args, ValueRange dim_values,
          ValueRange symbol_values) -> absl::StatusOr<SmallVector<Value>> {
    auto indices = ApplyIndexing(indexing_map, dim_values, symbol_values, b);
    SmallVector<Value> args{iter_args};
    for (int i = 0; i < instr->operand_count() / 2; ++i) {
      TF_ASSIGN_OR_RETURN(
          args.emplace_back(),
          GetSingleOperandValue(operand_provider, instr, i, indices));
    }
    auto reducer = call_target_provider(
        instr->called_computations().front()->root_instruction());
    return b.create<mlir::func::CallOp>(reducer, args).getResults();
  };

  return EmitLoopNestWithStatus(b, indices, init_values, indexing_map, body);
}

absl::StatusOr<SmallVector<Value>> EmitReduceWindow(
    const HloInstruction* instr, mlir::Type result_element_type,
    ValueRange indices, const OperandProvider& operand_provider,
    const CallTargetProvider& call_target_provider, ImplicitLocOpBuilder& b) {
  MLIRContext* mlir_context = b.getContext();
  HloInstructionIndexing indexing =
      ComputeOutputToInputIndexing(instr, 0, mlir_context);
  auto indexing_map = *indexing.indexing_maps[0].begin();
  indexing_map.RescaleSymbols();

  auto reduce_window = DynCast<HloReduceWindowInstruction>(instr);
  CHECK(reduce_window != nullptr);

  SmallVector<Value> init_values;
  for (auto [index, init_value] :
       llvm::enumerate(reduce_window->init_values())) {
    TF_ASSIGN_OR_RETURN(
        init_values.emplace_back(),
        GetSingleOperandValue(operand_provider, instr,
                              reduce_window->input_count() + index, {}));
  }

  auto body =
      [&](ValueRange iter_args, ValueRange dim_values,
          ValueRange symbol_values) -> absl::StatusOr<SmallVector<Value>> {
    auto indices = ApplyIndexing(indexing_map, dim_values, symbol_values, b);
    SmallVector<Value> args{iter_args};
    for (auto [index, input] : llvm::enumerate(reduce_window->inputs())) {
      TF_ASSIGN_OR_RETURN(
          args.emplace_back(),
          GetSingleOperandValue(operand_provider, instr, index, indices));
    }

    auto reducer = call_target_provider(
        instr->called_computations().front()->root_instruction());
    return b.create<mlir::func::CallOp>(reducer, args).getResults();
  };

  return EmitLoopNestWithStatus(b, indices, init_values, indexing_map, body);
}

absl::StatusOr<SmallVector<Value>> EmitConcat(
    const HloInstruction* instr, mlir::Type result_element_type,
    ValueRange indices, const OperandProvider& operand_provider,
    ImplicitLocOpBuilder& b) {
  int concat_dim =
      Cast<HloConcatenateInstruction>(instr)->concatenate_dimension();
  int64_t offset = 0;
  IfOp outermost_if = nullptr;
  SmallVector<Value> operand_indices = indices;
  for (auto [index, operand] : llvm::enumerate(instr->operands())) {
    int64_t limit = offset + operand->shape().dimensions(concat_dim);
    auto ins = b.create<CmpIOp>(CmpIPredicate::ult, indices[concat_dim],
                                b.create<ConstantIndexOp>(limit));

    auto generate_operand = [&, index = index]() {
      operand_indices[concat_dim] = b.create<arith::SubIOp>(
          indices[concat_dim], b.create<ConstantIndexOp>(offset));
      TF_ASSIGN_OR_RETURN(auto operand,
                          operand_provider(instr, index, operand_indices));
      b.create<YieldOp>(operand);
      return absl::OkStatus();
    };

    if (index < instr->operand_count() - 1) {
      auto if_op =
          b.create<IfOp>(mlir::TypeRange{result_element_type}, ins, true, true);
      if (outermost_if == nullptr) {
        outermost_if = if_op;
      } else {
        b.create<YieldOp>(if_op.getResults());
      }

      b.setInsertionPointToStart(if_op.getBody(0));
      TF_RETURN_IF_ERROR(generate_operand());
      b.setInsertionPointToStart(if_op.getBody(1));
    } else {
      TF_RETURN_IF_ERROR(generate_operand());
    }
    offset = limit;
  }

  b.setInsertionPointAfter(outermost_if);
  return outermost_if.getResults();
}

absl::StatusOr<llvm::SmallVector<Value>> EmitDynamicSlice(
    const HloInstruction* instr, ValueRange indices,
    const OperandProvider& operand_provider, ImplicitLocOpBuilder& b) {
  llvm::SmallVector<Value> input_indices(indices);

  const auto& input_shape = instr->operand(0)->shape();
  for (int i = 0; i < input_shape.rank(); ++i) {
    TF_ASSIGN_OR_RETURN(
        auto offset, GetSingleOperandValue(operand_provider, instr, i + 1, {}));
    offset =
        ClampIndex(offset,
                   primitive_util::IsUnsignedIntegralType(
                       instr->operand(i + 1)->shape().element_type()),
                   input_shape.dimensions(i) - instr->shape().dimensions(i), b);
    input_indices[i] = b.create<arith::AddIOp>(input_indices[i], offset);
  }

  return operand_provider(instr, 0, input_indices);
}

absl::StatusOr<llvm::SmallVector<Value>> EmitDynamicUpdateSlice(
    const HloInstruction* instr, mlir::Type result_element_type,
    ValueRange indices, const OperandProvider& operand_provider,
    ImplicitLocOpBuilder& b) {
  mlir::Value is_in_bounds =
      b.create<ConstantOp>(b.getIntegerAttr(b.getI1Type(), 1));
  mlir::SmallVector<Value> update_indices;
  const auto& updates_shape = instr->operand(1)->shape();
  for (int i = 0; i < instr->shape().rank(); ++i) {
    int64_t update_size = updates_shape.dimensions(i);
    TF_ASSIGN_OR_RETURN(
        auto start_index,
        GetSingleOperandValue(operand_provider, instr, i + 2, {}));
    start_index = ClampIndex(start_index,
                             primitive_util::IsUnsignedIntegralType(
                                 instr->operand(i + 2)->shape().element_type()),
                             instr->shape().dimensions(i) - update_size, b);

    auto end_index = b.create<arith::AddIOp>(
        start_index, b.create<ConstantOp>(b.getIndexAttr(update_size)));

    is_in_bounds = b.create<AndIOp>(
        is_in_bounds,
        b.create<CmpIOp>(CmpIPredicate::sge, indices[i], start_index));
    is_in_bounds = b.create<AndIOp>(
        is_in_bounds,
        b.create<CmpIOp>(CmpIPredicate::slt, indices[i], end_index));

    update_indices.push_back(b.create<arith::SubIOp>(indices[i], start_index));
  }

  auto if_op = b.create<IfOp>(mlir::TypeRange{result_element_type},
                              is_in_bounds, true, true);
  b.setInsertionPointToStart(if_op.getBody(0));
  TF_ASSIGN_OR_RETURN(
      auto updated_value,
      GetSingleOperandValue(operand_provider, instr, 1, update_indices));
  b.create<YieldOp>(updated_value);

  b.setInsertionPointToStart(if_op.getBody(1));
  TF_ASSIGN_OR_RETURN(
      auto original_value,
      GetSingleOperandValue(operand_provider, instr, 0, indices));
  b.create<YieldOp>(original_value);

  b.setInsertionPointAfter(if_op);
  return if_op.getResults();
}

absl::StatusOr<llvm::SmallVector<Value>> EmitGather(
    const HloInstruction* instr, ValueRange indices,
    const OperandProvider& operand_provider, ImplicitLocOpBuilder& b) {
  auto row = indices[0];
  auto zero = b.create<ConstantIndexOp>(0);
  // Gather allows the index vector to contain fewer elements than the rank
  // of the input. In that case, the remaining indices are 0.
  SmallVector<Value> operand_indices(instr->operand(0)->shape().rank(), zero);

  // Produce start indices.
  // HLO allows the index vector dimension to be implicit, and the algebraic
  // simplifier prefers this form. Therefore, we need to check the rank of the
  // indices here and do the implicit reshape in place.
  const auto& indices_shape = instr->operand(1)->shape();
  int num_indices = indices_shape.rank() == 1 ? 1 : indices_shape.dimensions(1);
  for (int i = 0; i < num_indices; ++i) {
    auto i_val = i == 0 ? zero : b.create<ConstantIndexOp>(i);
    int64_t slice_size = instr->gather_slice_sizes()[i];
    int64_t input_size = instr->operand(0)->shape().dimensions()[i];
    // Read and clamp index.
    TF_ASSIGN_OR_RETURN(
        auto input_index,
        operand_provider(instr, 1,
                         indices_shape.rank() == 1 ? ValueRange{row}
                                                   : ValueRange{row, i_val}));
    TF_RET_CHECK(input_index.size() == 1)
        << "Expected operand to be a single value.";
    operand_indices[i] =
        ClampIndex(input_index.front(),
                   primitive_util::IsUnsignedIntegralType(
                       instr->operand(1)->shape().element_type()),
                   input_size - slice_size, b);
  }

  // Add offsets.
  for (int i = 0; i < operand_indices.size(); ++i) {
    operand_indices[i] =
        b.createOrFold<arith::AddIOp>(operand_indices[i], indices[i + 1]);
  }

  return operand_provider(instr, 0, operand_indices);
}

// For a given instruction, deduces the indices of each parameter that are
// needed for a given output index.
SmallVector<SmallVector<Value>> GetInputIndices(
    const HloInstructionIndexing& indexing, ValueRange output_indices,
    ImplicitLocOpBuilder& b) {
  SmallVector<SmallVector<Value>> indices;
  for (auto& maps : indexing.indexing_maps) {
    CHECK_EQ(maps.size(), 1);
    CHECK(!maps.begin()->IsUndefined());
    indices.push_back(ApplyIndexing(*maps.begin(), output_indices, {}, b));
  }
  return indices;
}

absl::StatusOr<SmallVector<Value>> EmitPad(
    const HloInstruction* instr, mlir::Type result_element_type,
    ValueRange indices, const OperandProvider& operand_provider,
    ImplicitLocOpBuilder& b) {
  auto indexing = ComputeOutputToInputIndexing(instr, 0, b.getContext());
  const auto& indexing_map = *indexing.indexing_maps[0].begin();
  mlir::Value is_in_bounds = CheckConstraints(indexing_map, indices, {}, b);

  auto if_op = b.create<IfOp>(mlir::TypeRange{result_element_type},
                              is_in_bounds, true, true);
  b.setInsertionPointToStart(if_op.getBody(0));
  TF_ASSIGN_OR_RETURN(auto input_value,
                      GetSingleOperandValue(
                          operand_provider, instr, 0,
                          GetInputIndices(indexing, indices,
                                          b)[0 /* indexing for operand 0 */]));
  b.create<YieldOp>(input_value);

  b.setInsertionPointToStart(if_op.getBody(1));
  TF_ASSIGN_OR_RETURN(auto padding_value,
                      GetSingleOperandValue(operand_provider, instr, 1, {}));
  b.create<YieldOp>(padding_value);

  b.setInsertionPointAfter(if_op);
  return if_op.getResults();
}

absl::StatusOr<Value> EmitFloatCast(Value value, mlir::Type target_type,
                                    ImplicitLocOpBuilder& b) {
  if (value.getType().getIntOrFloatBitWidth() <
      target_type.getIntOrFloatBitWidth()) {
    return b.create<arith::ExtFOp>(target_type, value);
  }
  if (value.getType().getIntOrFloatBitWidth() >
      target_type.getIntOrFloatBitWidth()) {
    return b.create<arith::TruncFOp>(target_type, value);
  }
  return value;
}

absl::StatusOr<Value> EmitMulAdd(Value lhs, Value rhs, Value accumulator,
                                 mlir::Type result_element_type,
                                 mlir::Type accumulator_type,
                                 ImplicitLocOpBuilder& b) {
  if (mlir::isa<FloatType>(result_element_type)) {
    if (result_element_type.isBF16()) {
      lhs = b.create<arith::ExtFOp>(b.getF32Type(), lhs);
      rhs = b.create<arith::ExtFOp>(b.getF32Type(), rhs);
    }
    TF_ASSIGN_OR_RETURN(
        Value casted,
        EmitFloatCast(b.create<arith::MulFOp>(lhs, rhs), accumulator_type, b));
    return b.create<arith::AddFOp>(accumulator, casted);
  }
  if (result_element_type.isInteger(1)) {
    return b.create<arith::OrIOp>(accumulator,
                                  b.create<arith::AndIOp>(lhs, rhs));
  }
  return b.create<arith::AddIOp>(accumulator,
                                 b.create<arith::MulIOp>(lhs, rhs));
}

absl::StatusOr<SmallVector<Value>> EmitDotLoop(
    const HloInstruction* instr, mlir::Type result_element_type,
    ValueRange indices, const OperandProvider& operand_provider,
    ImplicitLocOpBuilder& b) {
  HloInstructionIndexing indexing =
      ComputeOutputToInputIndexing(instr, /*output_id=*/0, b.getContext());
  const IndexingMap& lhs_indexing_map = *indexing.indexing_maps.at(0).begin();
  const IndexingMap& rhs_indexing_map = *indexing.indexing_maps.at(1).begin();

  const mlir::Type accumulator_type =
      result_element_type.isBF16() ? b.getF32Type() : result_element_type;
  Value accum_init_value =
      b.create<ConstantOp>(b.getZeroAttr(accumulator_type)).getResult();

  // For convolutions with `batch_group_count` > 1, there is an additional
  // symbol for LHS (group id) - ignore it for RHS.
  size_t rhs_symbol_count = rhs_indexing_map.GetSymbolCount();

  auto body =
      [&](ValueRange iter_args, ValueRange dim_values,
          ValueRange symbol_values) -> absl::StatusOr<SmallVector<Value>> {
    auto lhs_indices =
        ApplyIndexing(lhs_indexing_map, dim_values, symbol_values, b);
    auto rhs_indices =
        ApplyIndexing(rhs_indexing_map, dim_values,
                      symbol_values.take_front(rhs_symbol_count), b);

    TF_ASSIGN_OR_RETURN(Value lhs_value, GetSingleOperandValue(
                                             operand_provider, instr,
                                             /*operand_index=*/0, lhs_indices));
    TF_ASSIGN_OR_RETURN(Value rhs_value, GetSingleOperandValue(
                                             operand_provider, instr,
                                             /*operand_index=*/1, rhs_indices));
    Value accum = iter_args[0];

    TF_ASSIGN_OR_RETURN(
        accum, EmitMulAdd(lhs_value, rhs_value, accum, result_element_type,
                          accumulator_type, b));
    return {{accum}};
  };

  TF_ASSIGN_OR_RETURN(SmallVector<Value> results,
                      EmitLoopNestWithStatus(b, indices, {accum_init_value},
                                             lhs_indexing_map, body));
  if (result_element_type.isBF16()) {
    results[0] = b.create<arith::TruncFOp>(b.getBF16Type(), results[0]);
  }
  return results;
}

absl::StatusOr<SmallVector<Value>> EmitDot(
    const HloInstruction* instr, mlir::Type result_element_type,
    ValueRange indices, const OperandProvider& operand_provider,
    ImplicitLocOpBuilder& b) {
  VLOG(1) << "EmitDot: " << instr->ToString() << " "
          << llvm_ir::DumpToString(result_element_type);

  if (!algorithm_util::IsSupportedByElementalIrEmitter(
          instr->precision_config().algorithm())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Algorithm not supported by the ElementalIrEmitter: %s",
                        PrecisionConfig::Algorithm_Name(
                            instr->precision_config().algorithm())));
  }
  auto* dot = DynCast<HloDotInstruction>(instr);
  TF_RET_CHECK(dot != nullptr);
  if (dot->sparse_operands()) {
    return absl::UnimplementedError(
        "Sparse dot is supported by Triton emitter only.");
  }

  return EmitDotLoop(instr, result_element_type, indices, operand_provider, b);
}

absl::StatusOr<SmallVector<Value>> EmitConvolution(
    const HloInstruction* instr, mlir::Type result_element_type,
    ValueRange indices, const OperandProvider& operand_provider,
    ImplicitLocOpBuilder& b) {
  VLOG(1) << "EmitConvolution: " << instr->ToString() << " "
          << llvm_ir::DumpToString(result_element_type);

  return EmitDotLoop(instr, result_element_type, indices, operand_provider, b);
}

absl::StatusOr<SmallVector<Value>> EmitParameter(const HloInstruction* instr,
                                                 mlir::func::FuncOp this_fn,
                                                 ValueRange indices,
                                                 ImplicitLocOpBuilder& b) {
  Value value = this_fn.getArgument(instr->parameter_number());
  if (mlir::isa<mlir::TensorType>(value.getType())) {
    value = b.create<mlir::tensor::ExtractOp>(value, indices);
  } else {
    TF_RET_CHECK(indices.empty());
  }
  return {{value}};
}

template <typename MhloOp, typename... ExtraArgs>
SmallVector<Value> MapHloOp(mlir::Type result_type,
                            llvm::ArrayRef<mlir::Type> arg_types,
                            llvm::ArrayRef<Value> args, ImplicitLocOpBuilder& b,
                            ExtraArgs&&... extra_args) {
  return {mhlo::MhloOpToStdScalarOp::mapOpOfType<MhloOp>(
      b.getLoc(), result_type, arg_types,
      typename MhloOp::Adaptor(args, std::forward<ExtraArgs>(extra_args)...),
      &b)};
}

template <typename MhloOp>
SmallVector<Value> MapElementwiseOp(llvm::ArrayRef<mlir::Type> arg_types,
                                    llvm::ArrayRef<Value> args,
                                    ImplicitLocOpBuilder& b) {
  // We use the last argument's type because of select.
  return MapHloOp<MhloOp>(args.back().getType(), arg_types, args, b);
}

}  // namespace

Value UnrealizedConversionCast(mlir::Type type, Value value,
                               ImplicitLocOpBuilder& b) {
  SmallVector<Value> converted;
  b.createOrFold<mlir::UnrealizedConversionCastOp>(converted, type, value);
  return converted.front();
}

SmallVector<Value> UnrealizedConversionCast(mlir::TypeRange types,
                                            ValueRange values,
                                            ImplicitLocOpBuilder& b) {
  SmallVector<Value> converted;
  for (auto [type, value] : llvm::zip(types, values)) {
    converted.push_back(UnrealizedConversionCast(type, value, b));
  }
  return converted;
}

Value ApplyAffineExpr(mlir::AffineExpr expr, ValueRange dims,
                      ValueRange symbols, ImplicitLocOpBuilder& b) {
  // For unknown (but undoubtedly good) reasons, affine.apply removes unused
  // trailing dimensions, but only in the expression.
  while (!dims.empty() && !expr.isFunctionOfDim(dims.size() - 1)) {
    dims = dims.drop_back();
  }
  while (!symbols.empty() && !expr.isFunctionOfSymbol(symbols.size() - 1)) {
    symbols = symbols.drop_back();
  }
  SmallVector<Value> args(dims);
  absl::c_copy(symbols, std::back_inserter(args));
  return b.createOrFold<mlir::affine::AffineApplyOp>(expr, args);
}

SmallVector<Value> ApplyIndexing(const IndexingMap& map, ValueRange dims,
                                 ValueRange symbols, ImplicitLocOpBuilder& b) {
  SmallVector<Value> results;
  for (unsigned int i = 0; i < map.GetAffineMap().getNumResults(); ++i) {
    SmallVector<Value> result;
    b.createOrFold<ApplyIndexingOp>(result, dims, symbols, map.GetSubMap(i));
    results.append(result);
  }
  return results;
}

Value CheckConstraint(mlir::Value constrained_value, Interval range,
                      ImplicitLocOpBuilder& b) {
  auto lb = b.create<ConstantOp>(b.getIndexAttr(range.lower));
  if (range.IsPoint()) {
    return b.create<CmpIOp>(CmpIPredicate::eq, constrained_value, lb);
  }
  auto ub = b.create<ConstantOp>(b.getIndexAttr(range.upper));
  return b.create<AndIOp>(
      b.create<CmpIOp>(CmpIPredicate::sge, constrained_value, lb),
      b.create<CmpIOp>(CmpIPredicate::sle, constrained_value, ub));
}

Value CheckConstraints(const IndexingMap& map, ValueRange dims,
                       ValueRange symbols, ImplicitLocOpBuilder& b) {
  llvm::SmallVector<mlir::AffineExpr> expressions;
  for (auto&& [expression, _] : map.GetConstraints()) {
    expressions.push_back(expression);
  }

  // Construct an indexing for the constraints, so we can use `apply_indexing`.
  auto input_map = map.GetAffineMap();
  IndexingMap constraints_map{
      mlir::AffineMap::get(input_map.getNumDims(), input_map.getNumSymbols(),
                           expressions, input_map.getContext()),
      map.GetDimVars(), map.GetRangeVars(), map.GetRTVars()};
  llvm::SmallVector<Value> constraints_values =
      ApplyIndexing(constraints_map, dims, symbols, b);

  Value ret = b.create<ConstantOp>(b.getIntegerAttr(b.getI1Type(), 1));
  for (auto&& [value, expression_and_range] :
       llvm::zip(constraints_values, map.GetConstraints())) {
    ret = b.create<AndIOp>(
        ret, CheckConstraint(value, expression_and_range.second, b));
  }
  for (auto&& [index, bound] : llvm::enumerate(map.GetDimensionBounds())) {
    ret = b.create<AndIOp>(ret, CheckConstraint(dims[index], bound, b));
  }
  return ret;
}

namespace {

absl::StatusOr<SmallVector<Value>> HloToMlir(
    const HloInstruction* instr, mlir::func::FuncOp this_fn, ValueRange indices,
    const OperandProvider& operand_provider,
    const CallTargetProvider& call_target_provider,
    ImplicitLocOpBuilder& builder) {
  CHECK(!kUnsupportedOps.contains(instr->opcode())) << instr->ToShortString();

  auto element_type = instr->shape().element_type();
  mlir::Type result_type_with_sign;
  mlir::Type result_element_type;
  if (!instr->shape().IsTuple()) {
    TF_ASSIGN_OR_RETURN(result_type_with_sign,
                        ConvertPrimitiveTypeToMlirType(element_type, builder));
    result_element_type =
        PrimitiveTypeToMlirType(instr->shape().element_type(), builder);
  }

  auto* mlir_context = builder.getContext();
  // Handle ops that aren't elementwise and aren't just indexing
  // transformations.
  switch (instr->opcode()) {
    case HloOpcode::kConcatenate:
      return EmitConcat(instr, result_element_type, indices, operand_provider,
                        builder);
    case HloOpcode::kConstant: {
      TF_ASSIGN_OR_RETURN(auto value_attr, CreateDenseElementsAttrFromLiteral(
                                               instr->literal(), builder));
      // Convert to signless if needed.
      if (primitive_util::IsUnsignedIntegralType(
              instr->shape().element_type())) {
        value_attr = value_attr.mapValues(
            result_element_type, [](const llvm::APInt& i) { return i; });
      }

      if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
        if (primitive_util::IsComplexType(element_type)) {
          return {{builder.create<mlir::complex::ConstantOp>(
              result_element_type,
              mlir::cast<mlir::ArrayAttr>(
                  value_attr.getValues<mlir::Attribute>()[0]))}};
        }
        auto val = mlir::cast<mlir::TypedAttr>(
            value_attr.getValues<mlir::Attribute>()[0]);
        return {{builder.create<ConstantOp>(val).getResult()}};
      }
      auto constant = builder.create<ConstantOp>(value_attr).getResult();
      return {{builder.create<mlir::tensor::ExtractOp>(constant, indices)}};
    }
    case HloOpcode::kConvolution:
      return EmitConvolution(instr, result_element_type, indices,
                             operand_provider, builder);
    case HloOpcode::kDynamicSlice:
      return EmitDynamicSlice(instr, indices, operand_provider, builder);
    case HloOpcode::kDynamicUpdateSlice:
      return EmitDynamicUpdateSlice(instr, result_element_type, indices,
                                    operand_provider, builder);
    case HloOpcode::kGather:
      return EmitGather(instr, indices, operand_provider, builder);
    case HloOpcode::kIota: {
      auto index = indices[Cast<HloIotaInstruction>(instr)->iota_dimension()];
      auto index_type = builder.getIntegerType(
          mlir::DataLayout::closest(builder.getInsertionBlock()->getParentOp())
              .getTypeSizeInBits(index.getType()));
      index = builder.create<arith::IndexCastUIOp>(index_type, index);
      return {{mhlo::MhloOpToStdScalarOp::mapConvertOpToStdScalarOp(
          builder.getLoc(), result_type_with_sign, result_element_type,
          {index_type}, {index}, &builder)}};
    }
    case HloOpcode::kPad:
      return EmitPad(instr, result_element_type, indices, operand_provider,
                     builder);
    case HloOpcode::kDot:
      return EmitDot(instr, result_element_type, indices, operand_provider,
                     builder);
    case HloOpcode::kParameter:
      return EmitParameter(instr, this_fn, indices, builder);
    case HloOpcode::kReduce:
      return EmitReduce(instr, indices, operand_provider, call_target_provider,
                        builder);
    case HloOpcode::kReduceWindow:
      return EmitReduceWindow(instr, result_element_type, indices,
                              operand_provider, call_target_provider, builder);
    case HloOpcode::kTuple: {
      const auto* first_shape = &instr->shape().tuple_shapes(0);
      while (first_shape->IsTuple()) {
        first_shape = &first_shape->tuple_shapes(0);
      }
      CHECK_EQ(first_shape->rank(), indices.size())
          << "Indices for tuple must be for the first tuple element";
      SmallVector<Value> operands;
      for (int i = 0; i < instr->operand_count(); ++i) {
        llvm::SmallVector<Value> operand_indices;
        // The tuple shapes only need to be bitcast compatible, so insert
        // bitcasts where necessary.
        const auto* operand = instr->operand(i);
        const auto* operand_index_shape = &operand->shape();
        while (operand_index_shape->IsTuple()) {
          operand_index_shape = &operand_index_shape->tuple_shapes(0);
        }
        if (i > 0 && !ShapeUtil::EqualIgnoringElementType(
                         *first_shape, *operand_index_shape)) {
          auto operand_map =
              GetBitcastMap(*first_shape, *operand_index_shape, mlir_context);
          operand_indices = ApplyIndexing(operand_map, indices, {}, builder);
        } else {
          operand_indices = indices;
        }
        TF_ASSIGN_OR_RETURN(auto values,
                            operand_provider(instr, i, operand_indices));
        operands.append(values);
      }
      return operands;
    }
    case HloOpcode::kGetTupleElement: {
      // We have to generate the entire tuple, but since we don't support
      // internal tuple operations (only root tuples), this will always be
      // cached and computed together anyway (e.g. it'll be a variadic reduce).
      TF_ASSIGN_OR_RETURN(auto tuple, operand_provider(instr, 0, indices));
      return {{tuple[instr->tuple_index()]}};
    }
    default:
      break;
  }

  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.reserve(instr->operands().size());
  for (auto operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto operand_element_type,
                        ConvertPrimitiveTypeToMlirType(
                            operand->shape().element_type(), builder));
    arg_types.push_back(operand_element_type);
  }
  auto input_indices = GetInputIndices(
      ComputeOutputToInputIndexing(instr, 0, mlir_context), indices, builder);
  SmallVector<Value> operands;
  for (auto&& [operand_number, operand_indices] :
       llvm::enumerate(input_indices)) {
    TF_ASSIGN_OR_RETURN(operands.emplace_back(),
                        GetSingleOperandValue(operand_provider, instr,
                                              operand_number, operand_indices));
    // Nulls can be pretty hard to debug, so guard against them here. The MHLO
    // conversion functions like to return nullptr for errors.
    TF_RET_CHECK(operands.back() != nullptr)
        << "null operand at index " << operand_number << " for "
        << instr->ToShortString();
  }
  CHECK_NE(operands.size(), 0);

  switch (instr->opcode()) {
    case HloOpcode::kAbs:
      return {MapHloOp<mhlo::AbsOp>(result_element_type, arg_types, operands,
                                    builder)};
    case HloOpcode::kAdd:
      if (element_type == PRED) {
        return MapElementwiseOp<mhlo::OrOp>(arg_types, operands, builder);
      } else {
        return MapElementwiseOp<mhlo::AddOp>(arg_types, operands, builder);
      }
    case HloOpcode::kAnd:
      return MapElementwiseOp<mhlo::AndOp>(arg_types, operands, builder);
    case HloOpcode::kAtan2:
      return MapElementwiseOp<mhlo::Atan2Op>(arg_types, operands, builder);
    case HloOpcode::kCbrt:
      return MapElementwiseOp<mhlo::CbrtOp>(arg_types, operands, builder);
    case HloOpcode::kCeil:
      return MapElementwiseOp<mhlo::CeilOp>(arg_types, operands, builder);
    case HloOpcode::kClamp:
      return MapElementwiseOp<mhlo::ClampOp>(arg_types, operands, builder);
    case HloOpcode::kClz:
      return MapElementwiseOp<mhlo::ClzOp>(arg_types, operands, builder);
    case HloOpcode::kCompare: {
      auto* context = builder.getContext();
      auto direction = mhlo::symbolizeComparisonDirection(
          ComparisonDirectionToString(instr->comparison_direction()));
      mhlo::CompareOp::Properties properties;
      properties.comparison_direction =
          mhlo::ComparisonDirectionAttr::get(context, direction.value());
      auto result_types = llvm::to_vector(mlir::TypeRange{builder.getI1Type()});
      return {{mhlo::MhloOpToStdScalarOp::mapOpOfType<mhlo::CompareOp>(
          builder.getLoc(), result_types, arg_types,
          mhlo::CompareOp::Adaptor(operands, nullptr, properties), &builder)}};
    }
    case HloOpcode::kComplex:
      return MapHloOp<mhlo::ComplexOp>(result_element_type, arg_types, operands,
                                       builder);
    case HloOpcode::kCos:
      return MapElementwiseOp<mhlo::CosineOp>(arg_types, operands, builder);
    case HloOpcode::kDivide:
      return MapElementwiseOp<mhlo::DivOp>(arg_types, operands, builder);
    case HloOpcode::kErf:
      return MapElementwiseOp<mhlo::ErfOp>(arg_types, operands, builder);
    case HloOpcode::kExp:
      return MapElementwiseOp<mhlo::ExpOp>(arg_types, operands, builder);
    case HloOpcode::kExpm1:
      return MapElementwiseOp<mhlo::Expm1Op>(arg_types, operands, builder);
    case HloOpcode::kFloor:
      return MapElementwiseOp<mhlo::FloorOp>(arg_types, operands, builder);
    case HloOpcode::kIsFinite:
      return MapHloOp<mhlo::IsFiniteOp>(builder.getI1Type(), arg_types,
                                        operands, builder);
    case HloOpcode::kImag:
      return MapHloOp<mhlo::ImagOp>(result_element_type, arg_types, operands,
                                    builder);
    case HloOpcode::kLog:
      return MapElementwiseOp<mhlo::LogOp>(arg_types, operands, builder);
    case HloOpcode::kLog1p:
      return MapElementwiseOp<mhlo::Log1pOp>(arg_types, operands, builder);
    case HloOpcode::kLogistic:
      return MapElementwiseOp<mhlo::LogisticOp>(arg_types, operands, builder);
    case HloOpcode::kMap: {
      auto mapper = call_target_provider(
          instr->called_computations().front()->root_instruction());
      return builder.create<PureCallOp>(mapper, operands).getResults();
    }
    case HloOpcode::kMaximum:
      return MapElementwiseOp<mhlo::MaxOp>(arg_types, operands, builder);
    case HloOpcode::kMinimum:
      return MapElementwiseOp<mhlo::MinOp>(arg_types, operands, builder);
    case HloOpcode::kMultiply:
      return MapElementwiseOp<mhlo::MulOp>(arg_types, operands, builder);
    case HloOpcode::kNegate:
      return MapElementwiseOp<mhlo::NegOp>(arg_types, operands, builder);
    case HloOpcode::kNot:
      return MapElementwiseOp<mhlo::NotOp>(arg_types, operands, builder);
    case HloOpcode::kOr:
      return MapElementwiseOp<mhlo::OrOp>(arg_types, operands, builder);
    case HloOpcode::kPopulationCount:
      return MapHloOp<mhlo::PopulationCountOp>(result_element_type, arg_types,
                                               operands, builder);
    case HloOpcode::kPower:
      return MapElementwiseOp<mhlo::PowOp>(arg_types, operands, builder);
    case HloOpcode::kReal:
      return MapHloOp<mhlo::RealOp>(result_element_type, arg_types, operands,
                                    builder);
    case HloOpcode::kReducePrecision: {
      mhlo::ReducePrecisionOp::Properties properties;
      properties.exponent_bits =
          builder.getI32IntegerAttr(instr->exponent_bits());
      properties.mantissa_bits =
          builder.getI32IntegerAttr(instr->mantissa_bits());
      return MapHloOp<mhlo::ReducePrecisionOp>(operands.front().getType(),
                                               arg_types, operands, builder,
                                               nullptr, properties);
    }
    case HloOpcode::kRemainder:
      return MapElementwiseOp<mhlo::RemOp>(arg_types, operands, builder);
    case HloOpcode::kRoundNearestAfz:
      return MapElementwiseOp<mhlo::RoundOp>(arg_types, operands, builder);
    case HloOpcode::kRoundNearestEven:
      return MapElementwiseOp<mhlo::RoundNearestEvenOp>(arg_types, operands,
                                                        builder);
    case HloOpcode::kRsqrt:
      return MapElementwiseOp<mhlo::RsqrtOp>(arg_types, operands, builder);
    case HloOpcode::kSelect:
      return MapElementwiseOp<mhlo::SelectOp>(arg_types, operands, builder);
    case HloOpcode::kShiftLeft:
      return MapElementwiseOp<mhlo::ShiftLeftOp>(arg_types, operands, builder);
    case HloOpcode::kShiftRightArithmetic:
      return MapElementwiseOp<mhlo::ShiftRightArithmeticOp>(arg_types, operands,
                                                            builder);
    case HloOpcode::kShiftRightLogical:
      return MapElementwiseOp<mhlo::ShiftRightLogicalOp>(arg_types, operands,
                                                         builder);
    case HloOpcode::kSign:
      return MapElementwiseOp<mhlo::SignOp>(arg_types, operands, builder);
    case HloOpcode::kSin:
      return MapElementwiseOp<mhlo::SineOp>(arg_types, operands, builder);
    case HloOpcode::kSqrt:
      return MapElementwiseOp<mhlo::SqrtOp>(arg_types, operands, builder);
    case HloOpcode::kSubtract:
      return MapElementwiseOp<mhlo::SubtractOp>(arg_types, operands, builder);
    case HloOpcode::kTan:
      return MapElementwiseOp<mhlo::TanOp>(arg_types, operands, builder);
    case HloOpcode::kTanh:
      return MapElementwiseOp<mhlo::TanhOp>(arg_types, operands, builder);
    case HloOpcode::kXor:
      return MapElementwiseOp<mhlo::XorOp>(arg_types, operands, builder);
    case HloOpcode::kBitcastConvert:
      return MapHloOp<mhlo::BitcastConvertOp>(result_element_type, arg_types,
                                              operands, builder);
    case HloOpcode::kConvert: {
      if (operands[0].getType().isa<mlir::FloatType>() &&
          element_type == PRED) {
        return {builder
                    .create<mlir::arith::CmpFOp>(
                        mlir::arith::CmpFPredicate::UNE, operands[0],
                        builder.create<ConstantOp>(
                            builder.getFloatAttr(operands[0].getType(), 0.0)))
                    ->getResults()};
      }
      auto out = mhlo::MhloOpToStdScalarOp::mapConvertOpToStdScalarOp(
          builder.getLoc(), result_type_with_sign, result_element_type,
          arg_types, operands, &builder);
      if (auto int_ty = out.getType().dyn_cast<mlir::IntegerType>()) {
        auto in = operands[0];
        if (auto float_ty = in.getType().dyn_cast<mlir::FloatType>()) {
          auto cst_int = [&](int64_t x) {
            return builder.create<arith::ConstantIntOp>(x, int_ty);
          };
          if (primitive_util::IsUnsignedIntegralType(element_type)) {
            auto cst_float = [&](uint64_t x) {
              return builder.create<ConstantOp>(
                  builder.getFloatAttr(float_ty, x));
            };
            int64_t min = 0;
            int64_t max = llvm::maxUIntN(int_ty.getWidth());
            // x <= 0 || isnan(x) ? 0 : ...
            out = builder.create<mlir::arith::SelectOp>(
                builder.create<mlir::arith::CmpFOp>(
                    mlir::arith::CmpFPredicate::ULE, in, cst_float(min)),
                cst_int(min), out);
            // x >= static_cast<float>(UINT_MAX) ? UINT_MAX : ...
            out = builder.create<mlir::arith::SelectOp>(
                builder.create<mlir::arith::CmpFOp>(
                    mlir::arith::CmpFPredicate::OGE, in, cst_float(max)),
                cst_int(max), out);
          } else {
            auto cst_float = [&](int64_t x) {
              return builder.create<ConstantOp>(
                  builder.getFloatAttr(float_ty, x));
            };
            int64_t min = llvm::minIntN(int_ty.getWidth());
            int64_t max = llvm::maxIntN(int_ty.getWidth());
            // x <= static_cast<float>(INT_MIN) ? INT_MIN : ...
            out = builder.create<mlir::arith::SelectOp>(
                builder.create<mlir::arith::CmpFOp>(
                    mlir::arith::CmpFPredicate::OLE, in, cst_float(min)),
                cst_int(min), out);
            // x >= static_cast<float>(INT_MAX) ? INT_MAX : ...
            out = builder.create<mlir::arith::SelectOp>(
                builder.create<mlir::arith::CmpFOp>(
                    mlir::arith::CmpFPredicate::OGE, in, cst_float(max)),
                cst_int(max), out);
            // isnan(x) ? 0 : ...
            out = builder.create<mlir::arith::SelectOp>(
                builder.create<mlir::arith::CmpFOp>(
                    mlir::arith::CmpFPredicate::UNO, in, in),
                cst_int(0), out);
          }
        }
      }
      return {{out}};
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kCopy:
    case HloOpcode::kSlice:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      return operands;
    default:
      break;
  }

  return absl::UnimplementedError(absl::StrCat("Unsupported: ", instr->name()));
}

}  // namespace

bool IsHloOpSupported(const HloInstruction* instr,
                      se::CudaComputeCapability compute_capability) {
  return !(kUnsupportedOps.contains(instr->opcode()) ||
           IsUnsupportedGather(instr));
}

bool IsHloConversionSupported(const HloComputation* computation,
                              se::GpuComputeCapability compute_capability) {
  if (!std::holds_alternative<se::CudaComputeCapability>(compute_capability)) {
    // ROCM is not tested.
    return false;
  }
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(compute_capability);

  return absl::c_all_of(
             computation->instructions(),
             [=](const HloInstruction* instr) {
               return absl::c_all_of(instr->called_computations(),
                                     [&](const HloComputation* called) {
                                       return IsHloConversionSupported(
                                           called, compute_capability);
                                     }) &&
                      IsHloOpSupported(instr, cuda_compute_capability);
             }) &&
         (computation->IsFusionComputation() ||
          (absl::c_all_of(
              computation->parameter_instructions(), [](auto* param) {
                return param->shape().IsArray() && param->shape().rank() == 0;
              })));
}

bool IsHloConversionSupported(const HloFusionAdaptor& fusion,
                              se::GpuComputeCapability compute_capability) {
  if (!std::holds_alternative<se::CudaComputeCapability>(compute_capability)) {
    // ROCM is not tested.
    return false;
  }
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(compute_capability);

  return !HloFindIf(
      fusion.GetRoots(), fusion, [=](HloInstructionAdaptor instr) {
        return !absl::c_all_of(instr.instruction().called_computations(),
                               [&](const HloComputation* called) {
                                 return IsHloConversionSupported(
                                     called, compute_capability);
                               }) ||
               !IsHloOpSupported(&instr.instruction(), cuda_compute_capability);
      });
}

llvm::SmallVector<Value> ProvideParameter(
    const PartitionedComputation& computation, const HloInstruction* instr,
    int operand_index, ValueRange indices,
    const CallTargetProvider& call_target_provider, mlir::func::FuncOp this_fn,
    ImplicitLocOpBuilder& builder,
    const PartitionedComputation::Subgraph* caller) {
  auto* operand = instr->operand(operand_index);

  if (!caller) {
    caller = &computation.FindSubgraph(instr);
  }
  const auto& injected_value_starts = caller->injected_value_starts;
  if (auto it = injected_value_starts.find(operand);
      it != injected_value_starts.end()) {
    return {this_fn.getArguments().take_back(
        caller->num_injected_values)[it->second]};
  }

  auto callee = call_target_provider(operand);
  SmallVector<Value> operands(
      this_fn.getArguments().take_front(instr->parent()->num_parameters()));
  absl::c_copy(indices, std::back_inserter(operands));
  auto results = builder.create<PureCallOp>(callee, operands).getResults();
  auto callee_subgraph = computation.FindSubgraph(operand);
  if (callee_subgraph.roots.size() == 1) {
    CHECK_EQ(callee_subgraph.roots.front(), operand)
        << "Expected " << operand->ToString() << " to be the root of "
        << callee_subgraph.ToString();
    return results;
  }

  int offset = 0;
  for (auto root : callee_subgraph.roots) {
    int root_arity =
        root->shape().IsTuple() ? root->shape().tuple_shapes_size() : 1;
    if (root == operand) {
      return results.slice(offset, root_arity);
    }
    offset += root_arity;
  }
  LOG(FATAL) << "Did not find operand " << operand->name() << " in roots of "
             << callee_subgraph.ToString();
}

SmallVector<Value> ProvideParameterRange(
    const PartitionedComputation& computation, const HloInstruction* instr,
    int start, int num, ValueRange indices,
    const CallTargetProvider& call_target_provider, mlir::func::FuncOp this_fn,
    ImplicitLocOpBuilder& builder) {
  SmallVector<Value> scalars;
  scalars.reserve(num);
  for (int i = 0; i < num; ++i) {
    scalars.append(ProvideParameter(computation, instr, i + start, indices,
                                    call_target_provider, this_fn, builder));
  }
  return scalars;
}

namespace {

absl::StatusOr<SmallVector<Value>> SubgraphToMlir(
    const PartitionedComputation& computation,
    const PartitionedComputation::Subgraph& subgraph,
    mlir::func::FuncOp this_fn, const CallTargetProvider& call_target_provider,
    ValueRange parameters, ValueRange indices, ImplicitLocOpBuilder& builder) {
  SmallVector<Value> results;
  absl::node_hash_map<std::pair<const HloInstruction*, std::vector<void*>>,
                      SmallVector<Value>>
      cached_instructions;

  std::function<absl::StatusOr<SmallVector<Value>>(const HloInstruction* instr,
                                                   ValueRange indices)>
      emit_instr;

  auto provide_operand =
      [&](const HloInstruction* instr, int index,
          ValueRange operand_indices) -> absl::StatusOr<SmallVector<Value>> {
    auto* operand = instr->operand(index);
    if (subgraph.instructions.contains(operand)) {
      return emit_instr(operand, operand_indices);
    }
    return ProvideParameter(computation, instr, index, operand_indices,
                            call_target_provider, this_fn, builder, &subgraph);
  };

  emit_instr = [&](const HloInstruction* instr,
                   ValueRange indices) -> absl::StatusOr<SmallVector<Value>> {
    std::vector<void*> indices_ptrs;
    indices_ptrs.reserve(indices.size());
    for (auto index : indices) {
      indices_ptrs.push_back(index.getAsOpaquePointer());
    }
    auto& entry = cached_instructions[std::make_pair(instr, indices_ptrs)];
    // Only use the entry if its parent block is still in scope. Note that this
    // should always be the case normally - if not, we risk exponential code
    // size.
    if (!entry.empty()) {
      auto* entry_block = entry.front().getParentBlock();
      auto* insertion_block = builder.getInsertionBlock();
      while (insertion_block != nullptr) {
        if (insertion_block == entry_block) return entry;
        if (insertion_block->getParentOp()) {
          insertion_block = insertion_block->getParentOp()->getBlock();
        } else {
          insertion_block = nullptr;
          VLOG(2) << "Failed dominance check while looking up cache for "
                  << instr->ToShortString()
                  << ". This is a bug in the computation partitioner.";
        }
      }
    }

    TF_ASSIGN_OR_RETURN(auto lowered_instr,
                        HloToMlir(instr, this_fn, indices, provide_operand,
                                  call_target_provider, builder));
    CHECK(!absl::c_linear_search(lowered_instr, nullptr))
        << "Failed to lower " << instr->name();

    entry = lowered_instr;
    TF_RET_CHECK(!absl::c_any_of(
        entry, [](const auto& entry) { return entry == nullptr; }))
        << "null result for " << instr->ToShortString();
    return entry;
  };

  TF_RET_CHECK(subgraph.roots.size() == subgraph.root_indexing.size())
      << "roots and root_indexing must have the same size in "
      << subgraph.ToString();
  for (const auto [root, indexing] :
       llvm::zip(subgraph.roots, subgraph.root_indexing)) {
    if (auto it = subgraph.injected_value_starts.find(root);
        it != subgraph.injected_value_starts.end()) {
      auto injected =
          this_fn.getArguments().take_back(subgraph.num_injected_values);
      int arity =
          root->shape().IsTuple() ? root->shape().tuple_shapes_size() : 1;
      absl::c_copy(injected.slice(it->second, arity),
                   std::back_inserter(results));
      continue;
    }
    int num_dims = indexing.GetAffineMap().getNumDims();
    auto root_indices =
        ApplyIndexing(indexing, /*dims=*/indices.take_front(num_dims),
                      /*symbols=*/indices.drop_front(num_dims), builder);
    TF_ASSIGN_OR_RETURN(auto root_results, emit_instr(root, root_indices));
    results.append(root_results.begin(), root_results.end());
  }
  return results;
}

void GetLoopBoundsFromIndexingMap(ImplicitLocOpBuilder& b,
                                  const IndexingMap& indexing_map,
                                  SmallVectorImpl<Value>* lbs,
                                  SmallVectorImpl<Value>* ubs,
                                  SmallVectorImpl<Value>* steps) {
  Value c1 = b.create<ConstantIndexOp>(1);

  for (const Interval& bound : indexing_map.GetSymbolBounds()) {
    lbs->push_back(b.create<ConstantIndexOp>(bound.lower));
    ubs->push_back(b.create<ConstantIndexOp>(bound.upper + 1));
    steps->push_back(c1);
  }
}

}  // namespace

absl::Status SubgraphToMlirFunction(
    const PartitionedComputation& computation,
    const PartitionedComputation::Subgraph& subgraph, mlir::func::FuncOp& func,
    const CallTargetProvider& call_target_provider) {
  TF_RET_CHECK(func != nullptr);
  ImplicitLocOpBuilder builder(func.getLoc(), func->getContext());
  builder.setInsertionPointToStart(func.addEntryBlock());
  auto parameters = func.getArguments().take_front(
      computation.computation().num_parameters());
  auto indices_and_injected_values = func.getArguments().drop_front(
      computation.computation().num_parameters());
  auto indices =
      indices_and_injected_values.drop_back(subgraph.num_injected_values);
  TF_ASSIGN_OR_RETURN(
      auto results,
      SubgraphToMlir(computation, subgraph, func, call_target_provider,
                     parameters, indices, builder));
  CHECK_EQ(results.size(), func.getResultTypes().size());

  builder.create<mlir::func::ReturnOp>(results);
  return absl::OkStatus();
}

namespace {

bool IsSymbolConstrained(const IndexingMap& map, int symbol_id) {
  for (const auto& [expr, _] : map.GetConstraints()) {
    bool result = false;
    expr.walk([&](mlir::AffineExpr leaf) {
      auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(leaf);
      if (sym && sym.getPosition() == symbol_id) {
        result = true;
      }
    });
    if (result) return true;
  }
  return false;
}

SmallVector<Value> EmitLoopNestImpl(
    ImplicitLocOpBuilder& b, ValueRange dim_values, ValueRange iter_args_inits,
    const IndexingMap& indexing_map,
    mlir::function_ref<SmallVector<Value>(ValueRange /*iter_args*/,
                                          ValueRange /*dim_values*/,
                                          ValueRange /*symbol_values*/)>
        create_body,
    bool vectorize) {
  SmallVector<Value, 4> lbs, ubs, steps;
  GetLoopBoundsFromIndexingMap(b, indexing_map, &lbs, &ubs, &steps);

  SmallVector<Value, 4> vector_inits;
  if (vectorize) {
    CHECK_EQ(indexing_map.GetSymbolBounds().back().lower, 0);
    int vector_size = indexing_map.GetSymbolBounds().back().upper + 1;
    vector_inits = iter_args_inits;
    for (auto& init : vector_inits) {
      if (!mlir::isa<mlir::ShapedType>(init.getType())) {
        auto vector_ty = mlir::VectorType::get({vector_size}, init.getType());
        init = b.create<mlir::vector::SplatOp>(vector_ty, init);
      }
    }
    iter_args_inits = vector_inits;
  }

  auto bb = [&](OpBuilder& nested_builder, Location loc,
                ValueRange symbol_values,
                ValueRange iter_args) -> scf::ValueVector {
    ImplicitLocOpBuilder nested_b(loc, nested_builder);
    auto is_in_bounds = mlir_converter::CheckConstraints(
        indexing_map, dim_values, symbol_values, nested_b);
    auto if_op = nested_b.create<scf::IfOp>(
        is_in_bounds,
        [&](OpBuilder& then_builder, Location then_loc) -> void {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(then_builder.getInsertionBlock());
          SmallVector<Value, 4> results;
          if (vectorize) {
            SmallVector<Value, 4> vector_args;
            vector_args = iter_args;
            // Extract the vector elements.
            for (auto& init : vector_args) {
              if (mlir::isa<mlir::VectorType>(init.getType())) {
                init = b.create<mlir::vector::ExtractOp>(init,
                                                         symbol_values.back());
              }
            }
            results = create_body(vector_args, dim_values, symbol_values);
            // Insert the results.
            for (auto [index, init] : llvm::enumerate(iter_args)) {
              if (mlir::isa<mlir::VectorType>(init.getType())) {
                results[index] = b.create<mlir::vector::InsertOp>(
                    results[index], iter_args[index], symbol_values.back());
              }
            }
          } else {
            results = create_body(iter_args, dim_values, symbol_values);
          }
          b.create<scf::YieldOp>(results);
        },
        [&](OpBuilder& else_b, Location else_loc) {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(else_b.getInsertionBlock());
          b.create<scf::YieldOp>(iter_args);
        });

    return if_op.getResults();
  };
  scf::LoopNest loop_nest =
      scf::buildLoopNest(b, b.getLoc(), lbs, ubs, steps, iter_args_inits, bb);
  return loop_nest.results;
}

}  // namespace

SmallVector<Value> EmitLoopNest(
    ImplicitLocOpBuilder& b, ValueRange dim_values, ValueRange iter_args_inits,
    const IndexingMap& indexing_map,
    mlir::function_ref<SmallVector<Value>(ValueRange /*iter_args*/,
                                          ValueRange /*dim_values*/,
                                          ValueRange /*symbol_values*/)>
        create_body,
    bool vectorize) {
  // TODO(b/343420432): Add an op that represents a constrained loop nest and
  // peel in a pass, instead of doing it ad hoc here.
  int64_t cumulative_loop_size = 1;
  for (int sym_index = indexing_map.GetSymbolCount() - 1;
       sym_index >= 0 && cumulative_loop_size < 64; --sym_index) {
    auto& bound = indexing_map.GetSymbolBound(sym_index);
    cumulative_loop_size *= bound.GetLoopTripCount();
    if (!IsSymbolConstrained(indexing_map, sym_index)) continue;

    IndexingMap peeled_map = indexing_map;
    if (bound.upper == bound.lower) continue;

    --peeled_map.GetMutableSymbolBound(sym_index).upper;
    peeled_map.Simplify();

    // If the symbol is still constrained, peeling does not help.
    if (IsSymbolConstrained(peeled_map, sym_index)) continue;

    auto first_results = EmitLoopNestImpl(b, dim_values, iter_args_inits,
                                          peeled_map, create_body, vectorize);

    IndexingMap remainder = indexing_map;
    remainder.GetMutableSymbolBound(sym_index).lower = bound.upper;
    remainder.Simplify();

    VLOG(5) << "Peeled indexing map " << indexing_map.ToString() << "\n into "
            << peeled_map.ToString() << "\nand remainder\n"
            << remainder.ToString();
    return EmitLoopNestImpl(b, dim_values, first_results, remainder,
                            create_body, vectorize);
  }

  return EmitLoopNestImpl(b, dim_values, iter_args_inits, indexing_map,
                          create_body, vectorize);
}

absl::StatusOr<SmallVector<Value>> EmitLoopNestWithStatus(
    ImplicitLocOpBuilder& b, ValueRange dim_values, ValueRange iter_args_inits,
    const IndexingMap& indexing_map,
    mlir::function_ref<absl::StatusOr<SmallVector<Value>>(
        ValueRange /*iter_args*/, ValueRange /*dim_values*/,
        ValueRange /*symbol_values*/)>
        create_body) {
  absl::Status status = absl::OkStatus();

  auto result = EmitLoopNest(
      b, dim_values, iter_args_inits, indexing_map,
      [&](ValueRange iter_args, ValueRange dim_values,
          ValueRange symbol_values) -> SmallVector<Value> {
        auto body_result = create_body(iter_args, dim_values, symbol_values);
        if (!body_result.ok()) {
          status = std::move(body_result.status());
          return SmallVector<Value>{};
        }

        return std::move(body_result.value());
      });

  if (!status.ok()) {
    return status;
  }
  return result;
}

mlir::Value ClampIndex(mlir::Value index, bool is_unsigned, int64_t high,
                       ImplicitLocOpBuilder& b) {
  auto zero = b.create<ConstantOp>(b.getIndexAttr(0));
  if (high <= 0) {
    return zero;
  }

  if (is_unsigned) {
    if (index.getType() != b.getIndexType()) {
      index = b.create<arith::IndexCastUIOp>(b.getIndexType(), index);
    }
    index = b.create<arith::MinUIOp>(
        index, b.create<ConstantOp>(b.getIndexAttr(high)));
  } else {
    if (index.getType() != b.getIndexType()) {
      index = b.create<arith::IndexCastOp>(b.getIndexType(), index);
    }
    index = b.create<arith::MinSIOp>(
        index, b.create<ConstantOp>(b.getIndexAttr(high)));
    index = b.create<arith::MaxSIOp>(index, zero);
  }
  return index;
}

SmallVector<Value, 2> InlineBlock(OpBuilder& builder, Block& src_block,
                                  ValueRange mapped_args) {
  IRMapping mapping;
  for (auto [from, to] : llvm::zip(src_block.getArguments(), mapped_args)) {
    mapping.map(from, to);
  }
  for (auto& op : src_block.without_terminator()) {
    builder.clone(op, mapping);
  }
  auto* terminator = src_block.getTerminator();
  SmallVector<Value, 2> mapped_results;

  mapped_results.reserve(terminator->getResults().size());
  for (mlir::Value result : src_block.getTerminator()->getOperands()) {
    mapped_results.push_back(mapping.lookup(result));
  }
  return mapped_results;
}

}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
