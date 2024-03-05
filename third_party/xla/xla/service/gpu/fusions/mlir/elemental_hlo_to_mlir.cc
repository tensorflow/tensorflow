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

#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
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
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/mlir_hlo/mhlo/utils/type_conversion.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
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
using mlir::ImplicitLocOpBuilder;
using mlir::Location;
using mlir::OpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::AndIOp;
using mlir::arith::CmpFOp;
using mlir::arith::CmpFPredicate;
using mlir::arith::CmpIOp;
using mlir::arith::CmpIPredicate;
using mlir::arith::ConstantIndexOp;
using mlir::arith::ConstantOp;
using mlir::arith::SelectOp;
using mlir::scf::ForOp;
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

static auto& kUnimplementedOps = *new absl::flat_hash_set<HloOpcode>{
    HloOpcode::kConvolution, HloOpcode::kDot, HloOpcode::kMap,
    HloOpcode::kReduceWindow};

bool IsUnsupportedConstant(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kConstant &&
         (!ShapeUtil::IsEffectiveScalar(instr->shape()) ||
          primitive_util::IsUnsignedIntegralType(
              instr->shape().element_type()) ||
          primitive_util::IsComplexType(instr->shape().element_type()));
}

bool IsUnsupportedTuple(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kTuple) {
    return false;
  }

  if (instr->user_count() > 0) {
    // Internal tuples are unsupported.
    return true;
  }

  // Nested tuples and tokens are unsupported.
  if (absl::c_any_of(instr->operands(),
                     [&](auto* op) { return !op->shape().IsArray(); })) {
    return true;
  }

  // All tuple elements must have the same dimensions (element types may
  // differ).
  auto first_shape = instr->shape().tuple_shapes(0);
  for (int i = 1; i < instr->operand_count(); ++i) {
    if (instr->shape().tuple_shapes(i).dimensions() !=
        first_shape.dimensions()) {
      return true;
    }
  }
  return false;
}

bool IsUnsupportedGather(const HloInstruction* instr) {
  // We assume gather simplifier ran, so we don't need to support all gather
  // forms.
  if (instr->opcode() != HloOpcode::kGather) return false;

  auto* gather = Cast<HloGatherInstruction>(instr);
  const auto& dims = gather->gather_dimension_numbers();
  if (dims.index_vector_dim() != 1 || !dims.collapsed_slice_dims().empty() ||
      gather->operand(1)->shape().rank() != 2) {
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
  SmallVector<Value> reduction_indices(indices);
  SmallVector<Value> accumulators;
  for (int i = instr->operand_count() / 2; i < instr->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(accumulators.emplace_back(),
                        GetSingleOperandValue(operand_provider, instr, i, {}));
  }
  auto dims = llvm::to_vector(instr->dimensions());
  absl::c_sort(dims);
  ForOp outermost_loop = nullptr;
  for (int dim : dims) {
    auto bound = instr->operands()[0]->shape().dimensions(dim);
    auto loop = b.create<ForOp>(b.create<ConstantIndexOp>(0),
                                b.create<ConstantIndexOp>(bound),
                                b.create<ConstantIndexOp>(1), accumulators);
    if (outermost_loop == nullptr) {
      outermost_loop = loop;
    } else {
      b.create<YieldOp>(loop.getResults());
    }
    b.setInsertionPointToStart(loop.getBody());
    reduction_indices.insert(reduction_indices.begin() + dim,
                             loop.getInductionVar());
    accumulators = {loop.getRegionIterArgs().begin(),
                    loop.getRegionIterArgs().end()};
  }
  SmallVector<Value> args = accumulators;
  for (int i = 0; i < instr->operand_count() / 2; ++i) {
    TF_ASSIGN_OR_RETURN(
        args.emplace_back(),
        GetSingleOperandValue(operand_provider, instr, i, reduction_indices));
  }
  auto reducer = call_target_provider(
      instr->called_computations().front()->root_instruction());
  b.create<YieldOp>(b.create<mlir::func::CallOp>(reducer, args).getResults());

  b.setInsertionPointAfter(outermost_loop);
  return outermost_loop.getResults();
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
    auto in_bounds = b.create<CmpIOp>(CmpIPredicate::ult, indices[concat_dim],
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
      auto if_op = b.create<IfOp>(mlir::TypeRange{result_element_type},
                                  in_bounds, true, true);
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
  int num_indices = instr->operand(1)->shape().dimensions(1);
  for (int i = 0; i < num_indices; ++i) {
    auto i_val = i == 0 ? zero : b.create<ConstantIndexOp>(i);
    int64_t slice_size = instr->gather_slice_sizes()[i];
    int64_t input_size = instr->operand(0)->shape().dimensions()[i];
    // Read and clamp index.
    TF_ASSIGN_OR_RETURN(auto input_index,
                        operand_provider(instr, 1, {row, i_val}));
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
    auto map = maps.begin()->GetAffineMap();
    CHECK(!maps.begin()->IsUndefined());
    indices.emplace_back() = ApplyAffineMap(map, output_indices, {}, b);
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

absl::StatusOr<SmallVector<Value>> EmitParameter(
    const HloInstruction* instr, ValueRange indices,
    const CallTargetProvider& call_target_provider, ImplicitLocOpBuilder& b) {
  auto this_fn = call_target_provider(instr);

  Value value = this_fn.getArgument(instr->parameter_number());
  if (value.getType().isa<mlir::TensorType>()) {
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

SmallVector<Value> ApplyAffineMap(mlir::AffineMap map, ValueRange dims,
                                  ValueRange symbols, ImplicitLocOpBuilder& b) {
  SmallVector<Value> result;
  result.reserve(map.getNumResults());
  for (auto expr : map.getResults()) {
    result.push_back(ApplyAffineExpr(expr, dims, symbols, b));
  }
  return result;
}

Value CheckConstraint(mlir::Value constrained_value, Range range,
                      ImplicitLocOpBuilder& b) {
  auto lb = b.create<ConstantOp>(b.getIndexAttr(range.lower_bound));
  if (range.IsPoint()) {
    return b.create<CmpIOp>(CmpIPredicate::eq, constrained_value, lb);
  }
  auto ub = b.create<ConstantOp>(b.getIndexAttr(range.upper_bound));
  return b.create<AndIOp>(
      b.create<CmpIOp>(CmpIPredicate::sge, constrained_value, lb),
      b.create<CmpIOp>(CmpIPredicate::sle, constrained_value, ub));
}

Value CheckConstraints(const IndexingMap& map, ValueRange dims,
                       ValueRange symbols, ImplicitLocOpBuilder& b) {
  Value ret = b.create<ConstantOp>(b.getIntegerAttr(b.getI1Type(), 1));
  for (auto&& [expression, range] : map.GetConstraints()) {
    ret = b.create<AndIOp>(
        ret, CheckConstraint(ApplyAffineExpr(expression, dims, symbols, b),
                             range, b));
  }
  for (auto&& [index, range] : llvm::enumerate(map.GetDimensionRanges())) {
    ret = b.create<AndIOp>(ret, CheckConstraint(dims[index], range, b));
  }
  return ret;
}

absl::StatusOr<SmallVector<Value>> HloToMlir(
    const HloInstruction* instr, ValueRange indices,
    const OperandProvider& operand_provider,
    const CallTargetProvider& call_target_provider,
    ImplicitLocOpBuilder& builder) {
  CHECK(!kUnsupportedOps.contains(instr->opcode())) << instr->ToShortString();
  CHECK(!kUnimplementedOps.contains(instr->opcode())) << instr->ToShortString();

  auto element_type = instr->shape().element_type();
  mlir::Type element_mlir_type;
  mlir::Type result_element_type;
  if (!instr->shape().IsTuple()) {
    TF_ASSIGN_OR_RETURN(element_mlir_type,
                        ConvertPrimitiveTypeToMLIRType(element_type, builder));

    // During mapping to the arith dialect, we need to convert from signed
    // integer types to signless integer types. Most mappings can infer the
    // signless integer type from the already converted operand, but e.g. for
    // Convert this is not possible, so we need to have the signless result
    // element type as well. But we also still need to pass the signed integer
    // element type, as that is needed to select the correct arith ops for
    // unsigned element types.
    mlir::mhlo::RemoveSignTypeConverter sign_converter;
    result_element_type = sign_converter.convertType(element_mlir_type);
  }

  // Handle ops that aren't elementwise and aren't just indexing
  // transformations.
  switch (instr->opcode()) {
    case HloOpcode::kConcatenate:
      return EmitConcat(instr, result_element_type, indices, operand_provider,
                        builder);
    case HloOpcode::kConstant:
      if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
        auto val = mlir::cast<mlir::TypedAttr>(
            CreateDenseElementsAttrFromLiteral(instr->literal(), builder)
                ->getValues<mlir::Attribute>()[0]);
        return {{builder.create<ConstantOp>(val).getResult()}};
      }
      return absl::UnimplementedError(
          absl::StrCat("Unimplemented: ", instr->ToShortString()));
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
      return MapHloOp<mhlo::ConvertOp>(result_element_type, {index_type},
                                       {index}, builder);
    }
    case HloOpcode::kPad:
      return EmitPad(instr, result_element_type, indices, operand_provider,
                     builder);
    case HloOpcode::kParameter:
      return EmitParameter(instr, indices, call_target_provider, builder);
    case HloOpcode::kReduce:
      return EmitReduce(instr, indices, operand_provider, call_target_provider,
                        builder);
    case HloOpcode::kTuple: {
      CHECK(!IsUnsupportedTuple(instr));
      SmallVector<Value> operands;
      for (int i = 0; i < instr->operand_count(); ++i) {
        TF_ASSIGN_OR_RETURN(
            operands.emplace_back(),
            GetSingleOperandValue(operand_provider, instr, i, indices));
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
                        ConvertPrimitiveTypeToMLIRType(
                            operand->shape().element_type(), builder));
    arg_types.push_back(operand_element_type);
  }
  auto input_indices = GetInputIndices(
      ComputeOutputToInputIndexing(instr, 0, builder.getContext()), indices,
      builder);
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
        return {MapHloOp<mhlo::AbsOp>(element_mlir_type, arg_types, operands,
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
      auto dir = builder.getDictionaryAttr(builder.getNamedAttr(
          "comparison_direction",
          mhlo::ComparisonDirectionAttr::get(
              context,
              mhlo::symbolizeComparisonDirection(
                  ComparisonDirectionToString(instr->comparison_direction()))
                  .value())));
      auto result_types = llvm::to_vector(mlir::TypeRange{builder.getI1Type()});
      return {{mhlo::MhloOpToStdScalarOp::mapOpOfType<mhlo::CompareOp>(
          builder.getLoc(), result_types, arg_types,
          mhlo::CompareOp::Adaptor(operands, dir), &builder)}};
    }
    case HloOpcode::kComplex:
      return MapHloOp<mhlo::ComplexOp>(element_mlir_type, arg_types, operands,
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
      return MapHloOp<mhlo::ImagOp>(element_mlir_type, arg_types, operands,
                                    builder);
    case HloOpcode::kLog:
      return MapElementwiseOp<mhlo::LogOp>(arg_types, operands, builder);
    case HloOpcode::kLog1p:
      return MapElementwiseOp<mhlo::Log1pOp>(arg_types, operands, builder);
    case HloOpcode::kLogistic:
      return MapElementwiseOp<mhlo::LogisticOp>(arg_types, operands, builder);
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
      return MapHloOp<mhlo::PopulationCountOp>(element_mlir_type, arg_types,
                                               operands, builder);
    case HloOpcode::kPower:
      return MapElementwiseOp<mhlo::PowOp>(arg_types, operands, builder);
    case HloOpcode::kReal:
      return MapHloOp<mhlo::RealOp>(element_mlir_type, arg_types, operands,
                                    builder);
    case HloOpcode::kReducePrecision: {
      mlir::NamedAttribute exponent_bits(
          builder.getStringAttr("exponent_bits"),
          builder.getI32IntegerAttr(instr->exponent_bits()));
      mlir::NamedAttribute mantissa_bits(
          builder.getStringAttr("mantissa_bits"),
          builder.getI32IntegerAttr(instr->mantissa_bits()));
      return MapHloOp<mhlo::ReducePrecisionOp>(
          operands.front().getType(), arg_types, operands, builder,
          mlir::DictionaryAttr::get(builder.getContext(),
                                    {exponent_bits, mantissa_bits}));
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
      return {{mhlo::MhloOpToStdScalarOp::mapConvertOpToStdScalarOp(
          builder.getLoc(), element_mlir_type, result_element_type, arg_types,
          operands, &builder)}};
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

bool IsHloOpSupported(const HloInstruction* instr,
                      se::CudaComputeCapability compute_capability) {
  auto is_unsupported_type = [](const HloInstruction* instr) {
    auto e = instr->shape().element_type();
    // TODO(akuegel): Fix remaining issues with complex.
    // TODO(jreiffers): Support fp8.
    // TODO(jreiffers): Support int4.
    return (primitive_util::IsIntegralType(e) &&
            primitive_util::BitWidth(e) > 1 &&
            primitive_util::BitWidth(e) < 8) ||
           primitive_util::IsComplexType(e) ||
           (primitive_util::IsFloatingPointType(e) &&
            primitive_util::BitWidth(e) < 16);
  };
  if (is_unsupported_type(instr) ||
      absl::c_any_of(instr->operands(), is_unsupported_type)) {
    return false;
  }

  return !(kUnsupportedOps.contains(instr->opcode()) ||
           kUnimplementedOps.contains(instr->opcode()) ||
           IsUnsupportedConstant(instr) || IsUnsupportedTuple(instr) ||
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

  if (fusion.GetRoots().size() > 1) {
    auto first_shape = fusion.GetRoots()[0].instruction().shape();
    for (int i = 1; i < fusion.GetRoots().size(); ++i) {
      if (fusion.GetRoots()[i].instruction().shape().dimensions() !=
          first_shape.dimensions()) {
        return false;
      }
    }
  }

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

SmallVector<Value> ProvideParameter(
    const PartitionedComputation& computation, const HloInstruction* instr,
    int operand_index, ValueRange indices,
    const CallTargetProvider& call_target_provider,
    ImplicitLocOpBuilder& builder) {
  auto& caller_subgraph = computation.FindSubgraph(instr);
  auto this_fn = call_target_provider(caller_subgraph.roots[0]);

  auto* operand = instr->operand(operand_index);

  const auto& injected_params = caller_subgraph.injected_param_indices;
  if (auto it = injected_params.find(std::make_pair(instr, operand_index));
      it != injected_params.end()) {
    auto injected_param_values =
        this_fn.getArguments().take_back(injected_params.size());
    return {{injected_param_values[it->second]}};
  }

  auto callee = call_target_provider(operand);
  SmallVector<Value> operands(
      this_fn.getArguments().take_front(instr->parent()->num_parameters()));
  absl::c_copy(indices, std::back_inserter(operands));
  return builder.create<PureCallOp>(callee, operands).getResults();
}

SmallVector<Value> ProvideParameterRange(
    const PartitionedComputation& computation, const HloInstruction* instr,
    int start, int num, ValueRange indices,
    const CallTargetProvider& call_target_provider,
    ImplicitLocOpBuilder& builder) {
  SmallVector<Value> scalars;
  for (int i = 0; i < num; ++i) {
    auto scalar = ProvideParameter(computation, instr, i + start, indices,
                                   call_target_provider, builder);
    CHECK_EQ(scalar.size(), 1);
    scalars.push_back(scalar.front());
  }
  return scalars;
}

namespace {

absl::StatusOr<SmallVector<Value>> SubgraphToMlir(
    const PartitionedComputation& computation,
    const PartitionedComputation::Subgraph& subgraph,
    const CallTargetProvider& call_target_provider, ValueRange parameters,
    ValueRange indices, ValueRange injected_param_values,
    ImplicitLocOpBuilder& builder) {
  SmallVector<Value> results;
  absl::node_hash_map<std::pair<const HloInstruction*, std::vector<void*>>,
                      SmallVector<Value>>
      cached_instructions;

  std::function<absl::StatusOr<SmallVector<Value>>(const HloInstruction* instr,
                                                   ValueRange indices)>
      emit_instr;

  auto provide_operand =
      [&](const HloInstruction* instr, int index,
          ValueRange indices) -> absl::StatusOr<SmallVector<Value>> {
    auto* operand = instr->operand(index);
    if (&computation.FindSubgraph(operand) == &subgraph) {
      return emit_instr(operand, indices);
    }
    auto results = ProvideParameter(computation, instr, index, indices,
                                    call_target_provider, builder);
    // Convert from signed to signless.
    mlir::mhlo::RemoveSignTypeConverter sign_converter;
    for (auto& result : results) {
      auto signless_type = sign_converter.convertType(result.getType());
      result =
          builder
              .create<mlir::UnrealizedConversionCastOp>(signless_type, result)
              .getResult(0);
    }
    return results;
  };

  emit_instr = [&](const HloInstruction* instr,
                   ValueRange indices) -> absl::StatusOr<SmallVector<Value>> {
    // TODO(jreiffers): Check dominance, e.g.:
    //
    // padding_value = log(param)
    // pad = pad(bar, padding_value)
    // broadcast = broadcast(padding_value)
    // pad + broadcasub
    //
    // If padding_value was first emitted in the context of pad, it'll be
    // inside an scf.if. For now this doesn't matter, because the indexing
    // is considered to be different, but once the partitioner is smarter,
    // it will matter.
    //
    // Also, this caching should be combined with parameter caching.
    std::vector<void*> indices_ptrs;
    indices_ptrs.reserve(indices.size());
    for (auto index : indices) {
      indices_ptrs.push_back(index.getAsOpaquePointer());
    }
    auto& entry = cached_instructions[std::make_pair(instr, indices_ptrs)];
    if (!entry.empty()) {
      return entry;
    }

    TF_ASSIGN_OR_RETURN(auto lowered_instr,
                        HloToMlir(instr, indices, provide_operand,
                                  call_target_provider, builder));

    // Convert from signed to signless.
    mlir::mhlo::RemoveSignTypeConverter sign_converter;
    for (auto& lowered : lowered_instr) {
      auto result_type = sign_converter.convertType(lowered.getType());
      lowered = builder
                    .create<mlir::UnrealizedConversionCastOp>(
                        lowered.getLoc(), result_type, lowered)
                    .getResult(0);
    }
    entry = lowered_instr;
    TF_RET_CHECK(!absl::c_any_of(
        entry, [](const auto& entry) { return entry == nullptr; }))
        << "null result for " << instr->ToShortString();
    return entry;
  };

  for (const auto* root : subgraph.roots) {
    TF_ASSIGN_OR_RETURN(auto root_results, emit_instr(root, indices));
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

  for (const Range& range : indexing_map.GetSymbolRanges()) {
    lbs->push_back(b.create<ConstantIndexOp>(range.lower_bound));
    ubs->push_back(b.create<ConstantIndexOp>(range.upper_bound + 1));
    // Note that this is not optimal, when there are mod constraints on symbols,
    // e.g. for reduce-window. In that case we have to extract loop steps from
    // the mod constraints.
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
  auto indices_and_injected_params = func.getArguments().drop_front(
      computation.computation().num_parameters());
  int num_injected_params = subgraph.injected_param_indices.size();
  auto indices = indices_and_injected_params.drop_back(num_injected_params);
  auto injected_params =
      indices_and_injected_params.take_back(num_injected_params);
  TF_ASSIGN_OR_RETURN(
      auto results,
      SubgraphToMlir(computation, subgraph, call_target_provider, parameters,
                     indices, injected_params, builder));

  // We have been converting signed types to signless types. To match the
  // function signature, we have to convert back to signed types.
  auto function = mlir::cast<mlir::func::FuncOp>(
      results.front().getDefiningOp()->getParentOp());
  const auto& function_results = function.getFunctionType().getResults();
  for (auto [index, function_result] : llvm::enumerate(function_results)) {
    results[index] =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                results[index].getLoc(), function_result, results[index])
            .getResult(0);
  }

  builder.create<mlir::func::ReturnOp>(results);
  return absl::OkStatus();
}

SmallVector<Value> EmitLoopNest(
    ImplicitLocOpBuilder& b, ValueRange dim_values, ValueRange iter_args_inits,
    const IndexingMap& indexing_map,
    const std::function<
        SmallVector<Value>(ValueRange /*iter_args*/, ValueRange /*dim_values*/,
                           ValueRange /*symbol_values*/)>& create_body) {
  SmallVector<Value, 4> lbs, ubs, steps;
  GetLoopBoundsFromIndexingMap(b, indexing_map, &lbs, &ubs, &steps);

  scf::LoopNest loop_nest = scf::buildLoopNest(
      b, b.getLoc(), lbs, ubs, steps, iter_args_inits,
      [&](OpBuilder& nested_builder, Location loc, ValueRange symbol_values,
          ValueRange iter_args) -> scf::ValueVector {
        ImplicitLocOpBuilder nested_b(loc, nested_builder);
        auto is_in_bounds = mlir_converter::CheckConstraints(
            indexing_map, dim_values, symbol_values, nested_b);
        auto if_op = nested_b.create<scf::IfOp>(
            is_in_bounds,
            [&](OpBuilder& then_builder, Location then_loc) -> void {
              b.setInsertionPointToStart(then_builder.getInsertionBlock());
              auto results = create_body(iter_args, dim_values, symbol_values);
              b.create<scf::YieldOp>(results);
            },
            [&](OpBuilder& else_b, Location else_loc) {
              b.setInsertionPointToStart(else_b.getInsertionBlock());
              b.create<scf::YieldOp>(iter_args);
            });

        return if_op.getResults();
      });
  return loop_nest.results;
}

}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
