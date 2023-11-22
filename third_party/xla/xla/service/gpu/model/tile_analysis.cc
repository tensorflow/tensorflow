/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/tile_analysis.h"

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::getAffineBinaryOpExpr;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::MLIRContext;

StatusOr<HloInstructionIndexing> ComputeCwiseOpIndexing(
    const HloInstruction* instr, MLIRContext* mlir_context) {
  auto dims = instr->shape().dimensions();
  IndexingMap identity_map{
      .affine_map =
          AffineMap::getMultiDimIdentityMap(dims.size(), mlir_context),
      .sizes = std::vector<int64_t>{dims.begin(), dims.end()}};

  std::vector<HloOperandIndexing> operand_indexing_maps;
  int64_t operand_count = instr->operand_count();
  operand_indexing_maps.reserve(operand_count);
  for (int64_t operand_id = 0; operand_id < operand_count; ++operand_id) {
    operand_indexing_maps.push_back({{identity_map}, operand_id});
  }
  return HloInstructionIndexing{std::move(operand_indexing_maps)};
}

StatusOr<HloInstructionIndexing> ComputeBroadcastOpIndexing(
    const HloBroadcastInstruction* bcast, MLIRContext* mlir_context) {
  auto output_dims = bcast->shape().dimensions();

  std::vector<AffineExpr> exprs;
  for (int64_t bcast_dim : bcast->dimensions()) {
    exprs.push_back(getAffineDimExpr(bcast_dim, mlir_context));
  }

  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .sizes = std::vector<int64_t>{output_dims.begin(), output_dims.end()}};

  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(indexing_map)}, .operand_id = 0}}};
}

StatusOr<HloInstructionIndexing> ComputeReduceOpIndexing(
    const HloReduceInstruction* reduce, int output_id,
    MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reduce_dims_ids(reduce->dimensions().begin(),
                                               reduce->dimensions().end());

  const Shape& input_shape = reduce->operand(output_id)->shape();
  const Shape& output_shape = reduce->shape().IsTuple()
                                  ? ShapeUtil::GetSubshape(reduce->shape(), {0})
                                  : reduce->shape();

  std::vector<int64_t> sizes(output_shape.dimensions().begin(),
                             output_shape.dimensions().end());
  int64_t reduced_dim_id = 0;
  int64_t output_dim_id = 0;
  std::vector<AffineExpr> exprs;
  for (auto [input_dim_id, input_dim] :
       llvm::enumerate(input_shape.dimensions())) {
    if (reduce_dims_ids.count(input_dim_id)) {
      exprs.push_back(getAffineSymbolExpr(reduced_dim_id++, mlir_context));
      sizes.push_back(input_dim);
      continue;
    }
    exprs.push_back(getAffineDimExpr(output_dim_id++, mlir_context));
  }
  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_shape.rank(), reduce_dims_ids.size(),
                                   exprs, mlir_context),
      .sizes = std::vector<int64_t>{sizes.begin(), sizes.end()}};

  std::vector<HloOperandIndexing> operand_indexing_maps;
  int64_t input_count = reduce->input_count();
  operand_indexing_maps.reserve(input_count);
  for (int64_t input_id = 0; input_id < input_count; ++input_id) {
    operand_indexing_maps.push_back({{indexing_map}, input_id});
  }
  return HloInstructionIndexing{std::move(operand_indexing_maps)};
}

StatusOr<HloInstructionIndexing> ComputeReverseOpIndexing(
    const HloReverseInstruction* reverse, MLIRContext* mlir_context) {
  absl::flat_hash_set<int64_t> reverse_dims(reverse->dimensions().begin(),
                                            reverse->dimensions().end());
  auto output_dims = reverse->shape().dimensions();

  std::vector<AffineExpr> exprs;
  for (auto [output_dim_id, output_dim] : llvm::enumerate(output_dims)) {
    auto dim_expr = getAffineDimExpr(output_dim_id, mlir_context);
    if (!reverse_dims.contains(output_dim_id)) {
      exprs.push_back(dim_expr);
      continue;
    }
    auto dim_size = getAffineConstantExpr(output_dim, mlir_context);
    auto neg_dim_expr = getAffineBinaryOpExpr(
        AffineExprKind::Mul, getAffineConstantExpr(-1, mlir_context), dim_expr);
    exprs.push_back(
        getAffineBinaryOpExpr(AffineExprKind::Add, neg_dim_expr, dim_size));
  }

  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .sizes = std::vector<int64_t>{output_dims.begin(), output_dims.end()}};

  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(indexing_map)}, .operand_id = 0}}};
}

StatusOr<HloInstructionIndexing> ComputeSliceOpIndexing(
    const HloSliceInstruction* slice, MLIRContext* mlir_context) {
  auto output_dims = slice->shape().dimensions();

  std::vector<AffineExpr> exprs;
  for (int64_t dim = 0; dim < output_dims.size(); ++dim) {
    AffineExpr offset =
        getAffineConstantExpr(slice->slice_starts()[dim], mlir_context);
    AffineExpr stride =
        getAffineConstantExpr(slice->slice_strides()[dim], mlir_context);
    AffineExpr dim_expr = getAffineDimExpr(dim, mlir_context);

    AffineExpr mul =
        getAffineBinaryOpExpr(AffineExprKind::Mul, stride, dim_expr);
    exprs.push_back(getAffineBinaryOpExpr(AffineExprKind::Add, offset, mul));
  }
  IndexingMap indexing_map{
      .affine_map = AffineMap::get(output_dims.size(), /*symbolCount=*/0, exprs,
                                   mlir_context),
      .sizes = std::vector<int64_t>{output_dims.begin(), output_dims.end()}};
  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(indexing_map)}, .operand_id = 0}}};
}

StatusOr<HloInstructionIndexing> ComputeTransposeOpIndexing(
    const HloTransposeInstruction* transpose, MLIRContext* mlir_context) {
  auto output_dims = transpose->shape().dimensions();
  std::vector<unsigned> permutation(transpose->dimensions().begin(),
                                    transpose->dimensions().end());

  IndexingMap permutation_map{
      .affine_map = mlir::inversePermutation(
          AffineMap::getPermutationMap(permutation, mlir_context)),
      .sizes = std::vector<int64_t>{output_dims.begin(), output_dims.end()}};

  return HloInstructionIndexing{{HloOperandIndexing{
      .indexing_maps = {std::move(permutation_map)}, .operand_id = 0}}};
}

template <typename T>
std::string ToStringImpl(const T& value) {
  std::string s;
  std::stringstream ss(s);
  ss << value;
  return ss.str();
}

}  // namespace

std::string ToString(const AffineMap& affine_map) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  affine_map.print(ss);
  return s;
}

std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map) {
  out << ToString(indexing_map.affine_map) << " with sizes "
      << absl::StrJoin(indexing_map.sizes, ", ") << "\n";
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         const HloOperandIndexing& operand_indexing) {
  out << "operand id = " << operand_indexing.operand_id << ' ';
  for (const auto& map : operand_indexing.indexing_maps) {
    out << map;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing) {
  for (const auto& operand_map : instr_indexing.operand_indexing_maps) {
    out << operand_map;
  }
  return out;
}

std::string IndexingMap::ToString() const { return ToStringImpl(*this); }

std::string HloOperandIndexing::ToString() const { return ToStringImpl(*this); }

std::string HloInstructionIndexing::ToString() const {
  return ToStringImpl(*this);
}

StatusOr<HloInstructionIndexing> ComputeInstructionIndexing(
    const HloInstruction* instr, int output_id, MLIRContext* mlir_context) {
  if (instr->IsElementwise()) {
    return ComputeCwiseOpIndexing(instr, mlir_context);
  }
  if (auto bcast = DynCast<HloBroadcastInstruction>(instr)) {
    return ComputeBroadcastOpIndexing(bcast, mlir_context);
  }
  if (auto reduce = DynCast<HloReduceInstruction>(instr)) {
    return ComputeReduceOpIndexing(reduce, output_id, mlir_context);
  }
  if (auto reverse = DynCast<HloReverseInstruction>(instr)) {
    return ComputeReverseOpIndexing(reverse, mlir_context);
  }
  if (auto slice = DynCast<HloSliceInstruction>(instr)) {
    return ComputeSliceOpIndexing(slice, mlir_context);
  }
  if (auto transpose = DynCast<HloTransposeInstruction>(instr)) {
    return ComputeTransposeOpIndexing(transpose, mlir_context);
  }
  return InvalidArgument("Unsupported instruction type");
}

}  // namespace gpu
}  // namespace xla
