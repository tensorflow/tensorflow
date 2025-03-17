/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/hlo_creation_utils.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/builder/lib/comparators.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
using absl::StrCat;

absl::StatusOr<HloInstruction*> MakeUnaryHlo(HloOpcode opcode,
                                             HloInstruction* operand,
                                             const OpMetadata* metadata) {
  HloComputation* computation = operand->parent();
  TF_ASSIGN_OR_RETURN(Shape unary_op_shape,
                      ShapeInference::InferUnaryOpShape(opcode, operand));
  return computation->AddInstruction(
      HloInstruction::CreateUnary(unary_op_shape, opcode, operand), metadata);
}

HloInstruction* MakeCopyHlo(HloInstruction* from, const Shape& to) {
  return from->AddInstruction(
      HloInstruction::CreateUnary(to, HloOpcode::kCopy, from));
}

absl::StatusOr<HloInstruction*> MakeBinaryHlo(
    HloOpcode opcode, HloInstruction* lhs, HloInstruction* rhs,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(Shape binary_op_shape,
                      ShapeInference::InferBinaryOpShape(opcode, lhs, rhs));
  return computation->AddInstruction(
      HloInstruction::CreateBinary(binary_op_shape, opcode, lhs, rhs), metadata,
      frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeCompareHlo(
    ComparisonDirection direction, HloInstruction* lhs, HloInstruction* rhs,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(
      Shape binary_op_shape,
      ShapeInference::InferBinaryOpShape(HloOpcode::kCompare, lhs, rhs));
  return computation->AddInstruction(
      HloInstruction::CreateCompare(binary_op_shape, lhs, rhs, direction),
      metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakePadHlo(
    HloInstruction* operand, HloInstruction* padding_value,
    const PaddingConfig& padding_config, const OpMetadata* metadata,
    const FrontendAttributes* frontend_attributes) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(computation, padding_value->parent());
  TF_ASSIGN_OR_RETURN(
      Shape pad_shape,
      ShapeInference::InferPadShape(operand->shape(), padding_value->shape(),
                                    padding_config));

  return computation->AddInstruction(
      HloInstruction::CreatePad(pad_shape, operand, padding_value,
                                padding_config),
      metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeSliceHlo(
    HloInstruction* operand, absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices, absl::Span<const int64_t> strides,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  HloComputation* computation = operand->parent();
  TF_ASSIGN_OR_RETURN(Shape slice_shape, ShapeInference::InferSliceShape(
                                             operand->shape(), start_indices,
                                             limit_indices, strides));
  return computation->AddInstruction(
      HloInstruction::CreateSlice(slice_shape, operand, start_indices,
                                  limit_indices, strides),
      metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeConvolveHlo(
    HloInstruction* lhs, HloInstruction* rhs, int64_t feature_group_count,
    int64_t batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config,
    std::optional<PrimitiveType> preferred_element_type,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(
      Shape convolve_shape,
      ShapeInference::InferConvolveShape(
          lhs->shape(), rhs->shape(), feature_group_count, batch_group_count,
          window, dimension_numbers, preferred_element_type));
  return computation->AddInstruction(
      HloInstruction::CreateConvolve(
          convolve_shape, lhs, rhs, feature_group_count, batch_group_count,
          window, dimension_numbers, precision_config),
      metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeTransposeHlo(
    HloInstruction* operand, absl::Span<const int64_t> dimensions) {
  TF_ASSIGN_OR_RETURN(
      Shape transpose_shape,
      ShapeInference::InferTransposeShape(operand->shape(), dimensions));
  return operand->AddInstruction(
      HloInstruction::CreateTranspose(transpose_shape, operand, dimensions));
}

absl::StatusOr<HloInstruction*> MakeReshapeHlo(const Shape& result_shape,
                                               HloInstruction* operand) {
  return operand->AddInstruction(
      HloInstruction::CreateReshape(result_shape, operand));
}

absl::StatusOr<HloInstruction*> MakeReshapeHlo(
    absl::Span<const int64_t> result_shape_dim_bounds,
    HloInstruction* operand) {
  Shape new_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                         result_shape_dim_bounds);
  return MakeReshapeHlo(new_shape, operand);
}

absl::StatusOr<HloInstruction*> MakeDynamicSliceHlo(
    HloInstruction* operand, absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64_t> slice_sizes, const OpMetadata* metadata) {
  // slice of a scalar is no-op
  if (start_indices.empty() || slice_sizes.empty()) {
    return operand;
  }
  HloComputation* computation = operand->parent();
  std::vector<Shape> scalar_start_indices_shapes(
      start_indices.size(),
      ShapeUtil::MakeShape(start_indices[0]->shape().element_type(), {}));
  TF_ASSIGN_OR_RETURN(
      Shape dynamic_slice_shape,
      ShapeInference::InferDynamicSliceShape(
          operand->shape(), scalar_start_indices_shapes, slice_sizes));
  return computation->AddInstruction(
      HloInstruction::CreateDynamicSlice(dynamic_slice_shape, operand,
                                         start_indices, slice_sizes),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeDynamicSliceHlo(
    HloInstruction* operand, HloInstruction* start_indices,
    absl::Span<const int64_t> slice_sizes, const OpMetadata* metadata) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(computation, start_indices->parent());
  int64_t rank = start_indices->shape().dimensions(0);
  std::vector<HloInstruction*> scalar_start_indices;
  for (int i = 0; i < rank; ++i) {
    // TODO(b/118437727): Update callers to provide scalars directly.
    auto slice = computation->AddInstruction(HloInstruction::CreateSlice(
        ShapeUtil::MakeShape(start_indices->shape().element_type(), {1}),
        start_indices, {i}, {i + 1}, {1}));
    scalar_start_indices.push_back(
        computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(start_indices->shape().element_type(), {}),
            slice)));
  }
  std::vector<Shape> scalar_start_indices_shapes(
      rank, ShapeUtil::MakeShape(start_indices->shape().element_type(), {}));
  TF_ASSIGN_OR_RETURN(
      Shape dynamic_slice_shape,
      ShapeInference::InferDynamicSliceShape(
          operand->shape(), scalar_start_indices_shapes, slice_sizes));
  return computation->AddInstruction(
      HloInstruction::CreateDynamicSlice(dynamic_slice_shape, operand,
                                         scalar_start_indices, slice_sizes),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeDynamicUpdateSliceHlo(
    HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices, const OpMetadata* metadata) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(computation, update->parent());
  CHECK_EQ(computation, start_indices->parent());
  int64_t rank = start_indices->shape().dimensions(0);
  std::vector<HloInstruction*> scalar_start_indices;
  for (int i = 0; i < rank; ++i) {
    // TODO(b/118437727): Update callers to provide scalars directly.
    auto slice = computation->AddInstruction(HloInstruction::CreateSlice(
        ShapeUtil::MakeShape(start_indices->shape().element_type(), {1}),
        start_indices, {i}, {i + 1}, {1}));
    scalar_start_indices.push_back(
        computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(start_indices->shape().element_type(), {}),
            slice)));
  }
  std::vector<Shape> scalar_start_indices_shapes(
      rank, ShapeUtil::MakeShape(start_indices->shape().element_type(), {}));
  TF_ASSIGN_OR_RETURN(
      Shape dynamic_update_slice_shape,
      ShapeInference::InferDynamicUpdateSliceShape(
          operand->shape(), update->shape(), scalar_start_indices_shapes));
  return computation->AddInstruction(
      HloInstruction::CreateDynamicUpdateSlice(
          dynamic_update_slice_shape, operand, update, scalar_start_indices),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeDynamicUpdateSliceHlo(
    HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices,
    const OpMetadata* metadata) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(computation, update->parent());
  std::vector<Shape> scalar_start_indices_shapes;
  scalar_start_indices_shapes.reserve(start_indices.size());
  for (auto start_index : start_indices) {
    scalar_start_indices_shapes.push_back(start_index->shape());
  }
  TF_ASSIGN_OR_RETURN(
      Shape dynamic_update_slice_shape,
      ShapeInference::InferDynamicUpdateSliceShape(
          operand->shape(), update->shape(), scalar_start_indices_shapes));
  return computation->AddInstruction(
      HloInstruction::CreateDynamicUpdateSlice(dynamic_update_slice_shape,
                                               operand, update, start_indices),
      metadata);
}

HloInstruction* MakeBroadcastHlo(
    HloInstruction* operand, absl::Span<const int64_t> broadcast_dimensions,
    absl::Span<const int64_t> result_shape_bounds, const OpMetadata* metadata,
    const FrontendAttributes* frontend_attributes) {
  Shape broadcast_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                               result_shape_bounds);
  return MakeBroadcastHlo(operand, broadcast_dimensions, broadcast_shape,
                          metadata, frontend_attributes);
}

HloInstruction* MakeBroadcastHlo(
    HloInstruction* operand, absl::Span<const int64_t> broadcast_dimensions,
    const Shape& shape, const OpMetadata* metadata,
    const FrontendAttributes* frontend_attributes) {
  HloComputation* computation = operand->parent();
  return computation->AddInstruction(
      HloInstruction::CreateBroadcast(shape, operand, broadcast_dimensions),
      metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeGetTupleElementHlo(
    HloInstruction* operand, int64_t index, const OpMetadata* metadata) {
  HloComputation* computation = operand->parent();

  TF_ASSIGN_OR_RETURN(
      Shape gte_shape,
      ShapeInference::InferGetTupleElementShape(operand->shape(), index));
  return computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(gte_shape, operand, index),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeConcatHlo(
    absl::Span<HloInstruction* const> operands, int64_t dimension,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  CHECK_GT(operands.size(), 0);

  HloComputation* computation = operands[0]->parent();
  CHECK(absl::c_all_of(operands, [&](HloInstruction* instr) {
    return instr->parent() == computation;
  }));

  std::vector<const Shape*> operand_shapes;
  absl::c_transform(operands, std::back_inserter(operand_shapes),
                    [](HloInstruction* instr) { return &instr->shape(); });

  TF_ASSIGN_OR_RETURN(Shape concat_shape, ShapeInference::InferConcatOpShape(
                                              operand_shapes, dimension));
  return computation->AddInstruction(
      HloInstruction::CreateConcatenate(concat_shape, operands, dimension),
      metadata, frontend_attributes);
}

HloInstruction* MakeConvertToHlo(HloInstruction* hlo, PrimitiveType type,
                                 const OpMetadata* metadata) {
  if (hlo->shape().element_type() == type) {
    return hlo;
  }
  Shape shape = ShapeUtil::ChangeElementType(hlo->shape(), type);
  if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
    shape.mutable_layout()->set_element_size_in_bits(
        primitive_util::BitWidth(shape.element_type()));
  } else {
    shape.mutable_layout()->set_element_size_in_bits(0);
  }

  hlo = hlo->parent()->AddInstruction(HloInstruction::CreateConvert(shape, hlo),
                                      metadata);
  CHECK_EQ(hlo->shape().element_type(), type);
  return hlo;
}

HloInstruction* MakeBitcastHlo(HloInstruction* hlo, const Shape& shape,
                               const OpMetadata* metadata) {
  return hlo->parent()->AddInstruction(
      HloInstruction::CreateBitcast(shape, hlo), metadata);
}

HloInstruction* MakeBitcastConvertToHlo(HloInstruction* hlo, PrimitiveType type,
                                        const OpMetadata* metadata) {
  if (hlo->shape().element_type() == type) {
    return hlo;
  }
  Shape shape = ShapeUtil::ChangeElementType(hlo->shape(), type);
  // PRED are stored as one byte, PRED have a BitWidth of 1, avoid this problem
  // by using a convert instead of bitcast convert.
  if (type == PRED || hlo->shape().element_type() == PRED) {
    return MakeConvertToHlo(hlo, type);
  }
  hlo = hlo->parent()->AddInstruction(
      HloInstruction::CreateBitcastConvert(shape, hlo), metadata);
  CHECK_EQ(hlo->shape().element_type(), type);
  return hlo;
}

HloInstruction* MakeIotaHlo(HloComputation* computation, const Shape& shape,
                            int64_t iota_dimension) {
  return computation->AddInstruction(
      HloInstruction::CreateIota(shape, iota_dimension));
}

absl::StatusOr<HloInstruction*> MakeDotHlo(
    HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config,
    std::optional<PrimitiveType> preferred_element_type,
    std::vector<SparsityDescriptor> sparsity,
    absl::Span<HloInstruction* const> sparse_meta, const OpMetadata* metadata) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  TF_ASSIGN_OR_RETURN(Shape dot_shape,
                      ShapeInference::InferDotOpShape(
                          lhs->shape(), rhs->shape(), dim_numbers,
                          preferred_element_type, absl::MakeSpan(sparsity)));
  return computation->AddInstruction(
      HloInstruction::CreateDot(dot_shape, lhs, rhs, dim_numbers,
                                precision_config, sparsity, sparse_meta),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeRaggedDotHlo(
    HloInstruction* lhs, HloInstruction* rhs, HloInstruction* group_sizes,
    const RaggedDotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config,
    std::optional<PrimitiveType> preferred_element_type) {
  HloComputation* computation = lhs->parent();
  CHECK_EQ(computation, rhs->parent());
  CHECK_EQ(computation, group_sizes->parent());
  TF_ASSIGN_OR_RETURN(Shape ragged_dot_shape,
                      ShapeInference::InferRaggedDotOpShape(
                          lhs->shape(), rhs->shape(), group_sizes->shape(),
                          dim_numbers, preferred_element_type));
  return computation->AddInstruction(HloInstruction::CreateRaggedDot(
      ragged_dot_shape, lhs, rhs, group_sizes, dim_numbers, precision_config));
}

absl::StatusOr<HloInstruction*> MakeMapHlo(
    absl::Span<HloInstruction* const> operands, HloComputation* map_computation,
    const OpMetadata* metadata) {
  CHECK(!operands.empty()) << "Map Hlo requires at least one operand.";
  HloComputation* computation = operands.front()->parent();
  std::vector<const Shape*> operand_shapes;
  int64_t max_operand_rank = 0;
  for (const HloInstruction* operand : operands) {
    CHECK_EQ(computation, operand->parent());
    operand_shapes.push_back(&operand->shape());
    max_operand_rank = std::max(max_operand_rank, operand->shape().rank());
  }
  std::vector<int64_t> map_dims(max_operand_rank);
  std::iota(map_dims.begin(), map_dims.end(), 0);
  TF_ASSIGN_OR_RETURN(
      Shape map_shape,
      ShapeInference::InferMapShape(
          operand_shapes, map_computation->ComputeProgramShape(), map_dims));
  return computation->AddInstruction(
      HloInstruction::CreateMap(map_shape, operands, map_computation),
      metadata);
}

HloInstruction* MakeReducePrecisionHlo(HloInstruction* operand,
                                       int exponent_bits, int mantissa_bits,
                                       const OpMetadata* metadata) {
  return operand->parent()->AddInstruction(
      HloInstruction::CreateReducePrecision(operand->shape(), operand,
                                            exponent_bits, mantissa_bits),
      metadata);
}

namespace {
static HloComputation* MakeBinaryScalarComputation(HloOpcode binary_opcode,
                                                   PrimitiveType dtype,
                                                   HloInstruction* ctx,
                                                   HloModule* module) {
  CHECK_NE(ctx, nullptr);
  HloComputation::Builder b(
      absl::StrCat(ctx->name(), ".reduce_sub_computation"));
  const Shape scalar_shape = ShapeUtil::MakeShape(dtype, {});
  HloInstruction* lhs =
      b.AddInstruction(HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  HloInstruction* rhs =
      b.AddInstruction(HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  b.AddInstruction(
      HloInstruction::CreateBinary(scalar_shape, binary_opcode, lhs, rhs));
  CHECK_NE(module, nullptr);
  return module->AddEmbeddedComputation(b.Build());
}
}  // namespace

absl::StatusOr<HloInstruction*> MakeReduceHlo(
    HloInstruction* operand, HloInstruction* init_value,
    absl::Span<const int64_t> dimensions, HloComputation* reduce_computation,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  auto scalar_shape = ShapeUtil::MakeShape(operand->shape().element_type(), {});
  auto result_shape = ShapeUtil::DeleteDimensions(dimensions, operand->shape());

  return operand->parent()->AddInstruction(
      HloInstruction::CreateReduce(result_shape, operand, init_value,
                                   dimensions, reduce_computation),
      metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeReduceWindowHlo(
    HloInstruction* operand, HloInstruction* init_value, const Window& window,
    HloComputation* reduce_computation, const OpMetadata* metadata) {
  TF_ASSIGN_OR_RETURN(Shape inferred_shape,
                      ShapeInference::InferReduceWindowShape(
                          operand->shape(), init_value->shape(), window,
                          reduce_computation->ComputeProgramShape()));
  return operand->parent()->AddInstruction(
      HloInstruction::CreateReduceWindow(inferred_shape, operand, init_value,
                                         window, reduce_computation),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeReduceWindowHlo(
    HloInstruction* operand, HloInstruction* init_value, const Window& window,
    HloOpcode binary_opcode, const OpMetadata* metadata) {
  HloComputation* reduce_computation = MakeBinaryScalarComputation(
      binary_opcode, operand->shape().element_type(), operand,
      operand->GetModule());
  TF_ASSIGN_OR_RETURN(Shape inferred_shape,
                      ShapeInference::InferReduceWindowShape(
                          operand->shape(), init_value->shape(), window,
                          reduce_computation->ComputeProgramShape()));
  return operand->parent()->AddInstruction(
      HloInstruction::CreateReduceWindow(inferred_shape, operand, init_value,
                                         window, reduce_computation),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeReduceHlo(
    HloInstruction* operand, HloInstruction* init_value,
    absl::Span<const int64_t> dimensions, HloOpcode binary_opcode,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  HloComputation* reduce_computation = MakeBinaryScalarComputation(
      binary_opcode, operand->shape().element_type(), operand,
      operand->GetModule());
  return MakeReduceHlo(operand, init_value, dimensions, reduce_computation,
                       metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeReduceHlo(
    HloInstruction* operand, HloInstruction* init_value,
    HloOpcode binary_opcode, HloModule* module, const OpMetadata* metadata,
    const FrontendAttributes* frontend_attributes) {
  DCHECK_NE(nullptr, module);
  std::vector<int64_t> all_dims(operand->shape().rank());
  std::iota(all_dims.begin(), all_dims.end(), 0);

  HloComputation* reduce_computation = MakeBinaryScalarComputation(
      binary_opcode, operand->shape().element_type(), operand, module);
  return MakeReduceHlo(operand, init_value, all_dims, reduce_computation,
                       metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeReduceHlo(
    absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values,
    absl::Span<const int64_t> dimensions, HloComputation* reduce_computation,
    const OpMetadata* metadata, const FrontendAttributes* frontend_attributes) {
  CHECK(!operands.empty());
  CHECK_EQ(operands.size(), init_values.size());
  auto root = reduce_computation->root_instruction();
  if (root->shape().IsTuple()) {
    CHECK_EQ(root->shape().tuple_shapes_size(), operands.size());
  } else {
    CHECK_EQ(operands.size(), 1);
  }

  std::vector<Shape> expected_shapes;
  for (auto operand : operands) {
    expected_shapes.push_back(ShapeUtil::FilterDimensions(
        [&](const int64_t dim) {
          return !absl::c_linear_search(dimensions, dim);
        },
        operand->shape()));
  }

  auto output_shape = ShapeUtil::MakeMaybeTupleShape(expected_shapes);

  return operands[0]->parent()->AddInstruction(
      HloInstruction::CreateReduce(output_shape, operands, init_values,
                                   dimensions, reduce_computation),
      metadata, frontend_attributes);
}

absl::StatusOr<HloInstruction*> MakeReverseHlo(
    HloInstruction* operand, absl::Span<const int64_t> dimensions,
    const OpMetadata* metadata) {
  HloComputation* computation = operand->parent();
  TF_ASSIGN_OR_RETURN(Shape reverse_shape, ShapeInference::InferReverseShape(
                                               operand->shape(), dimensions));
  return computation->AddInstruction(
      HloInstruction::CreateReverse(reverse_shape, operand, dimensions),
      metadata);
}

absl::StatusOr<HloInstruction*> MakeSelectHlo(
    HloInstruction* pred, HloInstruction* on_true, HloInstruction* on_false,
    HloInstruction* derived_from, const OpMetadata* metadata,
    const FrontendAttributes* frontend_attributes) {
  HloComputation* computation = pred->parent();
  DCHECK_EQ(computation, on_true->parent());
  DCHECK_EQ(computation, on_false->parent());
  Shape op_shape = on_true->shape();
  if (ShapeUtil::IsScalar(pred->shape())) {
    if (!ShapeUtil::IsScalar(op_shape) && !op_shape.IsTuple()) {
      // If the output is not scalar, we need to broadcast the condition
      // to match the contract of kSelect.
      pred = computation->AddInstruction(
          HloInstruction::CreateBroadcast(
              ShapeUtil::ChangeElementType(op_shape, PrimitiveType::PRED), pred,
              {}),
          metadata, frontend_attributes);
      if (derived_from) {
        derived_from->SetupDerivedInstruction(pred);
      }
    }
  }
  TF_RET_CHECK(!op_shape.IsTuple());
  HloOpcode select_op_code = HloOpcode::kSelect;
  TF_ASSIGN_OR_RETURN(Shape select_shape,
                      ShapeInference::InferTernaryOpShape(select_op_code, pred,
                                                          on_true, on_false));
  HloInstruction* select = computation->AddInstruction(
      HloInstruction::CreateTernary(select_shape, select_op_code, pred, on_true,
                                    on_false),
      metadata, frontend_attributes);
  if (derived_from) {
    derived_from->SetupDerivedInstruction(select);
  }
  return select;
}

HloInstruction* MaybeMakeTuple(absl::Span<HloInstruction* const> operands) {
  CHECK(!operands.empty());
  if (operands.size() == 1) {
    return operands[0];
  }
  return operands[0]->parent()->AddInstruction(
      HloInstruction::CreateTuple(operands));
}

absl::StatusOr<HloComputation*> XlaComputationToHloComputation(
    XlaComputation& src_comp, HloModule* dest_module) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, src_comp.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module,
                      HloModule::CreateFromProto(src_comp.proto(), config));
  HloCloneContext context(dest_module);
  return dest_module->DeepCloneComputation(new_module->entry_computation(),
                                           &context);
}

absl::StatusOr<HloInstruction*> MakeSortHlo(
    const Shape& sort_shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension_to_sort, bool is_stable, HloComputation::Builder* builder,
    HloModule* module, const OpMetadata* metadata) {
  CHECK(!operands.empty()) << "Sort Hlo requires at least one operand.";
  XlaBuilder b("Sort.Compare");
  if (metadata != nullptr) {
    b.SetOpMetadata(*metadata);
  }
  std::vector<PrimitiveType> operand_types(operands.size());
  for (int64_t i = 0; i < operands.size(); ++i) {
    operand_types[i] = operands[i]->shape().element_type();
  }
  XlaComputation comparator = CreateScalarLtComputation(operand_types, &b);
  TF_ASSIGN_OR_RETURN(HloComputation * compare_computation,
                      XlaComputationToHloComputation(comparator, module));
  return builder->AddInstruction(HloInstruction::CreateSort(
      sort_shape, dimension_to_sort, operands, compare_computation, is_stable));
}

absl::StatusOr<HloInstruction*> CollapseFirstNDims(HloInstruction* operand,
                                                   int64_t n) {
  CHECK_GT(n, 0);

  const Shape& operand_shape = operand->shape();
  CHECK_GE(operand_shape.rank(), n);
  int64_t new_shape_leading_bound = 1;
  bool new_shape_leading_is_dynamic = false;
  for (int64_t i = 0; i < n; i++) {
    new_shape_leading_bound *= operand_shape.dimensions(i);
    new_shape_leading_is_dynamic |= operand_shape.is_dynamic_dimension(i);
  }

  std::vector<int64_t> new_shape_dims;
  new_shape_dims.reserve(operand_shape.rank() - n + 1);
  new_shape_dims.push_back(new_shape_leading_bound);

  std::copy(operand_shape.dimensions().begin() + n,
            operand_shape.dimensions().end(),
            std::back_inserter(new_shape_dims));

  std::vector<bool> new_shape_dynamic_dims;
  new_shape_dynamic_dims.reserve(operand_shape.rank() - n + 1);
  new_shape_dynamic_dims.push_back(new_shape_leading_is_dynamic);
  std::copy(operand_shape.dynamic_dimensions().begin() + n,
            operand_shape.dynamic_dimensions().end(),
            std::back_inserter(new_shape_dynamic_dims));

  Shape output_shape = ShapeUtil::MakeShape(
      operand_shape.element_type(), new_shape_dims, new_shape_dynamic_dims);

  return MakeReshapeHlo(output_shape, operand);
}

absl::StatusOr<HloInstruction*> PrependDegenerateDims(HloInstruction* operand,
                                                      int64_t n) {
  CHECK_GT(n, 0);
  std::vector<int64_t> new_shape_dims;
  const Shape& operand_shape = operand->shape();
  new_shape_dims.reserve(n + operand_shape.rank());
  new_shape_dims.insert(new_shape_dims.begin(), n, 1);
  absl::c_copy(operand_shape.dimensions(), std::back_inserter(new_shape_dims));
  return MakeReshapeHlo(new_shape_dims, operand);
}

absl::StatusOr<HloInstruction*> ExpandFirstDimIntoNDims(
    HloInstruction* operand, absl::Span<const int64_t> expanded_dims) {
  CHECK_GT(operand->shape().rank(), 0);
  CHECK_EQ(operand->shape().dimensions(0), Product(expanded_dims));

  std::vector<int64_t> expanded_shape_dim_bounds;
  expanded_shape_dim_bounds.reserve(expanded_dims.size() +
                                    operand->shape().rank() - 1);
  absl::c_copy(expanded_dims, std::back_inserter(expanded_shape_dim_bounds));
  std::copy(operand->shape().dimensions().begin() + 1,
            operand->shape().dimensions().end(),
            std::back_inserter(expanded_shape_dim_bounds));
  Shape new_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                         expanded_shape_dim_bounds);
  return MakeReshapeHlo(new_shape, operand);
}

absl::StatusOr<HloInstruction*> ElideDegenerateDims(
    HloInstruction* operand, absl::Span<const int64_t> dims_to_elide) {
  return MakeReshapeHlo(ShapeUtil::FilterDimensions(
                            [&](int64_t dim) {
                              return !absl::c_linear_search(dims_to_elide, dim);
                            },
                            operand->shape()),
                        operand);
}

absl::StatusOr<HloInstruction*> InsertDegenerateDims(
    HloInstruction* operand, absl::Span<const int64_t> dims_to_insert) {
  CHECK(absl::c_is_sorted(dims_to_insert));

  const Shape& operand_shape = operand->shape();
  int64_t output_shape_rank = operand_shape.rank() + dims_to_insert.size();
  for (auto dim_to_insert : dims_to_insert) {
    CHECK_LT(dim_to_insert, output_shape_rank);
  }

  std::vector<int64_t> output_shape_dim_bounds;
  output_shape_dim_bounds.reserve(output_shape_rank);
  int64_t operand_dims_idx = 0;
  int64_t dims_to_insert_idx = 0;
  for (int64_t i = 0; i < output_shape_rank; ++i) {
    if (dims_to_insert_idx < dims_to_insert.size() &&
        i == dims_to_insert[dims_to_insert_idx]) {
      output_shape_dim_bounds.push_back(1);
      ++dims_to_insert_idx;
    } else {
      output_shape_dim_bounds.push_back(
          operand_shape.dimensions(operand_dims_idx));
      ++operand_dims_idx;
    }
  }

  Shape output_shape = ShapeUtil::MakeShape(operand_shape.element_type(),
                                            output_shape_dim_bounds);
  return MakeReshapeHlo(output_shape, operand);
}

absl::StatusOr<HloInstruction*> PadVectorWithZeros(HloInstruction* operand,
                                                   int64_t zeros_to_prepend,
                                                   int64_t zeros_to_append) {
  HloComputation* computation = operand->parent();
  CHECK_EQ(operand->shape().rank(), 1);
  PaddingConfig padding_config;
  PaddingConfig::PaddingConfigDimension padding_config_dim;
  padding_config_dim.set_edge_padding_low(zeros_to_prepend);
  padding_config_dim.set_edge_padding_high(zeros_to_append);
  *padding_config.add_dimensions() = padding_config_dim;

  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(operand->shape().element_type())));
  return MakePadHlo(operand, zero, padding_config);
}

HloInstruction* BroadcastZeros(HloComputation* computation,
                               PrimitiveType element_type,
                               absl::Span<const int64_t> broadcast_dimensions) {
  return BroadcastZeros(
      computation, ShapeUtil::MakeShape(element_type, broadcast_dimensions));
}

HloInstruction* BroadcastZeros(HloComputation* computation,
                               const Shape& broadcast_shape) {
  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(broadcast_shape.element_type())));
  return MakeBroadcastHlo(zero, /*broadcast_dimensions=*/{}, broadcast_shape);
}

HloInstruction* BroadcastOnes(HloComputation* computation,
                              PrimitiveType element_type,
                              absl::Span<const int64_t> broadcast_dimensions) {
  HloInstruction* one = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(element_type)));
  return MakeBroadcastHlo(one, /*broadcast_dimensions=*/{},
                          /*result_shape_bounds=*/broadcast_dimensions);
}

absl::StatusOr<HloInstruction*> MakeFusionInstruction(
    HloInstruction* fused, HloInstruction::FusionKind kind) {
  HloComputation* comp = fused->parent();
  HloInstruction* fusion_instruction = comp->AddInstruction(
      HloInstruction::CreateFusion(fused->shape(), kind, fused));
  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(fused, fusion_instruction));
  return fusion_instruction;
}

// Recursively creates a dummy op given a shape. Leaf nodes are broadcasted zero
// while internal nodes are tuples.
HloInstruction* CreateDummyOp(HloComputation::Builder* b, const Shape& shape) {
  if (shape.IsArray()) {
    auto zero = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(shape.element_type())));
    return b->AddInstruction(HloInstruction::CreateBroadcast(shape, zero, {}));
  }
  CHECK(shape.IsTuple());
  std::vector<HloInstruction*> sub_instructions;
  for (const Shape& subshape : shape.tuple_shapes()) {
    sub_instructions.push_back(CreateDummyOp(b, subshape));
  }
  return b->AddInstruction(HloInstruction::CreateTuple(sub_instructions));
}

absl::StatusOr<std::unique_ptr<HloComputation>> CreateComputationWithSignature(
    absl::Span<const Shape* const> domain, const Shape& range,
    absl::string_view name) {
  HloComputation::Builder b{name};
  int64_t param_idx = 0;
  for (const Shape* param_shape : domain) {
    b.AddInstruction(HloInstruction::CreateParameter(
        param_idx, *param_shape, StrCat("param.", param_idx)));
    param_idx++;
  }

  // We can't change the root type of a computation once it is created so create
  // a dummy root instruction to give the computation the right root shape.  Use
  // a (recursive) broadcast here to avoid creating large constants.
  CreateDummyOp(&b, range);
  return b.Build();
}

HloInstruction* CreateDegenerateRemovingReshape(HloInstruction* hlo,
                                                const int64_t index_to_remove) {
  Shape input_shape = hlo->shape();
  std::vector<int64_t> dims;
  dims.reserve(input_shape.rank() - 1);
  for (int64_t index = 0; index < input_shape.rank(); index++) {
    if (index == index_to_remove) {
      continue;
    }
    int64_t dim_size = input_shape.dimensions(index);
    dims.push_back(dim_size);
  }
  Shape new_shape = ShapeUtil::MakeShape(input_shape.element_type(), dims);
  return hlo->AddInstruction(HloInstruction::CreateReshape(new_shape, hlo));
}

HloInstruction* CreateDegenerateAddingReshape(HloInstruction* hlo,
                                              const int index_to_add) {
  Shape input_shape = hlo->shape();
  std::vector<int64_t> dims;
  dims.reserve(input_shape.rank() - 1);
  for (int64_t index = 0; index < input_shape.rank(); index++) {
    if (index == index_to_add) {
      dims.push_back(1);
    }
    int64_t dim_size = input_shape.dimensions(index);
    dims.push_back(dim_size);
  }
  if (index_to_add == input_shape.rank()) {
    dims.push_back(1);
  }
  Shape new_shape = ShapeUtil::MakeShape(input_shape.element_type(), dims);
  return hlo->AddInstruction(HloInstruction::CreateReshape(new_shape, hlo));
}

HloInstruction* ExpandDegenerateReshape(HloInstruction* inst) {
  std::optional<ShapeUtil::ShapeEqualityDescriptor> reshape_degenerate =
      inst->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
  if (reshape_degenerate.has_value()) {
    if (reshape_degenerate->deleted_dimensions.empty() &&
        reshape_degenerate->inserted_dimensions.size() == 1) {
      return nullptr;
    }
    if (reshape_degenerate->inserted_dimensions.empty() &&
        reshape_degenerate->deleted_dimensions.size() == 1) {
      return nullptr;
    }
    absl::c_reverse(reshape_degenerate->deleted_dimensions);
    HloInstruction* degenerate_removing_hlo = nullptr;
    if (!reshape_degenerate->deleted_dimensions.empty()) {
      degenerate_removing_hlo = CreateDegenerateRemovingReshape(
          inst->mutable_operand(0), reshape_degenerate->deleted_dimensions[0]);
      for (int64_t r = 1; r < reshape_degenerate->deleted_dimensions.size();
           r++) {
        degenerate_removing_hlo = CreateDegenerateRemovingReshape(
            degenerate_removing_hlo, reshape_degenerate->deleted_dimensions[r]);
      }
    }
    HloInstruction* degenerate_adding_hlo = degenerate_removing_hlo != nullptr
                                                ? degenerate_removing_hlo
                                                : inst->mutable_operand(0);
    if (!reshape_degenerate->inserted_dimensions.empty()) {
      for (int64_t a = 0; a < reshape_degenerate->inserted_dimensions.size();
           a++) {
        degenerate_adding_hlo = CreateDegenerateAddingReshape(
            degenerate_adding_hlo, reshape_degenerate->inserted_dimensions[a]);
      }
    }
    return degenerate_adding_hlo;
  }
  return nullptr;
}

}  // namespace xla
