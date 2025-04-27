/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/dynamic_shaped_ops.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Given a list of shapes, create a shape whose dimensions are largest among all
// inputs.
//
// e.g.,
// shape_a = f32[10, 50]
// shape_b = f32[100, 10]
//
// result = f32[max(shape_a[0], shape_b[0]), max(shape_a[1], shape_b[1])]
//        = f32[100, 50]
Shape FindMaxShape(absl::Span<const Shape*> shapes) {
  CHECK(!shapes.empty());
  if (shapes[0]->IsTuple()) {
    // Recurse into sub-element.
    std::vector<Shape> results;
    results.reserve(shapes[0]->tuple_shapes_size());
    for (int i = 0; i < shapes[0]->tuple_shapes_size(); ++i) {
      std::vector<const Shape*> subshapes;
      subshapes.reserve(shapes.size());
      for (int64_t j = 0; j < shapes.size(); ++j) {
        subshapes.push_back(&shapes[j]->tuple_shapes(i));
      }
      results.push_back(FindMaxShape(absl::MakeSpan(subshapes)));
    }
    return ShapeUtil::MakeTupleShape(results);
  }
  Shape result = *shapes[0];

  for (const Shape* shape : shapes) {
    CHECK(result.dimensions().size() == shape->dimensions().size());
    for (int64_t dim = 0; dim < result.dimensions().size(); ++dim) {
      if (shape->dimensions(dim) > result.dimensions(dim)) {
        result.set_dimensions(dim, shape->dimensions(dim));
      }
    }
  }
  return result;
}

absl::StatusOr<XlaOp> ReconsileBranchDifference(const Shape& left_branch_shape,
                                                const Shape& right_branch_shape,
                                                XlaOp left_root) {
  if (left_branch_shape.IsTuple()) {
    // Invariant sanity check -- Left branch and right branch need to have
    // compatible shapes.
    CHECK(right_branch_shape.IsTuple() &&
          left_branch_shape.tuple_shapes_size() ==
              right_branch_shape.tuple_shapes_size());
    // Recurse into sub-element.
    std::vector<XlaOp> results;
    results.reserve(left_branch_shape.tuple_shapes_size());
    for (int i = 0; i < left_branch_shape.tuple_shapes_size(); ++i) {
      XlaOp sub_tuple = GetTupleElement(left_root, i);
      TF_ASSIGN_OR_RETURN(XlaOp elem,
                          ReconsileBranchDifference(
                              left_branch_shape.tuple_shapes(i),
                              right_branch_shape.tuple_shapes(i), sub_tuple));
      results.push_back(elem);
    }
    return Tuple(left_root.builder(), results);
  }
  XlaOp result = left_root;
  // Invariant sanity check -- Left branch and right branch need to have
  // compatible shapes.
  if (right_branch_shape.IsTuple()) {
    return InvalidArgument(
        "right_branch_shape should not be a tuple, received %s",
        right_branch_shape.DebugString());
  }
  if (left_branch_shape.dimensions().size() !=
      right_branch_shape.dimensions().size()) {
    return InvalidArgument(
        "left_branch_shape.dimensions_size() != "
        "right_branch_shape.dimensions_size() (%d vs %d)",
        left_branch_shape.dimensions().size(),
        right_branch_shape.dimensions().size());
  }
  for (int64_t dim = 0; dim < left_branch_shape.dimensions().size(); ++dim) {
    XlaOp original_dim = GetDimensionSize(result, dim);
    if (left_branch_shape.dimensions(dim) <
        right_branch_shape.dimensions(dim)) {
      int64_t diff = right_branch_shape.dimensions(dim) -
                     left_branch_shape.dimensions(dim);

      result = PadInDim(
          result, Zero(result.builder(), left_branch_shape.element_type()), dim,
          0, diff);
    }
    if (left_branch_shape.dimensions(dim) !=
        right_branch_shape.dimensions(dim)) {
      result = SetDimensionSize(result, original_dim, dim);
    }
  }
  return result;
}
}  // namespace
XlaOp DynamicConditional(XlaBuilder* builder, XlaOp predicate,
                         XlaOp true_operand,
                         const XlaComputation& true_computation,
                         XlaOp false_operand,
                         const XlaComputation& false_computation) {
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    auto true_shape = true_computation.GetProgramShape().value().result();

    auto false_shape = false_computation.GetProgramShape().value().result();

    if (ShapeUtil::Compatible(true_shape, false_shape)) {
      return xla::Conditional(predicate, true_operand, true_computation,
                              false_operand, false_computation);
    }

    auto reconsile_branch =
        [](const Shape& root_shape, const Shape& operand_shape,
           const Shape& reference_root_shape, const XlaComputation& computation)
        -> absl::StatusOr<XlaComputation> {
      xla::XlaBuilder builder("dynamic_builder");
      auto param = xla::Parameter(&builder, 0, operand_shape, "param");
      auto call = Call(&builder, computation, {param});

      auto elem =
          ReconsileBranchDifference(root_shape, reference_root_shape, call);
      if (!elem.ok()) return elem.status();
      return builder.Build();
    };
    TF_ASSIGN_OR_RETURN(
        auto true_computation_rewritten,
        reconsile_branch(true_shape, builder->GetShape(true_operand).value(),
                         false_shape, true_computation));

    TF_ASSIGN_OR_RETURN(
        auto false_computation_rewritten,
        reconsile_branch(false_shape, builder->GetShape(false_operand).value(),
                         true_shape, false_computation));
    return xla::Conditional(predicate, true_operand, true_computation_rewritten,
                            false_operand, false_computation_rewritten);
  });
}

XlaOp DynamicConditional(
    XlaBuilder* builder, XlaOp branch_index,
    absl::Span<const XlaComputation* const> branch_computations,
    absl::Span<const XlaOp> branch_operands) {
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    std::vector<Shape> root_shapes;
    root_shapes.reserve(branch_computations.size());
    for (int64_t i = 0; i < branch_computations.size(); ++i) {
      TF_ASSIGN_OR_RETURN(auto program_shape,
                          branch_computations[i]->GetProgramShape());
      root_shapes.push_back(program_shape.result());
    }
    TF_RET_CHECK(!root_shapes.empty());
    bool all_shapes_compatible =
        absl::c_all_of(root_shapes, [&](const Shape& shape) {
          return ShapeUtil::Compatible(root_shapes[0], shape);
        });
    if (all_shapes_compatible) {
      // All shapes are compatible, fall back to static case.
      return xla::Conditional(branch_index, branch_computations,
                              branch_operands);
    }

    std::vector<const Shape*> root_shapes_ptrs;
    root_shapes_ptrs.reserve(root_shapes.size());
    for (int64_t i = 0; i < root_shapes.size(); ++i) {
      root_shapes_ptrs.push_back(&root_shapes[i]);
    }

    Shape max_shape = FindMaxShape(absl::MakeSpan(root_shapes_ptrs));

    auto reconsile_branch =
        [](const Shape& root_shape, const Shape& operand_shape,
           const Shape& reference_root_shape, const XlaComputation& computation)
        -> absl::StatusOr<XlaComputation> {
      xla::XlaBuilder builder("dynamic_builder");
      auto param = xla::Parameter(&builder, 0, operand_shape, "param");
      auto call = Call(&builder, computation, {param});

      auto elem =
          ReconsileBranchDifference(root_shape, reference_root_shape, call);
      if (!elem.ok()) return elem.status();
      return builder.Build();
    };
    std::vector<XlaComputation> rewritten_computations;
    rewritten_computations.reserve(branch_computations.size());

    for (int64_t i = 0; i < branch_computations.size(); ++i) {
      TF_ASSIGN_OR_RETURN(Shape branch_operand_shape,
                          builder->GetShape(branch_operands[i]));

      TF_ASSIGN_OR_RETURN(auto rewritten,
                          reconsile_branch(root_shapes[i], branch_operand_shape,
                                           max_shape, *branch_computations[i]));
      rewritten_computations.push_back(std::move(rewritten));
    }
    std::vector<const XlaComputation*> rewritten_computation_ptrs;
    rewritten_computation_ptrs.reserve(branch_computations.size());
    for (int64_t i = 0; i < branch_computations.size(); ++i) {
      rewritten_computation_ptrs.push_back(&rewritten_computations[i]);
    }
    return xla::Conditional(branch_index, rewritten_computation_ptrs,
                            branch_operands);
  });
}

absl::StatusOr<XlaOp> SetDimensionSizeWithRebound(
    ValueInference* value_inference, XlaOp operand, XlaOp dimension_size,
    int64_t dimension) {
  auto inferred_bound_status_or = value_inference->AnalyzeConstant(
      dimension_size, xla::ValueInferenceMode::kUpperBound);

  auto dynamism_status_or = value_inference->AnalyzeIsDynamic(dimension_size);
  TF_RETURN_IF_ERROR(inferred_bound_status_or.status());
  TF_RETURN_IF_ERROR(dynamism_status_or.status());
  if (inferred_bound_status_or->AllValid()) {
    int64_t inferred_bound = inferred_bound_status_or->Get<int32_t>({}).value();
    TF_ASSIGN_OR_RETURN(auto* shape_ptr,
                        operand.builder()->GetShapePtr(operand));
    // Found a tighter bound, do a slice.
    if (shape_ptr->dimensions(dimension) > inferred_bound) {
      operand = xla::SliceInDim(operand, 0, inferred_bound, 1, dimension);
    }
  }
  if (dynamism_status_or->Get<bool>({})) {
    // dimension size is dynamic, make output dynamic by calling set dimension
    // size.
    operand = xla::SetDimensionSize(operand, dimension_size, dimension);
  }
  return operand;
}

absl::StatusOr<XlaOp> SetAllDimensionSizes(ValueInference* value_inference,
                                           XlaOp operand, XlaOp size_vector) {
  auto builder = value_inference->builder();
  TF_RETURN_IF_ERROR(builder->GetCurrentStatus());
  TF_ASSIGN_OR_RETURN(auto shape_ptr, builder->GetShapePtr(operand));

  for (int64_t i = 0; i < shape_ptr->dimensions().size(); ++i) {
    // If a dimension is dynamic, call set-dimension-size on the output.
    auto dim_size = xla::Slice(size_vector, {i}, {i + 1}, {1});
    dim_size = xla::Reshape(dim_size, {});
    dim_size = xla::ConvertElementType(dim_size, xla::S32);
    TF_ASSIGN_OR_RETURN(auto dynamism,
                        value_inference->AnalyzeIsDynamic(dim_size));
    if (dynamism.Get<bool>({})) {
      operand = xla::SetDimensionSize(operand, dim_size, i);
    }
  }
  return operand;
}
}  // namespace xla
