/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/shape_inference.h"

#include <stddef.h>
#include <algorithm>
#include <numeric>
#include <set>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace {

using absl::StrFormat;
using absl::StrJoin;

// Returns true if no element is present in slice more than once.
bool AllUnique(absl::Span<const int64> slice) {
  return std::set<int64>(slice.begin(), slice.end()).size() == slice.size();
}

Status ExpectArray(const Shape& shape, absl::string_view op_type) {
  if (!ShapeUtil::IsArray(shape)) {
    return InvalidArgument("Expected array argument for %s, but got %s.",
                           string(op_type), ShapeUtil::HumanString(shape));
  }
  return Status::OK();
}

Status VerifyReducerShape(const ProgramShape& reducer_shape,
                          absl::Span<const Shape* const> init_value_shapes,
                          absl::Span<const PrimitiveType> input_element_types,
                          int64 inputs) {
  if (reducer_shape.parameters_size() != inputs * 2) {
    return InvalidArgument(
        "Reduction function must take %d parameters, but "
        "takes %d parameter(s).",
        inputs * 2, reducer_shape.parameters_size());
  }

  const Shape& accumulator_shape = reducer_shape.result();
  std::vector<const Shape*> accumulator_subshapes;
  if (ShapeUtil::IsArray(accumulator_shape)) {
    if (inputs != 1) {
      return InvalidArgument(
          "Reduction function must produce a tuple with %d elements, but "
          "produces a scalar",
          inputs);
    }
    accumulator_subshapes.push_back(&accumulator_shape);
  } else if (ShapeUtil::IsTuple(accumulator_shape)) {
    if (ShapeUtil::TupleElementCount(accumulator_shape) != inputs) {
      return InvalidArgument(
          "Reduction function must produce a tuple with %d elements, but has "
          "%d elements",
          inputs, ShapeUtil::TupleElementCount(accumulator_shape));
    }
    for (const Shape& element_shape : accumulator_shape.tuple_shapes()) {
      accumulator_subshapes.push_back(&element_shape);
    }
  } else {
    return InvalidArgument(
        "Reduction function must produce a scalar or tuple of scalars, but has "
        "shape: %s",
        ShapeUtil::HumanString(accumulator_shape));
  }

  for (const Shape* element_shape : accumulator_subshapes) {
    if (ShapeUtil::Rank(*element_shape) != 0) {
      return InvalidArgument(
          "Reduction function must return a scalar or tuple of scalars but "
          "returns shape: %s",
          ShapeUtil::HumanString(accumulator_shape));
    }
  }

  for (int64 i = 0; i < inputs; ++i) {
    // Check that the accumulator can be passed in as the first argument.
    // Note: comparing here and below with Compatible since we don't care about
    // layout in scalars - see b/26668201 for a longer-term vision.
    if (!ShapeUtil::Compatible(*accumulator_subshapes[i],
                               reducer_shape.parameters(i))) {
      return InvalidArgument(
          "Reduction function's %d-th parameter shape differs from the "
          "result shape: %s vs %s",
          i, ShapeUtil::HumanString(reducer_shape.parameters(i)),
          ShapeUtil::HumanString(*accumulator_subshapes[i]));
    }
    // Check that init_value's shapes are suitable for reducer_shape.
    if (!ShapeUtil::CompatibleIgnoringFpPrecision(*accumulator_subshapes[i],
                                                  *init_value_shapes[i])) {
      return InvalidArgument(
          "Reduction function's accumulator shape at index %d differs from "
          "the init_value shape: %s vs %s",
          i, ShapeUtil::HumanString(*accumulator_subshapes[i]),
          ShapeUtil::HumanString(*init_value_shapes[i]));
    }
    // Check that the inputs can be passed in as the non-accumulator arguments.
    const Shape input_element_shape =
        ShapeUtil::MakeShape(input_element_types[i], {});
    if (!ShapeUtil::CompatibleIgnoringFpPrecision(
            input_element_shape, reducer_shape.parameters(inputs + i))) {
      return InvalidArgument(
          "Reduction function's %d-th parameter shape differs from the "
          "input type element type: %s vs %s",
          inputs + i,
          ShapeUtil::HumanString(reducer_shape.parameters(inputs + i)),
          ShapeUtil::HumanString(input_element_shape));
    }
    // Check that the accumulator and inputs to the reducer function match.
    // If the accumulator is scalar, it must have the same type as the inputs
    // (up to fp precision). If it is a tuple, then the k-th element of the
    // tuple must have the same type as the K-th input (again, up to fp
    // precision.)
    if (!ShapeUtil::CompatibleIgnoringFpPrecision(
            *accumulator_subshapes[i], reducer_shape.parameters(inputs + i))) {
      return InvalidArgument(
          "Reduction function's %d-th parameter shape must "
          "match the result shape, but got %s vs %s.",
          inputs + i,
          ShapeUtil::HumanString(reducer_shape.parameters(inputs + i)),
          ShapeUtil::HumanString(*accumulator_subshapes[i]));
    }
  }

  return Status::OK();
}

StatusOr<Shape> InferWindowOutputShape(const Shape& base_shape,
                                       const Window& window,
                                       PrimitiveType element_type,
                                       bool allow_negative_padding) {
  if (window.dimensions_size() != ShapeUtil::Rank(base_shape)) {
    return InvalidArgument(
        "Window has dimension %d but base shape has dimension %d.",
        window.dimensions_size(), ShapeUtil::Rank(base_shape));
  }

  std::vector<int64> output_dimensions(window.dimensions_size());
  for (int64 i = 0; i < window.dimensions_size(); ++i) {
    const auto& dim = window.dimensions(i);
    if (dim.size() <= 0) {
      return InvalidArgument("Window %s has a non-positive dimension.",
                             window.DebugString());
    }
    if (dim.stride() <= 0) {
      return InvalidArgument("Window %s has a non-positive stride.",
                             window.DebugString());
    }
    if (!allow_negative_padding && dim.padding_low() < 0) {
      return InvalidArgument("Window %s has a negative low padding.",
                             window.DebugString());
    }
    if (!allow_negative_padding && dim.padding_high() < 0) {
      return InvalidArgument("Window %s has a negative high padding.",
                             window.DebugString());
    }
    if (dim.base_dilation() < 1) {
      return InvalidArgument(
          "Window %s has a non-positive base area dilation factor.",
          window.DebugString());
    }
    if (dim.window_dilation() < 1) {
      return InvalidArgument(
          "Window %s has a non-positive window dilation factor.",
          window.DebugString());
    }

    const int64 dilated_base = window_util::DilatedBound(
        ShapeUtil::GetDimension(base_shape, i), dim.base_dilation());
    const int64 padded_dilated_base =
        dim.padding_low() + dilated_base + dim.padding_high();
    const int64 dilated_window =
        window_util::DilatedBound(dim.size(), dim.window_dilation());

    output_dimensions[i] = window_util::StridedBound(
        padded_dilated_base, dilated_window, dim.stride());
  }

  return ShapeUtil::MakeValidatedShape(element_type, output_dimensions);
}

}  // namespace

/* static */ StatusOr<Shape> ShapeInference::InferUnaryOpShape(
    HloOpcode opcode, const HloInstruction* operand) {
  return InferUnaryOpShape(opcode, operand->shape());
}

/* static */ StatusOr<Shape> ShapeInference::InferUnaryOpShape(
    HloOpcode opcode, const Shape& shape) {
  // There is no copy operation at the proto level, so handle copy explicitly.
  // A domain shape is the same as the input one.
  if (opcode == HloOpcode::kCopy || opcode == HloOpcode::kDomain) {
    return shape;
  }

  TF_RETURN_IF_ERROR(ExpectArray(shape, "operand of unary operation"));

  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape));
  switch (opcode) {
    case HloOpcode::kFloor:
    case HloOpcode::kCeil:
    case HloOpcode::kRoundNearestAfz:
      if (!ShapeUtil::ElementIsFloating(shape)) {
        return InvalidArgument(
            "Expected element type in shape to be floating for %s operation; "
            "got %s.",
            HloOpcodeString(opcode), PrimitiveType_Name(shape.element_type()));
      }
      return shape;
    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kTanh:
      if (!ShapeUtil::ElementIsFloating(shape) &&
          !ShapeUtil::ElementIsComplex(shape)) {
        return InvalidArgument(
            "Expected element type in shape to be floating or complex for %s "
            "operation; got %s.",
            HloOpcodeString(opcode), PrimitiveType_Name(shape.element_type()));
      }
      return shape;
    case HloOpcode::kReal:
    case HloOpcode::kImag:
      if (ShapeUtil::ElementIsComplex(shape)) {
        return ShapeUtil::ComplexComponentShape(shape);
      } else if (ShapeUtil::ElementIsFloating(shape)) {
        return shape;
      } else {
        return InvalidArgument(
            "Expected element type in shape to be floating or complex for "
            "%s operation; got %s.",
            HloOpcodeString(opcode), PrimitiveType_Name(shape.element_type()));
      }
    case HloOpcode::kAbs:
      if (ShapeUtil::ElementIsComplex(shape)) {
        return ShapeUtil::ChangeElementType(
            shape, primitive_util::ComplexComponentType(shape.element_type()));
      } else if (ShapeUtil::ElementIsSigned(shape)) {
        return shape;
      } else {
        return InvalidArgument(
            "Expected element type in shape to be floating or complex for "
            "%s operation; got %s.",
            HloOpcodeString(opcode), PrimitiveType_Name(shape.element_type()));
      }
    case HloOpcode::kClz:
      if (!ShapeUtil::ElementIsIntegral(shape)) {
        return InvalidArgument(
            "Expected an integral element type in argument to Clz "
            "operation; got %s.",
            PrimitiveType_Name(shape.element_type()));
      }
      return shape;
    case HloOpcode::kNegate:
      if (!ShapeUtil::ElementIsIntegral(shape) &&
          !ShapeUtil::ElementIsFloating(shape) &&
          !ShapeUtil::ElementIsComplex(shape)) {
        return InvalidArgument(
            "Expected element type in shape to be integral, floating or "
            "complex for %s operation; got %s.",
            HloOpcodeString(opcode), PrimitiveType_Name(shape.element_type()));
      }
      return shape;
    case HloOpcode::kSign:
      if (!ShapeUtil::ElementIsSigned(shape) &&
          !ShapeUtil::ElementIsComplex(shape)) {
        return InvalidArgument(
            "Expected element type in shape to be signed or complex for "
            "%s operation; got %s.",
            HloOpcodeString(opcode), PrimitiveType_Name(shape.element_type()));
      }
      return shape;

    case HloOpcode::kNot:
      if (shape.element_type() != PRED &&
          !primitive_util::IsIntegralType(shape.element_type())) {
        return InvalidArgument(
            "Expected pred or an integral element type in argument to Not "
            "operation; got %s.",
            PrimitiveType_Name(shape.element_type()));
      }
      return shape;

    case HloOpcode::kIsFinite:
      if (!ShapeUtil::ElementIsFloating(shape)) {
        return InvalidArgument(
            "Expected element type in shape to be floating "
            "point for IsFinite "
            "operation; got %s.",
            PrimitiveType_Name(shape.element_type()));
      }
      return ShapeUtil::ChangeElementType(shape, PRED);

    default:
      return InvalidArgument(
          "Unknown operation for unary shape inference: \"%s\".",
          HloOpcodeString(opcode));
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferConcatOpShape(
    absl::Span<const Shape* const> arg_shapes, const int64 dimension) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Concatenate expects at least one argument.");
  }
  if (dimension < 0 || dimension >= ShapeUtil::Rank(*arg_shapes[0])) {
    return InvalidArgument("Concatenate dimension out of bounds: %d.",
                           dimension);
  }
  const Shape* arg_shape = nullptr;
  PrimitiveType element_type = PRIMITIVE_TYPE_INVALID;
  for (const Shape* shape : arg_shapes) {
    TF_RETURN_IF_ERROR(ExpectArray(*shape, "operand of concatenation"));
    if (!arg_shape) {
      arg_shape = shape;
      element_type = arg_shape->element_type();
      continue;
    }
    if (ShapeUtil::Rank(*arg_shape) != ShapeUtil::Rank(*shape)) {
      return InvalidArgument(
          "Cannot concatenate arrays with different ranks: %d (%s) vs %d "
          "(%s).",
          ShapeUtil::Rank(*arg_shape), ShapeUtil::HumanString(*arg_shape),
          ShapeUtil::Rank(*shape), ShapeUtil::HumanString(*shape));
    }
    if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(*arg_shape, *shape)) {
      return InvalidArgument(
          "Cannot concatenate arrays with different element types: %s vs %s.",
          PrimitiveType_Name(arg_shape->element_type()),
          PrimitiveType_Name(shape->element_type()));
    }
    for (int64 dimension_number = 0;
         dimension_number < ShapeUtil::Rank(*arg_shape); ++dimension_number) {
      if (arg_shape->dimensions(dimension_number) !=
          shape->dimensions(dimension_number)) {
        if (dimension_number == dimension) {
          continue;  // It's okay to differ in the dimension we're
                     // concatenating.
        }
        return InvalidArgument(
            "Cannot concatenate arrays that differ in dimensions other than "
            "the one being concatenated (the other array dimensions must be "
            "the same): %s vs %s in dimension %d.",
            ShapeUtil::HumanString(*arg_shape), ShapeUtil::HumanString(*shape),
            dimension);
      }
    }
    element_type = ShapeUtil::HigherPrecisionElementType(*shape, *arg_shape);
  }

  std::vector<int64> new_dimensions(arg_shape->dimensions().begin(),
                                    arg_shape->dimensions().end());
  for (size_t i = 1; i < arg_shapes.size(); ++i) {
    new_dimensions[dimension] += arg_shapes[i]->dimensions(dimension);
  }
  return ShapeUtil::MakeShape(element_type, new_dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferConvertShape(
    const Shape& operand_shape, PrimitiveType new_element_type) {
  auto old_element_type = operand_shape.element_type();
  if (primitive_util::IsComplexType(old_element_type) &&
      !primitive_util::IsComplexType(new_element_type)) {
    return Unimplemented(
        "Conversion from complex to real type %s => %s is not implemented.",
        ShapeUtil::HumanString(operand_shape),
        PrimitiveType_Name(new_element_type));
  }
  if (!ShapeUtil::IsArray(operand_shape) ||
      !primitive_util::IsArrayType(new_element_type)) {
    // Note: we may want to support tuple conversions via this operation in the
    // future, by recursing into the tuple elements to check all sub-conversions
    // are valid. For now we just reject them, though.
    return InvalidArgument(
        "Convert does not allow non-arrays, so cannot convert from %s to %s.",
        ShapeUtil::HumanString(operand_shape),
        PrimitiveType_Name(new_element_type));
  }

  return ShapeUtil::ChangeElementType(operand_shape, new_element_type);
}

/* static */ StatusOr<Shape> ShapeInference::InferBitcastConvertShape(
    const Shape& operand_shape, PrimitiveType new_element_type) {
  auto old_element_type = operand_shape.element_type();
  if (primitive_util::IsComplexType(old_element_type) !=
      primitive_util::IsComplexType(new_element_type)) {
    return InvalidArgument("Conversion from complex to real type %s => %s.",
                           ShapeUtil::HumanString(operand_shape),
                           PrimitiveType_Name(new_element_type));
  }
  if (!ShapeUtil::IsArray(operand_shape) ||
      !primitive_util::IsArrayType(new_element_type)) {
    // Note: we may want to support tuple conversions via this operation in the
    // future, by recursing into the tuple elements to check all sub-conversions
    // are valid. For now we just reject them, though.
    return InvalidArgument(
        "Cannot convert from or to tuple type; requested conversion: %s => %s.",
        ShapeUtil::HumanString(operand_shape),
        PrimitiveType_Name(new_element_type));
  }
  if (primitive_util::BitWidth(old_element_type) !=
      primitive_util::BitWidth(new_element_type)) {
    return InvalidArgument(
        "Cannot bitcast types with different bit-widths: %s => %s.",
        PrimitiveType_Name(old_element_type),
        PrimitiveType_Name(new_element_type));
  }

  return ShapeUtil::ChangeElementType(operand_shape, new_element_type);
}

/* static */ StatusOr<Shape> ShapeInference::InferReducePrecisionShape(
    const Shape& operand_shape, const int exponent_bits,
    const int mantissa_bits) {
  if (!ShapeUtil::ElementIsFloating(operand_shape)) {
    return InvalidArgument(
        "Expected element type in shape to be floating point for "
        "ReducePrecision operation; got %s.",
        PrimitiveType_Name(operand_shape.element_type()));
  }
  if (exponent_bits < 1) {
    // One exponent bit is necessary to distinguish 0 from infinity.  Having
    // no exponent bits doesn't produce a sensible number, so we require at
    // least one.
    return InvalidArgument("Expected exponent_bits >= 1; got %d.",
                           exponent_bits);
  }
  if (mantissa_bits < 0) {
    // A number with no mantissa bits is still meaningful, however.
    return InvalidArgument("Expected non-negative mantissa_bits; got %d.",
                           mantissa_bits);
  }
  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferPadShape(
    const Shape& operand_shape, const Shape& padding_value_shape,
    const PaddingConfig& padding_config) {
  if (!ShapeUtil::IsArray(operand_shape)) {
    return InvalidArgument(
        "Pad operation does not support tuple-shape operands.");
  }
  if (!ShapeUtil::IsScalar(padding_value_shape)) {
    return InvalidArgument(
        "Pad operation does not support non-scalar padding values.");
  }
  if (ShapeUtil::Rank(operand_shape) != padding_config.dimensions_size()) {
    return InvalidArgument(
        "The rank of the operand and the padding configuration do not match: "
        "%s vs %s.",
        ShapeUtil::HumanString(operand_shape),
        padding_config.ShortDebugString());
  }
  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(operand_shape,
                                                     padding_value_shape)) {
    return InvalidArgument(
        "The element types of the operands to Pad do not match.");
  }
  if (absl::c_any_of(padding_config.dimensions(),
                     [](const PaddingConfig::PaddingConfigDimension& p) {
                       return p.interior_padding() < 0;
                     })) {
    return InvalidArgument("Interior padding cannot be negative: %s",
                           padding_config.ShortDebugString());
  }

  std::vector<int64> dimensions(ShapeUtil::Rank(operand_shape));
  for (int64 i = 0; i < operand_shape.dimensions_size(); ++i) {
    const auto& p = padding_config.dimensions(i);
    dimensions[i] = operand_shape.dimensions(i) + p.edge_padding_low() +
                    p.edge_padding_high() +
                    std::max<int64>(operand_shape.dimensions(i) - 1, 0LL) *
                        p.interior_padding();
  }
  return ShapeUtil::MakeShape(
      ShapeUtil::HigherPrecisionElementType(operand_shape, padding_value_shape),
      dimensions);
}

// Current DotDimensionNumbers Requirements:
//
// Contracting Dimensions:
// *) Exactly one contracting dimension on both lhs and rhs.
// *) Contracting dimension size must be the same on both lhs and rhs.
// *) Contracting dimension numbers do not need to be the same (i.e. transposes
//    are passed on to emitter implementations).
//
// Batch Dimensions:
// *) Same number of batch dimensions on both lhs and rhs.
// *) Same batch dimension numbers (and sizes) on both lhs and rhs.
// *) Batch dimension numbers must be ordered before contracting and
//    non-contracting/non-batch dimension numbers.
//
// Non-Contracting-Non-Batch Dimensions:
// *) Can be 0 (matrix-vector) or 1 (matrix-matrix).
//

namespace {

Status ValidateDotDimensionNumbers(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  // Check that dimension numbers are in range.
  auto dims_in_range = [](const int64 rank,
                          absl::Span<const int64> contracting_dims,
                          absl::Span<const int64> batch_dims) -> bool {
    auto in_range = [&rank](int64 i) -> bool { return 0 <= i && i < rank; };
    return std::all_of(contracting_dims.begin(), contracting_dims.end(),
                       in_range) &&
           std::all_of(batch_dims.begin(), batch_dims.end(), in_range);
  };

  absl::Span<const int64> lhs_contracting_dimensions =
      AsInt64Slice(dimension_numbers.lhs_contracting_dimensions());
  absl::Span<const int64> rhs_contracting_dimensions =
      AsInt64Slice(dimension_numbers.rhs_contracting_dimensions());
  absl::Span<const int64> lhs_batch_dimensions =
      AsInt64Slice(dimension_numbers.lhs_batch_dimensions());
  absl::Span<const int64> rhs_batch_dimensions =
      AsInt64Slice(dimension_numbers.rhs_batch_dimensions());

  if (!dims_in_range(ShapeUtil::Rank(lhs), lhs_contracting_dimensions,
                     lhs_batch_dimensions) ||
      !dims_in_range(ShapeUtil::Rank(rhs), rhs_contracting_dimensions,
                     rhs_batch_dimensions)) {
    return InvalidArgument("A dimension number is out of range in Dot: %s.",
                           dimension_numbers.DebugString());
  }

  // Check that dimension numbers are unique.
  auto dims_unique = [](absl::Span<const int64> contracting_dims,
                        absl::Span<const int64> batch_dims) -> bool {
    absl::flat_hash_set<int64> dim_set;
    auto is_unique = [&dim_set](int64 i) -> bool {
      return dim_set.insert(i).second;
    };
    return std::all_of(contracting_dims.begin(), contracting_dims.end(),
                       is_unique) &&
           std::all_of(batch_dims.begin(), batch_dims.end(), is_unique);
  };

  if (!dims_unique(lhs_contracting_dimensions, lhs_batch_dimensions) ||
      !dims_unique(rhs_contracting_dimensions, rhs_batch_dimensions)) {
    return InvalidArgument("A dimension number is not unique in Dot: %s.",
                           dimension_numbers.DebugString());
  }

  // Check that the count of non-contracting-non-batch dimensions is in {0, 1}.
  const int64 lhs_non_contracting_non_batch_dims =
      ShapeUtil::Rank(lhs) -
      dimension_numbers.lhs_contracting_dimensions_size() -
      dimension_numbers.lhs_batch_dimensions_size();
  const int64 rhs_non_contracting_non_batch_dims =
      ShapeUtil::Rank(rhs) -
      dimension_numbers.rhs_contracting_dimensions_size() -
      dimension_numbers.rhs_batch_dimensions_size();
  if (lhs_non_contracting_non_batch_dims < 0 ||
      lhs_non_contracting_non_batch_dims > 1 ||
      rhs_non_contracting_non_batch_dims < 0 ||
      rhs_non_contracting_non_batch_dims > 1) {
    return InvalidArgument(
        "Batch and contracting dimension number mismatch with rank.");
  }

  // Check that batch dimension numbers are ordered before all others, and
  // that they are monotonically increasing.
  std::vector<int64> batch_dim_numbers(lhs_batch_dimensions.size());
  std::iota(batch_dim_numbers.begin(), batch_dim_numbers.end(), 0);
  if (!std::equal(batch_dim_numbers.begin(), batch_dim_numbers.end(),
                  lhs_batch_dimensions.begin()) ||
      !std::equal(batch_dim_numbers.begin(), batch_dim_numbers.end(),
                  rhs_batch_dimensions.begin())) {
    return InvalidArgument(
        "Batch dimension numbers must precede non-batch dimensions and be"
        "monotonically increasing.");
  }

  return Status::OK();
}

}  // namespace

/* static */ StatusOr<Shape> ShapeInference::InferDotOpShape(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  TF_RETURN_IF_ERROR(ExpectArray(lhs, "lhs of dot"));
  TF_RETURN_IF_ERROR(ExpectArray(rhs, "rhs of dot"));

  auto fail = [lhs, rhs](const string& addendum) -> Status {
    string message =
        StrFormat("Cannot infer shape for dot operation: %s <dot> %s.",
                  ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs));
    if (!addendum.empty()) {
      message += " " + addendum;
    }
    return InvalidArgument("%s", message);
  };

  // Check if both element types are the same.
  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) {
    return fail("Element types do not match.");
  }

  if ((ShapeUtil::Rank(lhs) < 1) || (ShapeUtil::Rank(rhs) < 1)) {
    return fail("Dot only supports rank 1 or above.");
  }

  // Validate basic properties of dot dimension numbers.
  TF_RETURN_IF_ERROR(ValidateDotDimensionNumbers(lhs, rhs, dimension_numbers));

  // Check that there is only one contracting dimension for both lhs and rhs.
  if (dimension_numbers.lhs_contracting_dimensions_size() !=
          dimension_numbers.rhs_contracting_dimensions_size() ||
      dimension_numbers.lhs_contracting_dimensions_size() != 1) {
    return fail("Must specify one contracting dimension for both lhs and rhs.");
  }

  // Check that contracting dimension sizes match.
  const int64 lhs_contracting_dimension =
      dimension_numbers.lhs_contracting_dimensions(0);
  const int64 rhs_contracting_dimension =
      dimension_numbers.rhs_contracting_dimensions(0);
  if (lhs.dimensions(lhs_contracting_dimension) !=
      rhs.dimensions(rhs_contracting_dimension)) {
    return fail("Contracting dimension sizes do not match.");
  }

  // Check that number of batch dimensions match.
  if (dimension_numbers.lhs_batch_dimensions_size() !=
      dimension_numbers.rhs_batch_dimensions_size()) {
    return fail("Must the same number of batch dimensions for lhs and rhs.");
  }

  // Check that batch dimension numbers and sizes match.
  for (int64 i = 0; i < dimension_numbers.lhs_batch_dimensions_size(); ++i) {
    if (dimension_numbers.lhs_batch_dimensions(i) !=
            dimension_numbers.rhs_batch_dimensions(i) ||
        lhs.dimensions(dimension_numbers.lhs_batch_dimensions(i)) !=
            rhs.dimensions(dimension_numbers.rhs_batch_dimensions(i))) {
      return fail("Batch dimension numbers and sizes must match for lhs/rhs.");
    }
  }

  // The ranks of lhs and rhs are decremented by 1 respectively due to the
  // contraction, and added for the rank of the result. When an input tensor is
  // a scalar, its contribution to the rank of the result is 0.
  // Generate the result dimensions in order, rhs dimensions followed by lhs
  // dimensions except the contracted and batch dimensions.
  std::vector<int64> dimensions;
  std::unordered_set<int64> rhs_batch_dims(
      dimension_numbers.rhs_batch_dimensions().begin(),
      dimension_numbers.rhs_batch_dimensions().end());
  for (int64 i = 0; i < ShapeUtil::Rank(lhs); i++) {
    if (i != lhs_contracting_dimension) {
      dimensions.push_back(lhs.dimensions(i));
    }
  }
  for (int64 i = 0; i < ShapeUtil::Rank(rhs); i++) {
    if (i != rhs_contracting_dimension && rhs_batch_dims.count(i) == 0) {
      dimensions.push_back(rhs.dimensions(i));
    }
  }
  Shape result = ShapeUtil::MakeShape(
      ShapeUtil::HigherPrecisionElementType(lhs, rhs), dimensions);

  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(result));
  VLOG(2) << "inferred dot shape: " << ShapeUtil::HumanString(result);
  return result;
}

/* static */ StatusOr<Shape>
ShapeInference::InferDegenerateDimensionBroadcastShape(HloOpcode operation,
                                                       const Shape& lhs,
                                                       const Shape& rhs) {
  TF_RET_CHECK(ShapeUtil::Rank(lhs) == ShapeUtil::Rank(rhs));

  // The shapes have to be compatible. That is, if some dimension d has a
  // different size in the two shapes, one of them has to be 1 (a "degenerate"
  // dimension). In that case, the output shape has the non-1 dimension size
  // from the lhs/rhs pair in every index.
  std::vector<int64> output_dimensions(ShapeUtil::Rank(lhs));
  for (int64 i = 0; i < ShapeUtil::Rank(lhs); ++i) {
    if (lhs.dimensions(i) == rhs.dimensions(i)) {
      output_dimensions[i] = lhs.dimensions(i);
    } else if (lhs.dimensions(i) == 1) {
      output_dimensions[i] = rhs.dimensions(i);
    } else if (rhs.dimensions(i) == 1) {
      output_dimensions[i] = lhs.dimensions(i);
    } else {
      return InvalidArgument(
          "Binary op %s with incompatible shapes: %s and %s.",
          HloOpcodeString(operation), ShapeUtil::HumanString(lhs),
          ShapeUtil::HumanString(rhs));
    }
  }
  return ShapeUtil::MakeShape(ShapeUtil::HigherPrecisionElementType(lhs, rhs),
                              output_dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferInDimBroadcastShape(
    const Shape& smaller_shape, const Shape& larger_shape,
    absl::Span<const int64> broadcast_dimensions) {
  if (broadcast_dimensions.empty() && !ShapeUtil::IsScalar(smaller_shape)) {
    // Reject "magic" inference for binops on different shapes, requiring
    // the user to provide an explicit broadcast dimension in this case.
    // See b/25177275 for more details.
    return InvalidArgument("Automatic shape inference not supported: %s and %s",
                           ShapeUtil::HumanString(smaller_shape),
                           ShapeUtil::HumanString(larger_shape));
  } else if (broadcast_dimensions.size() != ShapeUtil::Rank(smaller_shape)) {
    return InvalidArgument(
        "Size of broadcast_dimensions has to match lower-rank operand's "
        "rank; "
        " lower-rank operand's rank is %d, size of broadcast_dimensions is "
        "%u.",
        ShapeUtil::Rank(smaller_shape), broadcast_dimensions.size());
  }

  // broadcast_dimensions is a sequence of dimensions; its length is equal to
  // the rank of the lower-rank operand. The lower-rank operand's dimensions
  // have to be compatible with the higher-rank operand's dimensions at indices
  // specified by broadcast_dimensions. Here compatible means the dimension
  // sizes are equal or in one of the shapes the dimension size is
  // one. Examples:
  //
  // smaller_shape   larger_shape   broadcast_dimensions   output_shape
  //   []              [2, 3]          {}                    [2, 3]
  //   [3]             [4, 3]          {1}                   [4, 3]
  //   [2, 3]          [2, 3, 4]       {0, 1}                [2, 3, 4]
  //   [2, 1]          [2, 3, 4]       {0, 2}                [2, 3, 1]
  //   [2, 3]          [2, 1, 4]       {0, 1}                [2, 3, 4]
  //
  // The column output_shape may not be the final shape of the XLA
  // operation. After the "InDim" broadcasting implemented in this function
  // expands the rank, degenerate-dimension broadcasting (implemented in
  // InferDegenerateDimensionBroadcastShape) broadcasts dimensions of size one
  // up to match the dimension size of the other operand. For example, consider
  // the row in the table above with a smaller_shape of [2, 1]. The shape
  // returned by this function is [2, 3, 1] (output_shape) however, the result
  // shape of the XLA operation is [2, 3, 4] after degenerate-dimension
  // broadcasting.
  //
  // Invalid broadcasts:
  //
  // smaller_shape=[3], larger_shape=[4, 3], broadcast_dimensions={0}
  // Reason: Dimension zero** of larger_shape (size 4) is not compatible with
  //   dimension zero of smaller_shape(size 3). **Zero here comes from the value
  //   in broadcast_dimensions.
  //
  // smaller_shape=[2, 1], larger_shape=[2, 3, 4], broadcast_dimensions={1, 2}
  // Reason: Dimension one of larger_shape (size 3) is not compatible with
  //   dimension zero of smaller_shape(size 2)

  // The output shape is initially the larger_shape. Sizes of dimensions
  // specified in broadcast_dimensions are then changed to match the
  // corresponding dimension size in smaller_shape.
  Shape output_shape(larger_shape);
  output_shape.set_element_type(
      ShapeUtil::HigherPrecisionElementType(larger_shape, smaller_shape));

  for (int i = 0; i < smaller_shape.dimensions_size(); ++i) {
    int64 dimension_to_match = broadcast_dimensions.at(i);
    if (dimension_to_match < 0) {
      return InvalidArgument(
          "Broadcast dimension number (%d) cannot be negative.",
          dimension_to_match);
    }
    if (dimension_to_match >= larger_shape.dimensions_size()) {
      return InvalidArgument(
          "Broadcast dimension number (%d) too large; higher-rank "
          "operand has rank %d.",
          dimension_to_match, larger_shape.dimensions_size());
    }
    int64 small_dimension_size = smaller_shape.dimensions(i);
    int64 large_dimension_size = larger_shape.dimensions(dimension_to_match);
    // Dimension sizes must be compatible: match or be degenerate (degenerate
    // case is handled by degenerate dimension broadcasting which occurs after
    // InDim broadcasting).
    if (small_dimension_size != large_dimension_size &&
        small_dimension_size != 1 && large_dimension_size != 1) {
      return InvalidArgument(
          "Broadcast dimension %d mismatch: %d != %d; %s and %s.", i,
          small_dimension_size, large_dimension_size,
          ShapeUtil::HumanString(smaller_shape),
          ShapeUtil::HumanString(larger_shape));
    }
    // Make sure the broadcast dimensions are listed in a strictly increasing
    // order.
    if (i > 0 && broadcast_dimensions.at(i - 1) >= dimension_to_match) {
      return InvalidArgument(
          "Broadcast dimensions order is wrong: %d comes after %d.",
          dimension_to_match, broadcast_dimensions.at(i - 1));
    }

    output_shape.set_dimensions(dimension_to_match, small_dimension_size);
  }

  return output_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferElementwiseBinaryOpShape(
    HloOpcode operation, const Shape& lhs, const Shape& rhs,
    absl::Span<const int64> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(lhs, "lhs of elementwise binary operation"));
  TF_RETURN_IF_ERROR(ExpectArray(rhs, "rhs of elementwise binary operation"));

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) {
    return InvalidArgument(
        "Binary op %s with different element types: %s and %s.",
        HloOpcodeString(operation), ShapeUtil::HumanString(lhs),
        ShapeUtil::HumanString(rhs));
  }

  if (ShapeUtil::Rank(lhs) == ShapeUtil::Rank(rhs)) {
    std::vector<int64> identity_dims(ShapeUtil::Rank(lhs));
    std::iota(identity_dims.begin(), identity_dims.end(), 0);
    if (!broadcast_dimensions.empty() &&
        broadcast_dimensions != identity_dims) {
      return InvalidArgument(
          "Broadcast dimensions field must either be not set or be the "
          "identity on binary operations with operands of the same rank.");
    }
  }

  if (ShapeUtil::CompatibleIgnoringFpPrecision(lhs, rhs)) {
    // If the shapes are the same other than layout, the output shape is the
    // same (elementwise op).
    return ShapeUtil::ChangeElementType(
        lhs, ShapeUtil::HigherPrecisionElementType(lhs, rhs));
  }

  if (ShapeUtil::Rank(lhs) == ShapeUtil::Rank(rhs)) {
    return InferDegenerateDimensionBroadcastShape(operation, lhs, rhs);
  } else {
    // Ranks do not match, so perform InDim broadcasting using
    // broadcast_dimensions. Scalar broadcasting is a special case of this.
    const Shape& larger_shape =
        ShapeUtil::Rank(lhs) > ShapeUtil::Rank(rhs) ? lhs : rhs;
    const Shape& smaller_shape =
        ShapeUtil::Rank(lhs) > ShapeUtil::Rank(rhs) ? rhs : lhs;

    // After InDim broadcasting, perform degenerate dimensions broadcasting.
    TF_ASSIGN_OR_RETURN(Shape indim_broadcast_shape,
                        InferInDimBroadcastShape(smaller_shape, larger_shape,
                                                 broadcast_dimensions));

    return InferDegenerateDimensionBroadcastShape(
        operation, indim_broadcast_shape, larger_shape);
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferBinaryOpShape(
    HloOpcode opcode, const HloInstruction* lhs, const HloInstruction* rhs) {
  return InferBinaryOpShape(opcode, lhs->shape(), rhs->shape(),
                            /*broadcast_dimensions=*/{});
}

/* static */ StatusOr<Shape> ShapeInference::InferBinaryOpShape(
    HloOpcode opcode, const Shape& lhs, const Shape& rhs,
    absl::Span<const int64> broadcast_dimensions) {
  VLOG(2) << StrFormat(
      "inferring shape for <%s>(%s, %s) with broadcast_dimensions={%s}",
      HloOpcodeString(opcode), ShapeUtil::HumanString(lhs),
      ShapeUtil::HumanString(rhs), StrJoin(broadcast_dimensions, ", "));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));

  TF_RETURN_IF_ERROR(ExpectArray(
      lhs, absl::StrCat("lhs of binary operation ", HloOpcodeString(opcode))));
  TF_RETURN_IF_ERROR(ExpectArray(
      rhs, absl::StrCat("rhs of binary operation ", HloOpcodeString(opcode))));
  switch (opcode) {
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
      return InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                           broadcast_dimensions);

    case HloOpcode::kSubtract:
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kPower:
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
    case HloOpcode::kMultiply:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      if (lhs.element_type() == PRED || rhs.element_type() == PRED) {
        return InvalidArgument(
            "Expected element type in shape to be arithmetic type for "
            "operation %s; got PRED.",
            HloOpcodeString(opcode));
      }
      return InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                           broadcast_dimensions);

    case HloOpcode::kComplex: {
      if (!ShapeUtil::ElementIsFloating(lhs)) {
        return InvalidArgument(
            "Expected element type in shape to be floating for complex compose "
            "operation; got %s.",
            PrimitiveType_Name(lhs.element_type()));
      }
      TF_ASSIGN_OR_RETURN(const Shape& shape,
                          InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                                        broadcast_dimensions));
      if (lhs.element_type() == F32 && rhs.element_type() == F32) {
        return ShapeUtil::ChangeElementType(shape, C64);
      } else {
        return Unimplemented("Complex component type is not implemented.");
      }
    }
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      if (lhs.element_type() != PRED &&
          !primitive_util::IsIntegralType(lhs.element_type())) {
        return InvalidArgument(
            "Expected pred or integral type in argument to and/or operation; "
            "got %s.",
            PrimitiveType_Name(lhs.element_type()));
      }
      return InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                           broadcast_dimensions);
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe: {
      TF_ASSIGN_OR_RETURN(const Shape& shape,
                          InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                                        broadcast_dimensions));
      return ShapeUtil::ChangeElementType(shape, PRED);
    }
    default:
      return Unimplemented(
          "Binary op shape inference: %s; lhs: %s; rhs: %s is not implemented.",
          HloOpcodeString(opcode), lhs.ShortDebugString(),
          rhs.ShortDebugString());
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferTernaryOpShape(
    HloOpcode opcode, const HloInstruction* lhs, const HloInstruction* rhs,
    const HloInstruction* ehs) {
  return InferTernaryOpShape(opcode, lhs->shape(), rhs->shape(), ehs->shape());
}

/* static */ StatusOr<Shape> ShapeInference::InferTernaryOpShape(
    HloOpcode opcode, const Shape& lhs, const Shape& rhs, const Shape& ehs) {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(ehs));
  switch (opcode) {
    case HloOpcode::kClamp:
      return InferClampShape(lhs, rhs, ehs);
    case HloOpcode::kSelect:
      return InferSelectShape(lhs, rhs, ehs);
    case HloOpcode::kTupleSelect:
      return InferTupleSelectShape(lhs, rhs, ehs);
    default:
      return InvalidArgument("Unknown operation %s.", HloOpcodeString(opcode));
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferVariadicOpShape(
    HloOpcode opcode, absl::Span<const HloInstruction* const> operands) {
  std::vector<const Shape*> operand_shapes;
  operand_shapes.reserve(operands.size());
  for (const HloInstruction* operand : operands) {
    operand_shapes.push_back(&operand->shape());
  }
  return InferVariadicOpShape(opcode, operand_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferVariadicOpShape(
    HloOpcode opcode, absl::Span<const Shape* const> operand_shapes) {
  for (const Shape* shape : operand_shapes) {
    TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(*shape));
  }
  switch (opcode) {
    case HloOpcode::kTuple: {
      Shape result = ShapeUtil::MakeTupleShape({});
      result.mutable_tuple_shapes()->reserve(operand_shapes.size());
      for (const Shape* shape : operand_shapes) {
        ShapeUtil::AppendShapeToTuple(*shape, &result);
      }
      return result;
    }
    case HloOpcode::kSort: {
      if (operand_shapes.size() == 1) {
        return *operand_shapes[0];
      } else {
        for (int64 operand = 1; operand < operand_shapes.size(); ++operand) {
          if (!ShapeUtil::SameDimensions(*operand_shapes[0],
                                         *operand_shapes[operand])) {
            return InvalidArgument(
                "Sort keys and values dimensions must match. "
                "Keys shape is: %s\n, Values shape (operand index %lld) is: %s",
                ShapeUtil::HumanString(*operand_shapes[0]), operand,
                ShapeUtil::HumanString(*operand_shapes[operand]));
          }
        }
        std::vector<Shape> operand_shape_values;
        for (const Shape* operand_shape : operand_shapes) {
          operand_shape_values.push_back(*operand_shape);
        }
        return ShapeUtil::MakeTupleShape(operand_shape_values);
      }
      return InvalidArgument("Unexpected number of operands for sort");
    }
    default:
      return InvalidArgument("Unknown operation %s.", HloOpcodeString(opcode));
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferMapShape(
    absl::Span<const Shape* const> arg_shapes, const ProgramShape& to_apply,
    absl::Span<const int64> dimensions) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Map expects at least one argument.");
  }

  // All arguments must have the same shape.
  const Shape* arg_shape = arg_shapes[0];
  for (size_t i = 1; i < arg_shapes.size(); ++i) {
    TF_RETURN_IF_ERROR(ExpectArray(*arg_shapes[i], "operand of map"));

    if (ShapeUtil::CompatibleIgnoringFpPrecision(*arg_shapes[i], *arg_shape)) {
      continue;
    }
    if (ShapeUtil::SameElementTypeIgnoringFpPrecision(*arg_shapes[i],
                                                      *arg_shape)) {
      if (ShapeUtil::IsScalar(*arg_shapes[i])) {
        continue;
      }
      if (ShapeUtil::IsScalar(*arg_shape)) {
        arg_shape = arg_shapes[i];
        continue;
      }
    }

    std::vector<string> pieces;
    for (const Shape* shape : arg_shapes) {
      pieces.push_back(ShapeUtil::HumanString(*shape));
    }
    return InvalidArgument(
        "Map operation requires all operands to have the same shape; got: "
        "%s.",
        StrJoin(pieces, ", "));
  }

  // Check that dimensions.size == arg_shape.dimensions_size() (we currently
  // only support mapping across all dimensions: i.e. scalar map functions).
  if (dimensions.size() != arg_shape->dimensions_size()) {
    return InvalidArgument(
        "Map applied to a subset of dimensions currently not supported: "
        "arg_dimension_size: %d, requested_map_dimensions_size: %u.",
        arg_shape->dimensions_size(), dimensions.size());
  }

  // Check that requested map dimensions numbers are monotonically increasing.
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] != i) {
      return InvalidArgument(
          "Map requires monotonically increasing dimension numbers; got: %s.",
          StrJoin(dimensions, ", "));
    }
  }

  // The applied function's arity equals the number of arguments.
  if (arg_shapes.size() != to_apply.parameters_size()) {
    return InvalidArgument(
        "Map applied function arity must match number of arguments; got: "
        "arity: %d, arguments: %u.",
        to_apply.parameters_size(), arg_shapes.size());
  }

  // The parameters should all be scalars, and the output too.
  const Shape& output_shape = to_apply.result();
  if (!ShapeUtil::IsScalar(output_shape)) {
    return InvalidArgument(
        "Mapped computation's result has to be a scalar; got: %s.",
        ShapeUtil::HumanString(output_shape));
  }

  for (int i = 0; i < to_apply.parameters_size(); ++i) {
    const Shape& parameter_shape = to_apply.parameters(i);

    if (!ShapeUtil::IsScalar(parameter_shape)) {
      return InvalidArgument(
          "Mapped computation's parameter has to be a scalar; "
          "got parameter %d shape: %s.",
          i, ShapeUtil::HumanString(parameter_shape));
    }

    if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(parameter_shape,
                                                       *arg_shape)) {
      return InvalidArgument(
          "Mapped computation's parameter type has to match argument element "
          "type; got parameter %d shape: %s, argument shape: %s.",
          i, ShapeUtil::HumanString(parameter_shape),
          ShapeUtil::HumanString(*arg_shape));
    }
  }

  return ShapeUtil::MakeShape(output_shape.element_type(),
                              AsInt64Slice(arg_shape->dimensions()));
}

/* static */ StatusOr<Shape> ShapeInference::InferBatchNormTrainingShape(
    const Shape& operand_shape, const Shape& scale_shape,
    const Shape& offset_shape, int64 feature_index) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand of batch norm training"));
  TF_RETURN_IF_ERROR(
      ExpectArray(offset_shape, "offset input of batch norm training"));
  TF_RETURN_IF_ERROR(
      ExpectArray(scale_shape, "scale input of batch norm training"));

  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(operand_shape) ==
               Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(offset_shape) ==
               Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(scale_shape) ==
               Status::OK());

  if (feature_index >= ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-training to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %d, and rank %d.",
        feature_index, ShapeUtil::Rank(operand_shape));
  }

  if (feature_index < 0) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-training to "
        "be a non-negative number, got %d.",
        feature_index);
  }

  if (ShapeUtil::Rank(operand_shape) < 1) {
    return InvalidArgument(
        "Expected the rank of operand to "
        "batch-norm-training to be at least 1; got %d.",
        ShapeUtil::Rank(operand_shape));
  }

  if (ShapeUtil::Rank(offset_shape) != 1) {
    return InvalidArgument(
        "Offset input of batch-norm-training must have"
        " rank 1, but has rank %d.",
        ShapeUtil::Rank(offset_shape));
  }

  if (ShapeUtil::Rank(scale_shape) != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-training must have"
        " rank 1, but has rank %d.",
        ShapeUtil::Rank(scale_shape));
  }

  if (!ShapeUtil::ElementIsFloating(operand_shape)) {
    return InvalidArgument(
        "The operand to batch-norm-training must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(offset_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-training, "
        "but the shape of offset factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(offset_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(scale_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-training, "
        "but the shape of scale factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(scale_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  const int64 feature_count = operand_shape.dimensions(feature_index);
  Shape output_shape_for_mean_and_var =
      ShapeUtil::MakeShape(operand_shape.element_type(), {feature_count});

  if (ShapeUtil::GetDimension(offset_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of offset factor should be the same as feature count,"
        "but the size of offset factor is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(offset_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(scale_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of scale factor should be the same as feature count,"
        "but the size of scale factor is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(scale_shape, 0), feature_count);
  }

  return ShapeUtil::MakeTupleShape({operand_shape,
                                    output_shape_for_mean_and_var,
                                    output_shape_for_mean_and_var});
}

/* static */ StatusOr<Shape> ShapeInference::InferBatchNormInferenceShape(
    const Shape& operand_shape, const Shape& scale_shape,
    const Shape& offset_shape, const Shape& mean_shape,
    const Shape& variance_shape, int64 feature_index) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand of batch norm inference"));
  TF_RETURN_IF_ERROR(
      ExpectArray(offset_shape, "offset input of batch norm inference"));
  TF_RETURN_IF_ERROR(
      ExpectArray(scale_shape, "scale input of batch norm inference"));

  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(operand_shape) ==
               Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(offset_shape) ==
               Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(scale_shape) ==
               Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(mean_shape) ==
               Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(variance_shape) ==
               Status::OK());

  if (feature_index >= ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-inference to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %d, and rank %d.",
        feature_index, ShapeUtil::Rank(operand_shape));
  }

  if (feature_index < 0) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-inference to "
        "be a non-negative number, got %d.",
        feature_index);
  }

  if (ShapeUtil::Rank(operand_shape) < 1) {
    return InvalidArgument(
        "Expected the rank of operand to "
        "batch-norm-inference to be at least 1; got %d.",
        ShapeUtil::Rank(operand_shape));
  }

  if (ShapeUtil::Rank(offset_shape) != 1) {
    return InvalidArgument(
        "Offset input of batch-norm-inference must have"
        " rank 1, but has rank %d.",
        ShapeUtil::Rank(offset_shape));
  }

  if (ShapeUtil::Rank(scale_shape) != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-inference must have"
        " rank 1, but has rank %d.",
        ShapeUtil::Rank(scale_shape));
  }

  if (!ShapeUtil::ElementIsFloating(operand_shape)) {
    return InvalidArgument(
        "The operand to batch-norm-inference must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(offset_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of offset factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(offset_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(scale_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of scale factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(scale_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(mean_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of mean is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(variance_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of variance is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()),
        PrimitiveType_Name(variance_shape.element_type()));
  }

  const int64 feature_count = operand_shape.dimensions(feature_index);
  Shape output_shape_for_mean_and_var =
      ShapeUtil::MakeShape(operand_shape.element_type(), {feature_count});

  if (ShapeUtil::GetDimension(offset_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of offset factor should be the same as feature count,"
        "but the size of offset factor is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(offset_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(scale_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of scale factor should be the same as feature count,"
        "but the size of scale factor is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(scale_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(mean_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of mean should be the same as feature count,"
        "but the size of mean is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(mean_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(variance_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of variance should be the same as feature count,"
        "but the size of variance is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(variance_shape, 0), feature_count);
  }

  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferBatchNormGradShape(
    const Shape& operand_shape, const Shape& scale_shape,
    const Shape& mean_shape, const Shape& var_shape,
    const Shape& output_grad_shape, int64 feature_index) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of batch norm grad"));
  TF_RETURN_IF_ERROR(
      ExpectArray(scale_shape, "scale input of batch norm grad"));
  TF_RETURN_IF_ERROR(ExpectArray(mean_shape, "mean input of batch norm grad"));
  TF_RETURN_IF_ERROR(ExpectArray(var_shape, "var input of batch norm grad"));
  TF_RETURN_IF_ERROR(
      ExpectArray(output_grad_shape, "output_grad input of batch norm grad"));

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(operand_shape));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(mean_shape));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(scale_shape));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(var_shape));
  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(output_grad_shape));

  if (feature_index >= ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-grad to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %d, and rank %d.",
        feature_index, ShapeUtil::Rank(operand_shape));
  }

  if (ShapeUtil::Rank(operand_shape) != ShapeUtil::Rank(output_grad_shape)) {
    return InvalidArgument(
        "Expected operand_shape of batch-norm-grad to have the same rank as"
        " output_grad_shape; got rank(oprand_shape) %d, and"
        " rank(output_grad_shape) %d.",
        ShapeUtil::Rank(operand_shape), ShapeUtil::Rank(output_grad_shape));
  }

  if (ShapeUtil::Rank(mean_shape) != 1) {
    return InvalidArgument(
        "Mean input of batch-norm-grad must have"
        " rank 1, but has rank %d.",
        ShapeUtil::Rank(mean_shape));
  }

  if (ShapeUtil::Rank(scale_shape) != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-grad must have"
        " rank 1, but has rank %d.",
        ShapeUtil::Rank(scale_shape));
  }

  if (ShapeUtil::Rank(var_shape) != 1) {
    return InvalidArgument(
        "Var input of batch-norm-grad must have"
        " rank 1, but has rank %d.",
        ShapeUtil::Rank(var_shape));
  }

  if (!ShapeUtil::ElementIsFloating(operand_shape)) {
    return InvalidArgument(
        "The operand to batch-norm-grad must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::ElementIsFloating(output_grad_shape)) {
    return InvalidArgument(
        "The output_grad to batch-norm-grad must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(output_grad_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(output_grad_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of output_grad is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(output_grad_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(scale_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of scale factor is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(scale_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(mean_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of mean is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(var_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of mean is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()),
        PrimitiveType_Name(operand_shape.element_type()));
  }

  const int64 feature_count = operand_shape.dimensions(feature_index);

  Shape feature_shape =
      ShapeUtil::MakeShape(operand_shape.element_type(), {feature_count});

  if (ShapeUtil::GetDimension(mean_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of mean should be the same as feature count,"
        "but the size of offset factor is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(mean_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(scale_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of scale factor should be the same as feature count,"
        "but the size of scale factor is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(scale_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(var_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of variance should be the same as feature count,"
        "but the size of variance is %d "
        "and the feature count is %d.",
        ShapeUtil::GetDimension(var_shape, 0), feature_count);
  }

  // Verify operand_shape and output_grad_shape have same bounds.
  for (int64 i = 0; i < ShapeUtil::Rank(operand_shape); ++i) {
    if (ShapeUtil::GetDimension(operand_shape, i) !=
        ShapeUtil::GetDimension(output_grad_shape, i)) {
      return InvalidArgument(
          "The bounds of operand shape should be the same as output_grad's,"
          "but the bound of operand_shape at dimension %d is %d "
          "and the bound of output_grad_shape is %d.",
          i, ShapeUtil::GetDimension(operand_shape, i),
          ShapeUtil::GetDimension(output_grad_shape, i));
    }
  }

  return ShapeUtil::MakeTupleShape(
      {operand_shape, feature_shape, feature_shape});
}

/* static */ StatusOr<Shape> ShapeInference::InferConvolveShape(
    const Shape& lhs, const Shape& rhs, int64 feature_group_count,
    int64 batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dnums) {
  TF_RETURN_IF_ERROR(ExpectArray(lhs, "lhs of convolution"));
  TF_RETURN_IF_ERROR(ExpectArray(rhs, "rhs of convolution"));

  if (feature_group_count <= 0) {
    return InvalidArgument(
        "feature_group_count must be a positive number, got %d",
        feature_group_count);
  }

  if (batch_group_count <= 0) {
    return InvalidArgument(
        "batch_group_count must be a positive number, got %d",
        batch_group_count);
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) {
    return InvalidArgument(
        "Convolution with different element types: %s and %s.",
        ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs));
  }
  if (dnums.input_spatial_dimensions_size() !=
      dnums.kernel_spatial_dimensions_size()) {
    return InvalidArgument(
        "Both arguments to convolution must have same number of dimensions.\n"
        "Numbers: %s",
        dnums.DebugString());
  }

  if (dnums.input_spatial_dimensions_size() !=
      dnums.output_spatial_dimensions_size()) {
    return InvalidArgument(
        "Both input and output of convolution must have same number of "
        "dimensions.\nNumbers: %s",
        dnums.DebugString());
  }

  const int num_spatial_dims = dnums.input_spatial_dimensions_size();
  if (window.dimensions_size() != num_spatial_dims) {
    return InvalidArgument(
        "Window must have same number of dimensions as dimension numbers.\n"
        "Window: %s\nDimension numbers: %s.",
        window.DebugString(), dnums.DebugString());
  }

  const int num_dims = num_spatial_dims + 2;
  if (ShapeUtil::Rank(lhs) != num_dims) {
    return InvalidArgument(
        "The LHS argument to a convolution should have rank %d; lhs: %s.",
        num_dims, ShapeUtil::HumanString(lhs));
  }
  if (ShapeUtil::Rank(rhs) != num_dims) {
    return InvalidArgument(
        "The RHS argument to a convolution should have rank %d; rhs: %s.",
        num_dims, ShapeUtil::HumanString(rhs));
  }
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));

  // Verifies that the input and window dimensions are a permutation of
  // the dimension numbers.
  std::vector<int64> input_dnums(num_dims);
  input_dnums[0] = dnums.input_batch_dimension();
  input_dnums[1] = dnums.input_feature_dimension();
  std::copy(dnums.input_spatial_dimensions().begin(),
            dnums.input_spatial_dimensions().end(), input_dnums.begin() + 2);
  std::sort(input_dnums.begin(), input_dnums.end());

  std::vector<int64> window_dnums(num_dims);
  window_dnums[0] = dnums.kernel_input_feature_dimension();
  window_dnums[1] = dnums.kernel_output_feature_dimension();
  std::copy(dnums.kernel_spatial_dimensions().begin(),
            dnums.kernel_spatial_dimensions().end(), window_dnums.begin() + 2);
  std::sort(window_dnums.begin(), window_dnums.end());

  std::vector<int64> output_dnums(num_dims);
  output_dnums[0] = dnums.output_batch_dimension();
  output_dnums[1] = dnums.output_feature_dimension();
  std::copy(dnums.output_spatial_dimensions().begin(),
            dnums.output_spatial_dimensions().end(), output_dnums.begin() + 2);
  std::sort(output_dnums.begin(), output_dnums.end());

  std::vector<int64> expected_dnums(num_dims);
  std::iota(expected_dnums.begin(), expected_dnums.end(), 0);

  const auto in_range = [num_dims](int64 i) { return 0 <= i && i < num_dims; };
  if (!std::all_of(input_dnums.begin(), input_dnums.end(), in_range) ||
      !std::all_of(window_dnums.begin(), window_dnums.end(), in_range) ||
      !std::all_of(output_dnums.begin(), output_dnums.end(), in_range)) {
    return InvalidArgument(
        "A dimension number is out of range in convolution: %s.",
        dnums.DebugString());
  }

  if (input_dnums != expected_dnums) {
    return InvalidArgument(
        "Input dimensions of convolution must contain each dimension exactly "
        "once: %s.",
        dnums.DebugString());
  }
  if (window_dnums != expected_dnums) {
    return InvalidArgument(
        "Window dimensions of convolution must contain each dimension exactly "
        "once: %s.",
        dnums.DebugString());
  }
  if (output_dnums != expected_dnums) {
    return InvalidArgument(
        "Output dimensions of convolution must contain each dimension exactly "
        "once: %s.",
        dnums.DebugString());
  }

  std::vector<int64> input_spatial_dims(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_spatial_dims[i] = lhs.dimensions(dnums.input_spatial_dimensions(i));
  }
  const int64 input_features = lhs.dimensions(dnums.input_feature_dimension());
  const int64 input_batch = lhs.dimensions(dnums.input_batch_dimension());

  std::vector<int64> kernel_spatial_dims(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    kernel_spatial_dims[i] = rhs.dimensions(dnums.kernel_spatial_dimensions(i));
  }
  const int64 kernel_input_features =
      rhs.dimensions(dnums.kernel_input_feature_dimension());
  const int64 kernel_output_features =
      rhs.dimensions(dnums.kernel_output_feature_dimension());

  if (input_features % feature_group_count != 0 ||
      input_features / feature_group_count != kernel_input_features) {
    return InvalidArgument(
        "Expected LHS feature dimension (value %d) to be a multiple of "
        "feature_group_count (value %d), and LHS feature dimension / "
        "feature_group_count = RHS feature dimension (value %d); "
        "got <conv>(%s, %s)\n"
        "Dimension numbers: {%s}.",
        input_features, feature_group_count, kernel_input_features,
        ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs),
        dnums.DebugString());
  }
  if (kernel_output_features % feature_group_count > 0) {
    return InvalidArgument(
        "Expected output feature dimension (value %d) to be divisible by "
        "feature_group_count (value %d); "
        "got <conv>(%s, %s)\n"
        "Dimension numbers: {%s}.",
        kernel_output_features, feature_group_count,
        ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs),
        dnums.DebugString());
  }

  if (input_batch % batch_group_count > 0) {
    return InvalidArgument(
        "Expected input batch dimension (value %d) to be divisible by "
        "batch_group_count (value %d); "
        "got <conv>(%s, %s)\n"
        "Dimension numbers: {%s}.",
        input_batch, batch_group_count, ShapeUtil::HumanString(lhs),
        ShapeUtil::HumanString(rhs), dnums.DebugString());
  }

  std::vector<int64> window_dims(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    window_dims[i] = window.dimensions(i).size();
  }
  if (kernel_spatial_dims != window_dims) {
    return InvalidArgument(
        "Window dimensions do not match RHS shape:\n\t"
        "RHS shape: %s\n\t"
        "Window: {%s}\n\t"
        "Dimension numbers: {%s}.",
        ShapeUtil::HumanString(rhs), window.ShortDebugString(),
        dnums.ShortDebugString());
  }

  Shape base_shape =
      ShapeUtil::MakeShape(lhs.element_type(), input_spatial_dims);
  TF_ASSIGN_OR_RETURN(
      Shape window_output_shape,
      InferWindowOutputShape(base_shape, window, lhs.element_type(),
                             /*allow_negative_padding=*/true));

  std::vector<int64> dimensions(num_dims);
  dimensions[dnums.output_batch_dimension()] = input_batch / batch_group_count;
  dimensions[dnums.output_feature_dimension()] = kernel_output_features;
  for (int i = 0; i < num_spatial_dims; ++i) {
    dimensions[dnums.output_spatial_dimensions(i)] =
        window_output_shape.dimensions(i);
  }
  return ShapeUtil::MakeShape(ShapeUtil::HigherPrecisionElementType(lhs, rhs),
                              dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferFftShape(
    const Shape& in, const FftType fft_type,
    const absl::Span<const int64> fft_length) {
  const int64 fft_rank = fft_length.size();
  if (fft_rank < 1 || fft_rank > 3) {
    return InvalidArgument("FFT only supports ranks 1-3; got %d.", fft_rank);
  }
#define RET_CHECK_RANK(x)                            \
  if (x.dimensions_size() < fft_rank) {              \
    return InvalidArgument(                          \
        "FFT of rank %d requires input of at least " \
        "same rank; got input of rank %d",           \
        fft_rank, x.dimensions_size());              \
  }
  switch (fft_type) {
    case FFT:
    case IFFT:
      if (in.element_type() != C64) {
        return InvalidArgument("%s requires C64 input type, found %s.",
                               FftType_Name(fft_type),
                               PrimitiveType_Name(in.element_type()));
      }
      RET_CHECK_RANK(in);
      return in;
    case RFFT: {
      if (in.element_type() != F32) {
        return InvalidArgument("RFFT requires F32 input type, found %s.",
                               PrimitiveType_Name(in.element_type()));
      }
      RET_CHECK_RANK(in);
      for (int i = 0; i < fft_rank; i++) {
        if (in.dimensions(in.dimensions_size() - fft_rank + i) !=
            fft_length[i]) {
          return InvalidArgument(
              "RFFT requires innermost dimensions match fft_length but "
              "dimension %d is %d and should be %d.",
              in.dimensions_size() - fft_rank + i,
              in.dimensions(in.dimensions_size() - fft_rank + i),
              fft_length[i]);
        }
      }
      Shape result = ShapeUtil::ChangeElementType(in, C64);
      result.set_dimensions(result.dimensions_size() - 1,
                            fft_length[fft_rank - 1] / 2 + 1);
      return result;
    }
    case IRFFT: {
      if (in.element_type() != C64) {
        return InvalidArgument("IRFFT requires C64 input type, found %s.",
                               PrimitiveType_Name(in.element_type()));
      }
      RET_CHECK_RANK(in);
      Shape result = ShapeUtil::ComplexComponentShape(in);
      for (int i = 0; i < fft_rank - 1; i++) {
        if (in.dimensions(in.dimensions_size() - fft_rank + i) !=
            fft_length[i]) {
          return InvalidArgument(
              "IRFFT requires all but one innermost dimensions match "
              "fft_length, but dimension %d is %d and should be %d.",
              in.dimensions_size() - fft_rank + i,
              in.dimensions(in.dimensions_size() - fft_rank + i),
              fft_length[i]);
        }
      }
      if (in.dimensions(in.dimensions_size() - 1) !=
          fft_length[fft_rank - 1] / 2 + 1) {
        return InvalidArgument(
            "IRFFT requires innermost dimension matches fft_length/2+1, but "
            "dimension %d is %d and should be %d.",
            in.dimensions_size() - 1, in.dimensions(in.dimensions_size() - 1),
            fft_length[fft_rank - 1] / 2 + 1);
      }
      result.set_dimensions(result.dimensions_size() - 1,
                            fft_length[fft_rank - 1]);
      return result;
    }
    default:
      LOG(FATAL) << "Unexpected fft_type: " << fft_type;
  }
#undef RET_CHECK_RANK
}

/* static */ StatusOr<Shape> ShapeInference::InferAllReduceShape(
    absl::Span<const Shape* const> operand_shapes) {
  for (const Shape* operand_shape : operand_shapes) {
    TF_RETURN_IF_ERROR(
        ExpectArray(*operand_shape, "operand of cross replica sum"));
  }
  if (operand_shapes.size() == 1) {
    return *operand_shapes[0];
  }
  std::vector<Shape> operand_shape_values;
  for (const Shape* operand_shape : operand_shapes) {
    operand_shape_values.push_back(*operand_shape);
  }
  return ShapeUtil::MakeTupleShape(operand_shape_values);
}

/* static */ StatusOr<Shape> ShapeInference::InferAllToAllShape(
    const Shape& shape, int64 split_dimension, int64 concat_dimension,
    int64 split_count) {
  TF_RET_CHECK(split_count > 0);
  if (split_dimension >= ShapeUtil::Rank(shape) || split_dimension < 0) {
    return InvalidArgument(
        "AllToAll split_dimension %d is out-of-bounds in shape %s.",
        split_dimension, ShapeUtil::HumanString(shape));
  }
  if (concat_dimension >= ShapeUtil::Rank(shape) || concat_dimension < 0) {
    return InvalidArgument(
        "AllToAll concat_dimension %d is out-of-bounds in shape %s.",
        concat_dimension, ShapeUtil::HumanString(shape));
  }
  if (shape.dimensions(split_dimension) % split_count != 0) {
    return InvalidArgument(
        "AllToAll split dimension size %d must be dividable by split_count "
        "%d.",
        shape.dimensions(split_dimension), split_count);
  }
  std::vector<int64> new_dimensions(shape.dimensions().begin(),
                                    shape.dimensions().end());
  new_dimensions[split_dimension] /= split_count;
  new_dimensions[concat_dimension] *= split_count;
  return ShapeUtil::MakeShape(shape.element_type(), new_dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferAllToAllTupleShape(
    absl::Span<const Shape* const> operand_shapes) {
  // An Alltoall HLO instruction receives N operands (with the same shape) and
  // returns a tuple that contains N array shapes.
  TF_RET_CHECK(!operand_shapes.empty());
  for (int i = 0; i < operand_shapes.size(); i++) {
    if (!ShapeUtil::Equal(*operand_shapes[0], *operand_shapes[i])) {
      return InvalidArgument(
          "HLO all-to-all has operands with different shapes: the 0th "
          "operand shape %s, but the %dth operand has shape %s.",
          ShapeUtil::HumanString(*operand_shapes[0]), i,
          ShapeUtil::HumanString(*operand_shapes[i]));
    }
  }

  return InferVariadicOpShape(HloOpcode::kTuple, operand_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferCollectivePermuteShape(
    const Shape& shape) {
  TF_RET_CHECK(ShapeUtil::IsArray(shape));
  return shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceShape(
    absl::Span<const Shape* const> arg_shapes,
    absl::Span<const int64> dimensions_to_reduce,
    const ProgramShape& to_apply) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Reduce must have at least 2 arguments, has 0");
  }
  if (arg_shapes.size() % 2) {
    return InvalidArgument(
        "Reduce must have an even number of arguments, has %lu",
        arg_shapes.size());
  }
  int64 num_reduced_args = arg_shapes.size() / 2;

  auto reduced_args = arg_shapes.subspan(0, num_reduced_args);
  // Check that all of the reduced tensors have the same dimensions. The element
  // types may be different.
  for (int64 i = 1; i < num_reduced_args; ++i) {
    if (!ShapeUtil::SameDimensions(*reduced_args[0], *reduced_args[i])) {
      return InvalidArgument(
          "All reduced tensors must have the sime dimension. Tensor 0 has "
          "shape %s, Tensor %d has shape %s",
          ShapeUtil::HumanString(*reduced_args[0]), i,
          ShapeUtil::HumanString(*reduced_args[i]));
    }
  }

  // Check that the dimensions to reduce are in-bounds for the given shape.
  // We've already verified all reduced tensors have the same dimensions, so it
  // doesn't matter which one we choose.
  const Shape& arg = *reduced_args[0];
  for (int64 dimension : dimensions_to_reduce) {
    if (dimension >= ShapeUtil::Rank(arg) || dimension < 0) {
      return InvalidArgument("Reducing out-of-bounds dimension %d in shape %s.",
                             dimension, ShapeUtil::HumanString(arg));
    }
  }

  auto init_values = arg_shapes.subspan(num_reduced_args, arg_shapes.size());
  std::vector<PrimitiveType> element_types;
  for (const Shape* arg : reduced_args) {
    element_types.push_back(arg->element_type());
  }
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply, init_values, element_types,
                                        num_reduced_args));

  std::set<int64> dimensions_to_reduce_set(dimensions_to_reduce.begin(),
                                           dimensions_to_reduce.end());
  std::vector<int64> new_dimensions;
  for (int i = 0; i < ShapeUtil::Rank(arg); ++i) {
    if (dimensions_to_reduce_set.find(i) == dimensions_to_reduce_set.end()) {
      new_dimensions.push_back(arg.dimensions(i));
    }
  }

  if (ShapeUtil::IsScalar(to_apply.result())) {
    return ShapeUtil::MakeShape(to_apply.result().element_type(),
                                new_dimensions);
  } else {
    std::vector<Shape> result_subshapes;
    for (const Shape& subshape : to_apply.result().tuple_shapes()) {
      result_subshapes.push_back(
          ShapeUtil::MakeShape(subshape.element_type(), new_dimensions));
    }
    return ShapeUtil::MakeTupleShape(result_subshapes);
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceWindowShape(
    const Shape& operand_shape, const Shape& init_value_shape,
    const Window& window, const ProgramShape& to_apply_shape) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of reduce-window"));
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply_shape, {&init_value_shape},
                                        {operand_shape.element_type()},
                                        /*inputs=*/1));
  return InferWindowOutputShape(operand_shape, window,
                                init_value_shape.element_type(),
                                /*allow_negative_padding=*/false);
}

/* static */ StatusOr<Shape> ShapeInference::InferSelectAndScatterShape(
    const Shape& operand_shape, const ProgramShape& select_shape,
    const Window& window, const Shape& source_shape,
    const Shape& init_value_shape, const ProgramShape& scatter_shape) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand of select-and-scatter"));

  // Check if the select function has a proper shape of (T,T) -> PRED.
  if (select_shape.parameters_size() != 2) {
    return InvalidArgument(
        "Select function must take 2 parameters, but "
        "takes %d parameter(s).",
        select_shape.parameters_size());
  }
  const Shape& select_result_shape = select_shape.result();
  if (!ShapeUtil::Compatible(select_result_shape,
                             ShapeUtil::MakeShape(PRED, {}))) {
    return InvalidArgument("Select function must have rank-0 PRED result.");
  }
  const Shape& operand_element_shape =
      ShapeUtil::MakeShape(operand_shape.element_type(), {});
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(operand_element_shape,
                                                select_shape.parameters(0))) {
    return InvalidArgument(
        "Select function's first parameter shape currently must "
        "match the operand element shape, but got %s vs %s.",
        ShapeUtil::HumanString(select_shape.parameters(0)),
        ShapeUtil::HumanString(operand_element_shape));
  }
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(operand_element_shape,
                                                select_shape.parameters(1))) {
    return InvalidArgument(
        "Select function's second parameter shape currently must "
        "match the operand element shape, but got %s vs %s.",
        ShapeUtil::HumanString(select_shape.parameters(1)),
        ShapeUtil::HumanString(operand_element_shape));
  }

  // Check if the scatter function has a proper shape as a reduction.
  TF_RETURN_IF_ERROR(VerifyReducerShape(scatter_shape, {&init_value_shape},
                                        {source_shape.element_type()},
                                        /*inputs=*/1));

  // Check if the result shape of window operation matches the source shape.
  TF_ASSIGN_OR_RETURN(const Shape& window_result_shape,
                      InferWindowOutputShape(operand_shape, window,
                                             operand_shape.element_type(),
                                             /*allow_negative_padding=*/false));
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(source_shape,
                                                window_result_shape)) {
    return InvalidArgument(
        "Source shape does not match the shape of window-reduced operand: "
        "source(%s), window-reduced operand(%s).",
        ShapeUtil::HumanString(source_shape),
        ShapeUtil::HumanString(window_result_shape));
  }
  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferGetDimensionSizeShape(
    const Shape& shape, int64 dimension) {
  if (dimension < 0 || dimension >= ShapeUtil::Rank(shape)) {
    return InvalidArgument("GetDimensionSize dimension out of bounds: %d.",
                           dimension);
  }

  // TODO(b/119580730): Remove this restriction when very large dimension size
  // is needed.
  if (shape.dimensions(dimension) > std::numeric_limits<uint32>::max()) {
    return InvalidArgument(
        "GetDimensionSize's input shape is %s, the %dth dimension exceeds the "
        "UINT_MAX limit.",
        ShapeUtil::HumanString(shape), dimension);
  }

  return ShapeUtil::MakeShape(U32, {});
}

/* static */ StatusOr<Shape> ShapeInference::InferSliceShape(
    const Shape& arg, absl::Span<const int64> starts,
    absl::Span<const int64> limits, absl::Span<const int64> strides) {
  auto error = [&](const string& message) {
    return InvalidArgument(
        "%s in slice operation; argument shape: %s; starts: {%s}; limits: "
        "{%s}; strides: {%s}.",
        message, ShapeUtil::HumanString(arg), StrJoin(starts, ","),
        StrJoin(limits, ","), StrJoin(strides, ","));
  };
  TF_RETURN_IF_ERROR(ExpectArray(arg, "operand of slice"));
  VLOG(2) << StrFormat("slicing shape %s starts={%s} limits={%s}",
                       ShapeUtil::HumanString(arg), StrJoin(starts, ", "),
                       StrJoin(limits, ", "));

  if (starts.size() != limits.size()) {
    return error(StrFormat("slice start and limit sizes differ: %u vs %u",
                           starts.size(), limits.size()));
  }

  if (starts.size() != strides.size()) {
    return error(StrFormat("slice start and strides sizes differ: %u vs %u",
                           starts.size(), strides.size()));
  }

  if (starts.size() != ShapeUtil::Rank(arg)) {
    return InvalidArgument(
        "Slice index count does not match argument rank: %u vs %d.",
        starts.size(), ShapeUtil::Rank(arg));
  }

  std::vector<int64> sizes;
  for (int64 dimension = 0; dimension < starts.size(); ++dimension) {
    int64 start_index = starts[dimension];
    int64 limit_index = limits[dimension];
    int64 stride = strides[dimension];
    if (start_index < 0) {
      return InvalidArgument("Negative start index to slice: %d.", start_index);
    }
    if (limit_index > arg.dimensions(dimension)) {
      return error(
          StrFormat("limit index (%d) must be less than or equal to dimension "
                    "size (%d)",
                    limit_index, arg.dimensions(dimension)));
    }
    VLOG(2) << StrFormat("starts[%d] = %d", dimension, start_index);
    VLOG(2) << StrFormat("limits[%d] = %d", dimension, limit_index);
    if (start_index > limit_index) {
      return error(
          StrFormat("limit index (%d) must be greater or equal to "
                    "start index (%d) in slice with positive stride",
                    limit_index, start_index));
    }
    if (stride <= 0) {
      return InvalidArgument("Stride (%d) must be positive.", stride);
    }
    sizes.push_back((limit_index - start_index + stride - 1) / stride);
  }

  return ShapeUtil::MakeShape(arg.element_type(), sizes);
}

/* static */ StatusOr<Shape> ShapeInference::InferDynamicSliceShape(
    const Shape& operand_shape, const Shape& start_indices_shape,
    absl::Span<const int64> slice_sizes) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of dynamic slice"));
  TF_RETURN_IF_ERROR(
      ExpectArray(start_indices_shape, "start indices of dynamic slice"));

  VLOG(2) << StrFormat(
      "slicing shape %s at dynamic start_indices %s with slice_sizes={%s}",
      ShapeUtil::HumanString(operand_shape),
      ShapeUtil::HumanString(start_indices_shape), StrJoin(slice_sizes, ", "));

  if (ShapeUtil::Rank(start_indices_shape) != 1) {
    return InvalidArgument(
        "Dynamic slice start indices of rank %d must be rank1.",
        ShapeUtil::Rank(start_indices_shape));
  }

  if (!ShapeUtil::ElementIsIntegral(start_indices_shape)) {
    return InvalidArgument(
        "Dynamic slice start indices must be of integral type.");
  }

  const int64 start_num_dims = start_indices_shape.dimensions(0);
  if (ShapeUtil::Rank(operand_shape) != start_num_dims) {
    return InvalidArgument(
        "Dynamic slice start number of dimensions %d (%s) must match rank "
        "%d of slice input (%s).",
        start_num_dims, ShapeUtil::HumanString(start_indices_shape),
        ShapeUtil::Rank(operand_shape), ShapeUtil::HumanString(operand_shape));
  }

  if (slice_sizes.size() != ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Dynamic slice index count does not match argument rank: %u vs %d.",
        slice_sizes.size(), ShapeUtil::Rank(operand_shape));
  }

  for (int64 dim = 0; dim < slice_sizes.size(); ++dim) {
    const int64 input_dim_size = operand_shape.dimensions(dim);
    const int64 slice_dim_size = slice_sizes[dim];
    if (slice_dim_size < 0) {
      return InvalidArgument("Negative size index to dynamic slice: %d.",
                             slice_dim_size);
    }
    if (slice_dim_size > input_dim_size) {
      return InvalidArgument(
          "Slice dim size %d greater than dynamic slice dimension: %d.",
          slice_dim_size, input_dim_size);
    }
    VLOG(2) << StrFormat("slice_sizes[%d] = %d", dim, slice_dim_size);
  }

  return ShapeUtil::MakeShape(operand_shape.element_type(), slice_sizes);
}

/* static */ StatusOr<Shape> ShapeInference::InferDynamicUpdateSliceShape(
    const Shape& operand_shape, const Shape& update_shape,
    const Shape& start_indices_shape) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand of dynamic update slice"));
  TF_RETURN_IF_ERROR(
      ExpectArray(update_shape, "update of dynamic update slice"));
  TF_RETURN_IF_ERROR(ExpectArray(start_indices_shape,
                                 "start indices of dynamic update slice"));

  VLOG(2) << StrFormat(
      "updating slice of shape %s at dynamic start_indices %s with update "
      "shape %s",
      ShapeUtil::HumanString(operand_shape),
      ShapeUtil::HumanString(start_indices_shape),
      ShapeUtil::HumanString(update_shape));

  if (ShapeUtil::Rank(start_indices_shape) != 1) {
    return InvalidArgument(
        "Dynamic update slice start indices of rank %d must be rank1.",
        ShapeUtil::Rank(start_indices_shape));
  }

  if (!ShapeUtil::ElementIsIntegral(start_indices_shape)) {
    return InvalidArgument(
        "Dynamic update slice start indices must be of integral type.");
  }

  const int64 start_num_dims = start_indices_shape.dimensions(0);
  if (ShapeUtil::Rank(operand_shape) != start_num_dims) {
    return InvalidArgument(
        "Dynamic update slice start number of dimensions %d (%s) must match "
        "rank %d of slice input (%s).",
        start_num_dims, ShapeUtil::HumanString(start_indices_shape),
        ShapeUtil::Rank(operand_shape), ShapeUtil::HumanString(operand_shape));
  }

  if (ShapeUtil::Rank(update_shape) != ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Dynamic update slice update rank does not match argument rank: "
        "%d vs %d.",
        ShapeUtil::Rank(update_shape), ShapeUtil::Rank(operand_shape));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(operand_shape,
                                                     update_shape)) {
    return InvalidArgument(
        "Dynamic update slice update element type does not match argument. "
        "operand.element_type: %s vs update.element_type: %s.",
        PrimitiveType_Name(operand_shape.element_type()),
        PrimitiveType_Name(update_shape.element_type()));
  }

  for (int64 dim = 0; dim < ShapeUtil::Rank(operand_shape); ++dim) {
    const int64 input_dim_size = operand_shape.dimensions(dim);
    const int64 update_dim_size = update_shape.dimensions(dim);
    if (update_dim_size < 0) {
      return InvalidArgument(
          "Size index %d to dynamic update slice must be >= 0.",
          update_dim_size);
    }
    if (update_dim_size > input_dim_size) {
      return InvalidArgument(
          "Update dim size %d greater than dynamic slice dimension: %d.",
          update_dim_size, input_dim_size);
    }
    VLOG(2) << StrFormat("update_sizes[%d] = %d", dim, update_dim_size);
  }

  return operand_shape;
}

/*static */ StatusOr<Shape> ShapeInference::InferReverseShape(
    const Shape& operand_shape, absl::Span<const int64> dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of reverse"));
  if (!AllUnique(dimensions)) {
    return InvalidArgument("a dimension number is duplicated in reverse");
  }
  for (int64 dimension : dimensions) {
    if (dimension >= ShapeUtil::Rank(operand_shape) || dimension < 0) {
      return InvalidArgument(
          "One of the reverse dimensions (%d) is out-of-bounds in shape %s.",
          dimension, ShapeUtil::HumanString(operand_shape));
    }
  }
  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferGetTupleElementShape(
    const Shape& arg, int64 index) {
  if (!ShapeUtil::IsTuple(arg)) {
    return InvalidArgument(
        "Cannot infer shape: attempting to index into non-tuple: %s.",
        ShapeUtil::HumanString(arg));
  }

  if (index >= arg.tuple_shapes_size()) {
    return InvalidArgument(
        "Cannot infer shape: attempt to index out of tuple bounds: %d "
        ">= %d in shape %s.",
        index, arg.tuple_shapes_size(), ShapeUtil::HumanString(arg));
  }

  return arg.tuple_shapes(index);
}

/* static */ StatusOr<Shape> ShapeInference::InferWhileShape(
    const ProgramShape& condition, const ProgramShape& body,
    const Shape& init) {
  // Check the number of parameters for given computations.
  if (condition.parameters_size() != 1) {
    return InvalidArgument("Condition must take 1 arguments; got %d.",
                           condition.parameters_size());
  }
  if (body.parameters_size() != 1) {
    return InvalidArgument("Body must take 1 arguments; got %d.",
                           body.parameters_size());
  }

  auto shape_string = [&]() {
    return StrFormat(
        "Condition: %s; body: %s; init: %s.", ShapeUtil::HumanString(condition),
        ShapeUtil::HumanString(body), ShapeUtil::HumanString(init));
  };

  // Check the shapes of computation parameters and return types.
  if (!ShapeUtil::ShapeIs(condition.result(), PRED, {})) {
    return InvalidArgument("Condition must return a boolean; got %s.",
                           shape_string());
  }
  if (!ShapeUtil::Compatible(body.result(), condition.parameters(0)) ||
      !ShapeUtil::Compatible(body.result(), body.parameters(0)) ||
      !ShapeUtil::Compatible(body.result(), init)) {
    return InvalidArgument(
        "The parameter of condition and body, the result of the body, and init "
        "must all have the same shape; got %s.",
        shape_string());
  }

  return init;
}

/* static */ StatusOr<Shape> ShapeInference::InferConditionalShape(
    const Shape& predicate, const Shape& true_operand,
    const Shape& false_operand, const ProgramShape& true_computation,
    const ProgramShape& false_computation) {
  if (!ShapeUtil::ShapeIs(predicate, PRED, {})) {
    return InvalidArgument("Predicate must be a boolean; got %s.",
                           ShapeUtil::HumanString(predicate));
  }

  if (true_computation.parameters_size() != 1) {
    return InvalidArgument("true_computation must take 1 argument; got %d.",
                           true_computation.parameters_size());
  }
  if (!ShapeUtil::Compatible(true_computation.parameters(0), true_operand)) {
    auto true_shape_string = [&]() {
      return StrFormat("true_operand: %s; true_computation: %s",
                       ShapeUtil::HumanString(true_operand),
                       ShapeUtil::HumanString(true_computation));
    };
    return InvalidArgument(
        "true_operand must match the shape of the only parameter of "
        "true_computation: got %s.",
        true_shape_string());
  }

  if (false_computation.parameters_size() != 1) {
    return InvalidArgument("false_computation must take 1 argument; got %d.",
                           false_computation.parameters_size());
  }
  if (!ShapeUtil::Compatible(false_computation.parameters(0), false_operand)) {
    auto false_shape_string = [&]() {
      return StrFormat("false_operand: %s; false_computation: %s",
                       ShapeUtil::HumanString(false_operand),
                       ShapeUtil::HumanString(false_computation));
    };
    return InvalidArgument(
        "false_operand must match the shape of the only parameter of "
        "false_computation: got %s.",
        false_shape_string());
  }
  if (!ShapeUtil::Compatible(true_computation.result(),
                             false_computation.result())) {
    auto shape_string = [&]() {
      return StrFormat(
          "true_computation result: %s; false_computation result: %s.",
          ShapeUtil::HumanString(true_computation.result()),
          ShapeUtil::HumanString(false_computation.result()));
    };
    return InvalidArgument(
        "the result of true_computation and false_computation must have the "
        "same shape: got %s.",
        shape_string());
  }
  return true_computation.result();
}

/* static */ StatusOr<Shape> ShapeInference::InferBroadcastShape(
    const Shape& operand, absl::Span<const int64> broadcast_sizes) {
  TF_RETURN_IF_ERROR(ExpectArray(operand, "operand of broadcast"));
  for (int64 size : broadcast_sizes) {
    if (size < 0) {
      return InvalidArgument("Broadcast with negative dimension size %d.",
                             size);
    }
  }

  std::vector<int64> dimensions(operand.dimensions_size() +
                                broadcast_sizes.size());
  std::copy(broadcast_sizes.begin(), broadcast_sizes.end(), dimensions.begin());
  std::copy(operand.dimensions().begin(), operand.dimensions().end(),
            dimensions.begin() + broadcast_sizes.size());
  return ShapeUtil::MakeShape(operand.element_type(), dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferBroadcastShape(
    const Shape& operand_shape, const Shape& output_shape,
    absl::Span<const int64> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of broadcast"));
  TF_RETURN_IF_ERROR(ExpectArray(output_shape, "operand of broadcast"));
  const int64 operand_rank = ShapeUtil::Rank(operand_shape);
  const int64 output_rank = ShapeUtil::Rank(output_shape);
  if (operand_rank > output_rank) {
    return InvalidArgument(
        "InDim style broadcast must be to an equal or higher ranked shape; "
        "operand rank: %lld; output rank: %lld",
        operand_rank, output_rank);
  }
  if (operand_rank != broadcast_dimensions.size()) {
    return InvalidArgument(
        "Size of broadcast_dimensions has to match operand's rank; operand "
        "rank: %lld, size of broadcast_dimensions %u.",
        operand_rank, broadcast_dimensions.size());
  }
  for (int64 i = 0; i < operand_rank; i++) {
    if (broadcast_dimensions[i] < 0 || broadcast_dimensions[i] >= output_rank) {
      return InvalidArgument("Broadcast dimension %lld is out of bound",
                             broadcast_dimensions[i]);
    }
    if (operand_shape.dimensions(i) !=
            output_shape.dimensions(broadcast_dimensions[i]) &&
        operand_shape.dimensions(i) != 1) {
      return InvalidArgument(
          "Input dimension should be either 1 or equal to the output dimension "
          "it's broadcasting into; the %lldth operand dimension is %lld, the "
          "%lldth output dimension is %lld.",
          i, operand_shape.dimensions(i), broadcast_dimensions[i],
          output_shape.dimensions(broadcast_dimensions[i]));
    }
    // Make sure the broadcast dimensions are listed in a strictly increasing
    // order.
    if (i > 0 && broadcast_dimensions[i - 1] >= broadcast_dimensions[i]) {
      return InvalidArgument(
          "Broadcast dimensions order is wrong: %d comes after %d.",
          broadcast_dimensions[i], broadcast_dimensions.at(i - 1));
    }
  }

  return output_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferReshapeShape(
    const Shape& operand, absl::Span<const int64> dimensions,
    absl::Span<const int64> new_sizes) {
  TF_RETURN_IF_ERROR(ExpectArray(operand, "reshape"));

  Shape inferred_shape =
      ShapeUtil::MakeShape(operand.element_type(), new_sizes);
  VLOG(3) << "Reshape inferred shape: "
          << ShapeUtil::HumanString(inferred_shape);

  if (ShapeUtil::ElementsIn(operand) != ShapeUtil::ElementsIn(inferred_shape)) {
    return InvalidArgument(
        "Reshape operation has mismatched element counts: from=%d (%s) "
        "to=%d (%s).",
        ShapeUtil::ElementsIn(operand), ShapeUtil::HumanString(operand),
        ShapeUtil::ElementsIn(inferred_shape),
        ShapeUtil::HumanString(inferred_shape));
  }

  std::vector<int64> indices(ShapeUtil::Rank(operand));
  std::iota(indices.begin(), indices.end(), 0);
  if (dimensions.size() != ShapeUtil::Rank(operand) ||
      !std::is_permutation(dimensions.begin(), dimensions.end(),
                           indices.begin())) {
    return InvalidArgument(
        "Reshape dimensions [%s] are not a permutation of the operand "
        "dimensions (operand shape is %s).",
        StrJoin(dimensions, ","), ShapeUtil::HumanString(operand));
  }

  return inferred_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferTransposeShape(
    const Shape& operand, absl::Span<const int64> dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(operand, "transpose"));

  std::vector<int64> indices(ShapeUtil::Rank(operand));
  std::iota(indices.begin(), indices.end(), 0);
  if (dimensions.size() != ShapeUtil::Rank(operand) ||
      !std::is_permutation(dimensions.begin(), dimensions.end(),
                           indices.begin())) {
    return InvalidArgument(
        "Transpose dimensions [%s] are not a permutation of the operand "
        "dimensions (operand shape is %s).",
        StrJoin(dimensions, ","), ShapeUtil::HumanString(operand));
  }

  // Permute(dimensions,input) computes output[dimensions[i]]=input[i]. However,
  // we need output[i]=input[dimensions[i]] which is
  // Permute(Inverse(dimensions),input).
  return ShapeUtil::PermuteDimensions(InversePermutation(dimensions), operand);
}

// TODO(b/36794510): Make broadcast semantics more consistent, by supporting
// "degenerate" cases, as with binary elementwise ops.
/* static */ StatusOr<Shape> ShapeInference::InferClampShape(
    const Shape& min, const Shape& operand, const Shape& max) {
  TF_RETURN_IF_ERROR(ExpectArray(min, "clamp min"));
  TF_RETURN_IF_ERROR(ExpectArray(operand, "clamp operand"));
  TF_RETURN_IF_ERROR(ExpectArray(max, "clamp max"));
  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(min, operand) ||
      !ShapeUtil::SameElementTypeIgnoringFpPrecision(max, operand)) {
    return InvalidArgument("Clamp with different operand types: %s, %s, %s.",
                           ShapeUtil::HumanString(min),
                           ShapeUtil::HumanString(operand),
                           ShapeUtil::HumanString(max));
  }
  if (((ShapeUtil::CompatibleIgnoringFpPrecision(min, operand) ||
        ShapeUtil::IsScalar(min)) &&
       (ShapeUtil::CompatibleIgnoringFpPrecision(max, operand) ||
        ShapeUtil::IsScalar(max)))) {
    return operand;
  }
  if (ShapeUtil::IsScalar(operand)) {
    if (ShapeUtil::CompatibleIgnoringFpPrecision(min, max)) {
      return ShapeUtil::ChangeElementType(min, operand.element_type());
    } else if (ShapeUtil::IsScalar(min)) {
      return ShapeUtil::ChangeElementType(max, operand.element_type());
    } else if (ShapeUtil::IsScalar(max)) {
      return ShapeUtil::ChangeElementType(min, operand.element_type());
    }
  }
  return Unimplemented("%s, %s <clamp> %s is not implemented.",
                       min.ShortDebugString(), max.ShortDebugString(),
                       operand.ShortDebugString());
}

// TODO(b/36794510): Make broadcast semantics more consistent, by supporting
// "degenerate" cases, as with binary elementwise ops, as well as scalar
// broadcast from all operands, not just the predicate.
/* static */ StatusOr<Shape> ShapeInference::InferSelectShape(
    const Shape& pred, const Shape& on_true, const Shape& on_false) {
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(on_true, on_false)) {
    return InvalidArgument(
        "Operands to select must be the same shape; got %s and %s.",
        ShapeUtil::HumanString(on_true), ShapeUtil::HumanString(on_false));
  }
  if (pred.element_type() != PRED) {
    return InvalidArgument(
        "Select's pred operand must have PRED element type; got %s.",
        ShapeUtil::HumanString(pred));
  }
  if (ShapeUtil::CompatibleIgnoringElementType(pred, on_true) ||
      ShapeUtil::IsScalar(pred)) {
    // By this stage we know that pred's element type is PRED. Therefore, this
    // check restricts pred to be a PRED scalar, or a PRED array with the same
    // dimensions as on_true and on_false.
    return ShapeUtil::ChangeElementType(
        on_true, ShapeUtil::HigherPrecisionElementType(on_true, on_false));
  } else {
    return InvalidArgument(
        "Select operation with non-scalar predicate with dimensionality "
        " different from the other operands: %s.",
        ShapeUtil::HumanString(pred));
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferTupleSelectShape(
    const Shape& pred, const Shape& on_true, const Shape& on_false) {
  // Select only defines the top-level buffer, so if it's a tuple, the two
  // input must match exactly.
  if (!ShapeUtil::Compatible(on_true, on_false)) {
    return InvalidArgument(
        "Operands to tuple-select must be the same shape; got %s and %s.",
        ShapeUtil::HumanString(on_true), ShapeUtil::HumanString(on_false));
  }
  if (pred.element_type() != PRED) {
    return InvalidArgument(
        "TupleSelect's pred operand must have PRED element type; got %s.",
        ShapeUtil::HumanString(pred));
  }
  if (!ShapeUtil::IsScalar(pred)) {
    return InvalidArgument(
        "TupleSelect operation with non-scalar predicate: %s.",
        ShapeUtil::HumanString(pred));
  }
  return on_true;
}

/* static */ StatusOr<Shape> ShapeInference::InferCallShape(
    absl::Span<const Shape* const> arg_shapes, const ProgramShape& to_apply) {
  // The applied function's arity equals the number of arguments.
  if (arg_shapes.size() != to_apply.parameters_size()) {
    string computation_signature = ShapeUtil::HumanString(to_apply);
    string argument_shapes =
        StrJoin(arg_shapes, ", ", [](string* out, const Shape* shape) {
          absl::StrAppend(out, ShapeUtil::HumanString(*shape));
        });
    return InvalidArgument(
        "Call applied function arity must match number of arguments; got: "
        "arity: %d, arguments: %u; computation signature: %s; argument "
        "shapes: [%s].",
        to_apply.parameters_size(), arg_shapes.size(), computation_signature,
        argument_shapes);
  }

  // All arguments must be compatible with the program shape.
  for (int i = 0; i < arg_shapes.size(); ++i) {
    const Shape& arg_shape = *arg_shapes[i];
    const Shape& param_shape = to_apply.parameters(i);
    if (!ShapeUtil::Compatible(arg_shape, param_shape)) {
      return InvalidArgument(
          "Call parameter must match argument; got parameter %d shape: %s, "
          "argument shape: %s.",
          i, ShapeUtil::HumanString(param_shape),
          ShapeUtil::HumanString(arg_shape));
    }
  }

  return to_apply.result();
}

static Status ValidateGatherDimensionNumbers(
    const Shape& input_shape, absl::Span<const int64> start_indices_shape,
    const GatherDimensionNumbers& dim_numbers) {
  if (!absl::c_is_sorted(dim_numbers.offset_dims())) {
    return InvalidArgument(
        "Output window dimensions in gather op must be ascending; got: %s.",
        StrJoin(dim_numbers.offset_dims(), ", "));
  }

  if (absl::c_adjacent_find(dim_numbers.offset_dims()) !=
      dim_numbers.offset_dims().end()) {
    return InvalidArgument(
        "Output window dimensions in gather op must not repeat; got: %s.",
        StrJoin(dim_numbers.offset_dims(), ", "));
  }

  const int64 output_offset_dim_count = dim_numbers.offset_dims_size();
  const int64 output_shape_rank =
      output_offset_dim_count + start_indices_shape.size() - 1;

  for (int i = 0; i < dim_numbers.offset_dims_size(); ++i) {
    int64 offset_dim = dim_numbers.offset_dims(i);
    if (offset_dim < 0 || offset_dim >= output_shape_rank) {
      return InvalidArgument(
          "Offset dimension %d in gather op is out of bounds; got %d, but "
          "should "
          "have been in [0,%d).",
          i, offset_dim, output_shape_rank);
    }
  }

  if (dim_numbers.start_index_map_size() !=
      start_indices_shape[dim_numbers.index_vector_dim()]) {
    return InvalidArgument(
        "Gather op has %d elements in start_index_map and the "
        "bound of dimension index_vector_dim=%d of start_indices is "
        "%d. These two numbers must be equal.",
        dim_numbers.start_index_map_size(), dim_numbers.index_vector_dim(),
        start_indices_shape[dim_numbers.index_vector_dim()]);
  }

  for (int i = 0; i < dim_numbers.start_index_map_size(); i++) {
    int64 operand_dim_for_start_index_i = dim_numbers.start_index_map(i);
    if (operand_dim_for_start_index_i < 0 ||
        operand_dim_for_start_index_i >= input_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid start_index_map; domain is [0, %d), got: %d->%d.",
          input_shape.dimensions_size(), i, operand_dim_for_start_index_i);
    }
  }

  std::vector<int64> sorted_start_index_map(
      dim_numbers.start_index_map().begin(),
      dim_numbers.start_index_map().end());

  absl::c_sort(sorted_start_index_map);

  if (absl::c_adjacent_find(sorted_start_index_map) !=
      sorted_start_index_map.end()) {
    return InvalidArgument(
        "Repeated dimensions are not allowed in start_index_map; "
        "got: %s.",
        StrJoin(dim_numbers.start_index_map(), ", "));
  }

  for (int64 collapsed_dim : dim_numbers.collapsed_slice_dims()) {
    if (collapsed_dim < 0 || collapsed_dim >= input_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid collapsed_slice_dims set in gather op; valid range is [0, "
          "%d), got: %d.",
          input_shape.dimensions_size(), collapsed_dim);
    }
  }

  if (!absl::c_is_sorted(dim_numbers.collapsed_slice_dims())) {
    return InvalidArgument(
        "collapsed_slice_dims in gather op must be sorted; got: %s",
        StrJoin(dim_numbers.collapsed_slice_dims(), ", "));
  }

  if (absl::c_adjacent_find(dim_numbers.collapsed_slice_dims()) !=
      dim_numbers.collapsed_slice_dims().end()) {
    return InvalidArgument(
        "Repeated dimensions not allowed in collapsed_slice_dims in gather op; "
        "got: %s.",
        StrJoin(dim_numbers.collapsed_slice_dims(), ", "));
  }

  return Status::OK();
}

/*static*/ StatusOr<Shape> ShapeInference::InferGatherShape(
    const Shape& input_shape, const Shape& start_indices_shape,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64> slice_sizes) {
  TF_RETURN_IF_ERROR(
      ExpectArray(input_shape, "input tensor operand gather op"));
  TF_RETURN_IF_ERROR(
      ExpectArray(start_indices_shape, "gather indices operand of gather op"));

  if (!ShapeUtil::ElementIsIntegral(start_indices_shape)) {
    return InvalidArgument(
        "Gather indices parameter must be an integral tensor; got %s.",
        ShapeUtil::HumanString(start_indices_shape));
  }

  // We implicitly reshape gather indices of shape P[A,B,C] to P[A,B,C,1] if
  // index_vector_dim is rank(P).  The bounds of this expanded shape is
  // stored in expanded_start_indices_shape.

  if (start_indices_shape.dimensions_size() <
          gather_dim_numbers.index_vector_dim() ||
      gather_dim_numbers.index_vector_dim() < 0) {
    return InvalidArgument(
        "Gather index leaf dimension must be within [0, rank(start_indices) + "
        "1). rank(start_indices) is %d and gather index leaf dimension is "
        "%d.",
        start_indices_shape.dimensions_size(),
        gather_dim_numbers.index_vector_dim());
  }

  std::vector<int64> expanded_start_indices_shape;
  expanded_start_indices_shape.reserve(start_indices_shape.dimensions_size());
  absl::c_copy(start_indices_shape.dimensions(),
               std::back_inserter(expanded_start_indices_shape));
  if (expanded_start_indices_shape.size() ==
      gather_dim_numbers.index_vector_dim()) {
    expanded_start_indices_shape.push_back(1);
  }

  TF_RETURN_IF_ERROR(ValidateGatherDimensionNumbers(
      input_shape, expanded_start_indices_shape, gather_dim_numbers));

  if (slice_sizes.size() != input_shape.dimensions_size()) {
    return InvalidArgument(
        "Gather op must have one slice size for every input dimension; got: "
        "len(slice_sizes)=%lu, input_shape.rank=%d.",
        slice_sizes.size(), input_shape.dimensions_size());
  }

  if (slice_sizes.size() !=
      gather_dim_numbers.offset_dims_size() +
          gather_dim_numbers.collapsed_slice_dims_size()) {
    return InvalidArgument(
        "All components of the offset index in a gather op must either be a "
        "offset dimension or explicitly collapsed; got len(slice_sizes)=%lu, "
        "output_slice_sizes=%s, collapsed_slice_dims=%s.",
        slice_sizes.size(), StrJoin(gather_dim_numbers.offset_dims(), ","),
        StrJoin(gather_dim_numbers.collapsed_slice_dims(), ","));
  }

  for (int i = 0; i < slice_sizes.size(); i++) {
    int64 slice_size = slice_sizes[i];
    int64 corresponding_input_size = input_shape.dimensions(i);
    if (slice_size < 0 || slice_size > corresponding_input_size) {
      return InvalidArgument(
          "Slice size at index %d in gather op is out of range, must be "
          "within [0, %d), got %d.",
          i, corresponding_input_size + 1, slice_size);
    }
  }

  for (int i = 0; i < gather_dim_numbers.collapsed_slice_dims_size(); i++) {
    if (slice_sizes[gather_dim_numbers.collapsed_slice_dims(i)] != 1) {
      return InvalidArgument(
          "Gather op can only collapse slice dims with bound 1, but bound is "
          "%d for index %d at position %d.",
          slice_sizes[gather_dim_numbers.collapsed_slice_dims(i)],
          gather_dim_numbers.collapsed_slice_dims(i), i);
    }
  }

  int64 result_rank = gather_dim_numbers.offset_dims_size() +
                      (expanded_start_indices_shape.size() - 1);
  int64 offset_dims_seen = 0;
  int64 gather_dims_seen = 0;
  std::vector<int64> output_dim_bounds;
  output_dim_bounds.reserve(result_rank);
  for (int64 i = 0; i < result_rank; i++) {
    int64 current_bound;
    bool is_window_index =
        absl::c_binary_search(gather_dim_numbers.offset_dims(), i);
    if (is_window_index) {
      while (absl::c_binary_search(gather_dim_numbers.collapsed_slice_dims(),
                                   offset_dims_seen)) {
        offset_dims_seen++;
      }
      current_bound = slice_sizes[offset_dims_seen++];
    } else {
      if (gather_dims_seen == gather_dim_numbers.index_vector_dim()) {
        gather_dims_seen++;
      }
      current_bound = expanded_start_indices_shape[gather_dims_seen++];
    }

    output_dim_bounds.push_back(current_bound);
  }

  return ShapeUtil::MakeShape(input_shape.element_type(), output_dim_bounds);
}

namespace {

Status ValidateScatterDimensionNumbers(
    const Shape& operand_shape, absl::Span<const int64> scatter_indices_shape,
    const Shape& updates_shape, const ScatterDimensionNumbers& dim_numbers) {
  // Validate update_window_dims in ScatterDimensionNumbers.
  if (!absl::c_is_sorted(dim_numbers.update_window_dims())) {
    return InvalidArgument(
        "update_window_dims in scatter op must be sorted; got: %s.",
        StrJoin(dim_numbers.update_window_dims(), ", "));
  }
  if (absl::c_adjacent_find(dim_numbers.update_window_dims()) !=
      dim_numbers.update_window_dims().end()) {
    return InvalidArgument(
        "update_window_dims in scatter op must not repeat; got: %s.",
        StrJoin(dim_numbers.update_window_dims(), ", "));
  }
  const int64 updates_rank = ShapeUtil::Rank(updates_shape);
  for (int64 window_dim : dim_numbers.update_window_dims()) {
    if (window_dim < 0 || window_dim >= updates_rank) {
      return InvalidArgument(
          "Invalid update_window_dims set in scatter op; valid range is [0, "
          "%d). got: %d.",
          updates_rank, window_dim);
    }
  }

  // Validate inserted_window_dims in ScatterDimensionNumbers.
  if (!absl::c_is_sorted(dim_numbers.inserted_window_dims())) {
    return InvalidArgument(
        "inserted_window_dims in scatter op must be sorted; got: %s.",
        StrJoin(dim_numbers.inserted_window_dims(), ", "));
  }
  if (absl::c_adjacent_find(dim_numbers.inserted_window_dims()) !=
      dim_numbers.inserted_window_dims().end()) {
    return InvalidArgument(
        "inserted_window_dims in scatter op must not repeat; got: %s.",
        StrJoin(dim_numbers.inserted_window_dims(), ", "));
  }
  for (int64 inserted_dim : dim_numbers.inserted_window_dims()) {
    if (inserted_dim < 0 || inserted_dim >= operand_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid inserted_window_dims set in scatter op; valid range is [0, "
          "%d), got: %d.",
          operand_shape.dimensions_size(), inserted_dim);
    }
  }

  // Validate window size.
  auto window_size = dim_numbers.update_window_dims_size() +
                     dim_numbers.inserted_window_dims_size();
  if (window_size != ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Scatter op has window of size %d; doesn't match operand of rank %d.",
        window_size, ShapeUtil::Rank(operand_shape));
  }

  // Validate scatter_dims_to_operand_dims in ScatterDimensionNumbers.
  if (dim_numbers.scatter_dims_to_operand_dims_size() !=
      scatter_indices_shape[dim_numbers.index_vector_dim()]) {
    return InvalidArgument(
        "Scatter op has %d elements in scatter_dims_to_operand_dims and the "
        "bound of dimension index_vector_dim=%d of scatter_indices is %d. "
        "These two numbers must be equal.",
        dim_numbers.scatter_dims_to_operand_dims_size(),
        dim_numbers.index_vector_dim(),
        scatter_indices_shape[dim_numbers.index_vector_dim()]);
  }
  for (int i = 0; i < dim_numbers.scatter_dims_to_operand_dims_size(); ++i) {
    int64 scatter_dim_to_operand_dim =
        dim_numbers.scatter_dims_to_operand_dims(i);
    if (scatter_dim_to_operand_dim < 0 ||
        scatter_dim_to_operand_dim >= operand_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid scatter_dims_to_operand_dims mapping; domain is [0, %d), "
          "got: %d->%d.",
          operand_shape.dimensions_size(), i, scatter_dim_to_operand_dim);
    }
  }
  std::vector<int64> sorted_scatter_dims_to_operand_dims(
      dim_numbers.scatter_dims_to_operand_dims().begin(),
      dim_numbers.scatter_dims_to_operand_dims().end());
  absl::c_sort(sorted_scatter_dims_to_operand_dims);
  if (absl::c_adjacent_find(sorted_scatter_dims_to_operand_dims) !=
      sorted_scatter_dims_to_operand_dims.end()) {
    return InvalidArgument(
        "Repeated dimensions not allowed in scatter_dims_to_operand_dims; "
        "got: %s.",
        StrJoin(dim_numbers.scatter_dims_to_operand_dims(), ", "));
  }

  return Status::OK();
}

}  // namespace

/*static*/ StatusOr<Shape> ShapeInference::InferScatterShape(
    const Shape& operand_shape, const Shape& scatter_indices_shape,
    const Shape& updates_shape, const ProgramShape& to_apply_shape,
    const ScatterDimensionNumbers& scatter_dim_numbers) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand tensor of scatter op"));
  TF_RETURN_IF_ERROR(
      ExpectArray(scatter_indices_shape, "scatter indices of scatter op"));
  TF_RETURN_IF_ERROR(ExpectArray(updates_shape, "updates of scatter op"));

  if (!ShapeUtil::ElementIsIntegral(scatter_indices_shape)) {
    return InvalidArgument(
        "Scatter indices parameter must be an integral tensor; got %s.",
        ShapeUtil::HumanString(scatter_indices_shape));
  }

  if (scatter_indices_shape.dimensions_size() <
          scatter_dim_numbers.index_vector_dim() ||
      scatter_dim_numbers.index_vector_dim() < 0) {
    return InvalidArgument(
        "Scatter index leaf dimension must be within [0, rank(scatter_indices)"
        " + 1). rank(scatter_indices) is %d and scatter index leaf dimension "
        "is %d.",
        scatter_indices_shape.dimensions_size(),
        scatter_dim_numbers.index_vector_dim());
  }

  // Check if the update computation has a proper shape as a reduction.
  const Shape init_value_shape =
      ShapeUtil::MakeShape(operand_shape.element_type(), {});
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply_shape, {&init_value_shape},
                                        {updates_shape.element_type()},
                                        /*inputs=*/1));

  std::vector<int64> expanded_scatter_indices_shape =
      ArraySliceToVector(AsInt64Slice(scatter_indices_shape.dimensions()));
  if (expanded_scatter_indices_shape.size() ==
      scatter_dim_numbers.index_vector_dim()) {
    expanded_scatter_indices_shape.push_back(1);
  }

  int64 expected_updates_rank = expanded_scatter_indices_shape.size() - 1 +
                                scatter_dim_numbers.update_window_dims_size();
  if (ShapeUtil::Rank(updates_shape) != expected_updates_rank) {
    return InvalidArgument("Updates tensor must be of rank %d; got %d.",
                           expected_updates_rank,
                           ShapeUtil::Rank(updates_shape));
  }

  TF_RETURN_IF_ERROR(ValidateScatterDimensionNumbers(
      operand_shape, expanded_scatter_indices_shape, updates_shape,
      scatter_dim_numbers));

  int64 inserted_dims_seen = 0;
  std::vector<int64> max_update_slice_sizes;
  for (int i = 0; i < operand_shape.dimensions_size(); ++i) {
    if (inserted_dims_seen < scatter_dim_numbers.inserted_window_dims_size() &&
        scatter_dim_numbers.inserted_window_dims(inserted_dims_seen) == i) {
      ++inserted_dims_seen;
    } else {
      max_update_slice_sizes.push_back(operand_shape.dimensions(i));
    }
  }
  for (int i = 0; i < scatter_dim_numbers.update_window_dims_size(); ++i) {
    auto update_window_dim = scatter_dim_numbers.update_window_dims(i);
    if (updates_shape.dimensions(update_window_dim) >
        max_update_slice_sizes[i]) {
      return InvalidArgument(
          "Bounds of the window dimensions of updates must not exceed the "
          "bounds of the corresponding dimensions of operand. For dimension "
          "%d, updates bound is %d, operand bound is %d.",
          update_window_dim, updates_shape.dimensions(update_window_dim),
          max_update_slice_sizes[i]);
    }
  }

  int64 scatter_dims_seen = 0;
  for (int64 i = 0; i < ShapeUtil::Rank(updates_shape); ++i) {
    bool is_update_window_dim =
        absl::c_binary_search(scatter_dim_numbers.update_window_dims(), i);
    if (is_update_window_dim) {
      continue;
    }
    if (scatter_dims_seen == scatter_dim_numbers.index_vector_dim()) {
      ++scatter_dims_seen;
    }
    if (updates_shape.dimensions(i) !=
        expanded_scatter_indices_shape[scatter_dims_seen]) {
      return InvalidArgument(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices. For "
          "scatter dimension %d, updates bound is %d, scatter_indices "
          "bound is %d.",
          i, updates_shape.dimensions(i),
          expanded_scatter_indices_shape[scatter_dims_seen]);
    }
    ++scatter_dims_seen;
  }

  return operand_shape;
}

}  // namespace xla
