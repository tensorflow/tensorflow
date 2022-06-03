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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <set>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
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
#include "tensorflow/core/platform/statusor.h"

namespace xla {
namespace {

using absl::StrFormat;
using absl::StrJoin;

// Returns true if no element is present in slice more than once.
bool AllUnique(absl::Span<const int64_t> slice) {
  return std::set<int64_t>(slice.begin(), slice.end()).size() == slice.size();
}

Status ExpectArray(const Shape& shape, absl::string_view op_type) {
  if (!shape.IsArray()) {
    return InvalidArgument("Expected array argument for %s, but got %s.",
                           std::string(op_type), ShapeUtil::HumanString(shape));
  }
  return OkStatus();
}

Status VerifyReducerShape(const ProgramShape& reducer_shape,
                          absl::Span<const Shape* const> init_value_shapes,
                          absl::Span<const PrimitiveType> input_element_types,
                          int64_t inputs) {
  if (reducer_shape.parameters_size() != inputs * 2) {
    return InvalidArgument(
        "Reduction function must take %d parameters, but "
        "takes %d parameter(s).",
        inputs * 2, reducer_shape.parameters_size());
  }

  const Shape& accumulator_shape = reducer_shape.result();
  std::vector<const Shape*> accumulator_subshapes;
  if (accumulator_shape.IsArray()) {
    if (inputs != 1) {
      return InvalidArgument(
          "Reduction function must produce a tuple with %d elements, but "
          "produces a scalar",
          inputs);
    }
    accumulator_subshapes.push_back(&accumulator_shape);
  } else if (accumulator_shape.IsTuple()) {
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
    if (element_shape->rank() != 0) {
      return InvalidArgument(
          "Reduction function must return a scalar or tuple of scalars but "
          "returns shape: %s",
          ShapeUtil::HumanString(accumulator_shape));
    }
  }

  for (int64_t i = 0; i < inputs; ++i) {
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

  return OkStatus();
}

StatusOr<Shape> InferWindowOutputShape(const Shape& base_shape,
                                       const Window& window,
                                       PrimitiveType element_type) {
  if (window.dimensions_size() != base_shape.rank()) {
    return InvalidArgument(
        "Window has dimension %d but base shape has dimension %d.",
        window.dimensions_size(), base_shape.rank());
  }

  std::vector<int64_t> output_dimensions(window.dimensions_size());
  std::vector<bool> output_is_dynamic(window.dimensions_size());
  for (int64_t i = 0; i < window.dimensions_size(); ++i) {
    const auto& dim = window.dimensions(i);
    if (dim.size() <= 0) {
      return InvalidArgument("Window %s has a non-positive dimension.",
                             window.DebugString());
    }
    if (dim.stride() <= 0) {
      return InvalidArgument("Window %s has a non-positive stride.",
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

    const int64_t dilated_base = window_util::DilatedBound(
        ShapeUtil::GetDimension(base_shape, i), dim.base_dilation());
    const int64_t padded_dilated_base =
        dim.padding_low() + dilated_base + dim.padding_high();
    const int64_t dilated_window =
        window_util::DilatedBound(dim.size(), dim.window_dilation());

    output_dimensions[i] = window_util::StridedBound(
        padded_dilated_base, dilated_window, dim.stride());
    output_is_dynamic[i] = base_shape.is_dynamic_dimension(i);
  }

  return ShapeUtil::MakeValidatedShape(element_type, output_dimensions,
                                       output_is_dynamic);
}

StatusOr<PrimitiveType> MaybeUpcast(
    PrimitiveType from_type,
    absl::optional<PrimitiveType> preferred_element_type) {
  if (!preferred_element_type.has_value() ||
      *preferred_element_type == from_type) {
    return from_type;
  }
  if (!primitive_util::IsFloatingPointType(from_type) &&
      primitive_util::BitWidth(*preferred_element_type) <
          primitive_util::BitWidth(from_type)) {
    return InvalidArgument(
        "`preferred_element_type` must not be narrower than the original "
        "type.");
  }
  return *preferred_element_type;
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
    case HloOpcode::kRoundNearestEven:
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
    case HloOpcode::kLogistic:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
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
    case HloOpcode::kPopulationCount:
      if (!ShapeUtil::ElementIsIntegral(shape)) {
        return InvalidArgument(
            "Expected an integral element type in argument to PopulationCount "
            "operation; got %s.",
            PrimitiveType_Name(shape.element_type()));
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
    absl::Span<const Shape* const> arg_shapes, const int64_t dimension) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Concatenate expects at least one argument.");
  }
  if (dimension < 0 || dimension >= arg_shapes[0]->rank()) {
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
    if (arg_shape->rank() != shape->rank()) {
      return InvalidArgument(
          "Cannot concatenate arrays with different ranks: %d (%s) vs %d "
          "(%s).",
          arg_shape->rank(), ShapeUtil::HumanString(*arg_shape), shape->rank(),
          ShapeUtil::HumanString(*shape));
    }
    if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(*arg_shape, *shape)) {
      return InvalidArgument(
          "Cannot concatenate arrays with different element types: %s vs %s.",
          PrimitiveType_Name(arg_shape->element_type()),
          PrimitiveType_Name(shape->element_type()));
    }
    for (int64_t dimension_number = 0; dimension_number < arg_shape->rank();
         ++dimension_number) {
      if (arg_shape->dimensions(dimension_number) !=
          shape->dimensions(dimension_number)) {
        if (dimension_number == dimension) {
          continue;  // It's okay to differ in the dimension we're
                     // concatenating.
        }
        return InvalidArgument(
            "Cannot concatenate arrays that differ in dimensions other than "
            "the one being concatenated. Dimension %d in both shapes must be "
            "equal: %s vs %s.",
            dimension_number, ShapeUtil::HumanString(*arg_shape),
            ShapeUtil::HumanString(*shape));
      }
    }
    element_type = ShapeUtil::HigherPrecisionElementType(*shape, *arg_shape);
  }

  std::vector<int64_t> new_dimensions(arg_shape->dimensions().begin(),
                                      arg_shape->dimensions().end());
  for (size_t i = 1; i < arg_shapes.size(); ++i) {
    new_dimensions[dimension] += arg_shapes[i]->dimensions(dimension);
  }

  Shape result = ShapeUtil::MakeShape(element_type, new_dimensions);

  // Set dynamic dimensions if any input has dynamic dimension.
  for (const Shape* shape : arg_shapes) {
    for (int64_t i = 0; i < shape->dimensions_size(); ++i) {
      if (shape->is_dynamic_dimension(i)) {
        result.set_dynamic_dimension(i, true);
      }
    }
  }
  return result;
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
  if (!operand_shape.IsArray() ||
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
    return InvalidArgument("Conversion between complex and real type %s => %s.",
                           ShapeUtil::HumanString(operand_shape),
                           PrimitiveType_Name(new_element_type));
  }
  if (!operand_shape.IsArray() ||
      !primitive_util::IsArrayType(new_element_type)) {
    // Note: we may want to support tuple conversions via this operation in the
    // future, by recursing into the tuple elements to check all sub-conversions
    // are valid. For now we just reject them, though.
    return InvalidArgument(
        "Cannot convert from or to tuple type; requested conversion: %s => %s.",
        ShapeUtil::HumanString(operand_shape),
        PrimitiveType_Name(new_element_type));
  }

  int input_bitwidth = primitive_util::BitWidth(old_element_type);
  int output_bitwidth = primitive_util::BitWidth(new_element_type);
  if (std::max(input_bitwidth, output_bitwidth) %
          std::min(input_bitwidth, output_bitwidth) !=
      0) {
    return InvalidArgument(
        "Cannot bitcast types with undivisible bit-widths: %s => %s.",
        PrimitiveType_Name(old_element_type),
        PrimitiveType_Name(new_element_type));
  }
  int ratio = std::max(output_bitwidth, input_bitwidth) /
              std::min(output_bitwidth, input_bitwidth);

  Shape new_shape = operand_shape;
  new_shape.set_element_type(new_element_type);
  if (input_bitwidth > output_bitwidth) {
    ShapeUtil::AppendMinorDimension(ratio, &new_shape);
  } else if (input_bitwidth < output_bitwidth) {
    int last_dimension_idx = operand_shape.dimensions_size() - 1;
    if (operand_shape.dimensions_size() < 1 ||
        operand_shape.dimensions(last_dimension_idx) != ratio) {
      return InvalidArgument(
          "Last dimension of input shape=%d is not equal to ratio of "
          "bit-widths=%d "
          "for bitcast-convert from %s to %s",
          operand_shape.dimensions(last_dimension_idx), ratio,
          ShapeUtil::HumanString(operand_shape),
          PrimitiveType_Name(new_element_type));
    }
    new_shape.DeleteDimension(last_dimension_idx);
  }
  return new_shape;
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
  if (!operand_shape.IsArray()) {
    return InvalidArgument(
        "Pad operation does not support tuple-shape operands.");
  }
  if (!ShapeUtil::IsScalar(padding_value_shape)) {
    return InvalidArgument(
        "Pad operation does not support non-scalar padding values.");
  }
  if (operand_shape.rank() != padding_config.dimensions_size()) {
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

  if (!padding_value_shape.is_static()) {
    return InvalidArgument("Dynamic padding value is not supported");
  }

  std::vector<int64_t> dimensions(operand_shape.rank());
  std::vector<bool> is_dynamic(operand_shape.rank());
  for (int64_t i = 0; i < operand_shape.dimensions_size(); ++i) {
    const auto& p = padding_config.dimensions(i);
    dimensions[i] = operand_shape.dimensions(i) + p.edge_padding_low() +
                    p.edge_padding_high() +
                    std::max<int64_t>(operand_shape.dimensions(i) - 1, 0LL) *
                        p.interior_padding();
    if (dimensions[i] < 0) {
      return InvalidArgument("Padding result in negative size for dimension %d",
                             i);
    }
    is_dynamic[i] = operand_shape.is_dynamic_dimension(i);
  }

  return ShapeUtil::MakeShape(
      ShapeUtil::HigherPrecisionElementType(operand_shape, padding_value_shape),
      dimensions, is_dynamic);
}

// Current DotDimensionNumbers Requirements:
//
// Contracting Dimensions:
// *) Same number of contracting dimensions on both lhs and rhs.
// *) Contracting dimension size must be the same on both lhs and rhs.
//
// Batch Dimensions:
// *) Same number of batch dimensions on both lhs and rhs.
// *) Same batch dimension sizes on both lhs and rhs.
//

namespace {

Status ValidateDotDimensionNumbers(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  // Check that dimension numbers are in range.
  auto dims_in_range = [](const int64_t rank,
                          absl::Span<const int64_t> contracting_dims,
                          absl::Span<const int64_t> batch_dims) -> bool {
    auto in_range = [&rank](int64_t i) -> bool { return 0 <= i && i < rank; };
    return absl::c_all_of(contracting_dims, in_range) &&
           absl::c_all_of(batch_dims, in_range);
  };

  absl::Span<const int64_t> lhs_contracting_dimensions =
      dimension_numbers.lhs_contracting_dimensions();
  absl::Span<const int64_t> rhs_contracting_dimensions =
      dimension_numbers.rhs_contracting_dimensions();
  absl::Span<const int64_t> lhs_batch_dimensions =
      dimension_numbers.lhs_batch_dimensions();
  absl::Span<const int64_t> rhs_batch_dimensions =
      dimension_numbers.rhs_batch_dimensions();

  if (!dims_in_range(lhs.rank(), lhs_contracting_dimensions,
                     lhs_batch_dimensions) ||
      !dims_in_range(rhs.rank(), rhs_contracting_dimensions,
                     rhs_batch_dimensions)) {
    return InvalidArgument("A dimension number is out of range in Dot: %s.",
                           dimension_numbers.DebugString());
  }

  // Check that dimension numbers are unique.
  auto dims_unique = [](absl::Span<const int64_t> contracting_dims,
                        absl::Span<const int64_t> batch_dims) -> bool {
    absl::flat_hash_set<int64_t> dim_set;
    auto is_unique = [&dim_set](int64_t i) -> bool {
      return dim_set.insert(i).second;
    };
    return absl::c_all_of(contracting_dims, is_unique) &&
           absl::c_all_of(batch_dims, is_unique);
  };

  if (!dims_unique(lhs_contracting_dimensions, lhs_batch_dimensions) ||
      !dims_unique(rhs_contracting_dimensions, rhs_batch_dimensions)) {
    return InvalidArgument("A dimension number is not unique in Dot: %s.",
                           dimension_numbers.DebugString());
  }

  return OkStatus();
}

}  // namespace

/* static */ StatusOr<Shape> ShapeInference::InferDotOpShape(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers,
    absl::optional<PrimitiveType> preferred_element_type) {
  TF_RETURN_IF_ERROR(ExpectArray(lhs, "lhs of dot"));
  TF_RETURN_IF_ERROR(ExpectArray(rhs, "rhs of dot"));

  auto fail = [lhs, rhs](const std::string& addendum) -> Status {
    std::string message =
        StrFormat("Cannot infer shape for dot operation: %s <dot> %s.",
                  ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs));
    if (!addendum.empty()) {
      message += " " + addendum;
    }
    return InvalidArgument("%s", message);
  };

  // Validate basic properties of dot dimension numbers.
  TF_RETURN_IF_ERROR(ValidateDotDimensionNumbers(lhs, rhs, dimension_numbers));

  // Check that number of contracting dimensions match.
  if (dimension_numbers.lhs_contracting_dimensions_size() !=
      dimension_numbers.rhs_contracting_dimensions_size()) {
    return fail(
        "Must specify the same number of contracting dimensions for lhs and "
        "rhs.");
  }
  // Check that contracting dimension sizes match.
  for (int64_t i = 0; i < dimension_numbers.lhs_contracting_dimensions_size();
       ++i) {
    const int64_t lhs_contracting_dimension =
        dimension_numbers.lhs_contracting_dimensions(i);
    const int64_t rhs_contracting_dimension =
        dimension_numbers.rhs_contracting_dimensions(i);
    if (lhs.dimensions(lhs_contracting_dimension) !=
        rhs.dimensions(rhs_contracting_dimension)) {
      return fail("Contracting dimension sizes do not match.");
    }
  }

  // Check that number of batch dimensions match.
  if (dimension_numbers.lhs_batch_dimensions_size() !=
      dimension_numbers.rhs_batch_dimensions_size()) {
    return fail("Must the same number of batch dimensions for lhs and rhs.");
  }

  // Check that batch dimension numbers and sizes match.
  for (int64_t i = 0; i < dimension_numbers.lhs_batch_dimensions_size(); ++i) {
    if (lhs.dimensions(dimension_numbers.lhs_batch_dimensions(i)) !=
        rhs.dimensions(dimension_numbers.rhs_batch_dimensions(i))) {
      return fail("Batch dimension sizes must match for lhs/rhs.");
    }
  }

  // The ranks of lhs and rhs are decremented by 1 respectively due to the
  // contraction, and added for the rank of the result. When an input tensor is
  // a scalar, its contribution to the rank of the result is 0.
  // Generate the result dimensions in order, rhs dimensions followed by lhs
  // dimensions except the contracted and batch dimensions.
  std::vector<int64_t> dimensions;
  std::vector<bool> is_dynamic;
  const auto& lhs_batch_dimensions = dimension_numbers.lhs_batch_dimensions();
  const auto lhs_batch_dimensions_size =
      lhs.rank() - dimension_numbers.lhs_contracting_dimensions().size() +
      rhs.rank() - dimension_numbers.rhs_contracting_dimensions().size() -
      dimension_numbers.rhs_batch_dimensions().size();
  dimensions.reserve(lhs_batch_dimensions_size);
  is_dynamic.reserve(lhs_batch_dimensions_size);
  for (const int64_t lhs_dim : lhs_batch_dimensions) {
    dimensions.push_back(lhs.dimensions(lhs_dim));
    is_dynamic.push_back(lhs.is_dynamic_dimension(lhs_dim));
  }
  for (int64_t i = 0; i < lhs.rank(); i++) {
    if (!absl::c_linear_search(dimension_numbers.lhs_contracting_dimensions(),
                               i) &&
        !absl::c_linear_search(dimension_numbers.lhs_batch_dimensions(), i)) {
      dimensions.push_back(lhs.dimensions(i));
      is_dynamic.push_back(lhs.is_dynamic_dimension(i));
    }
  }
  for (int64_t i = 0; i < rhs.rank(); i++) {
    if (!absl::c_linear_search(dimension_numbers.rhs_contracting_dimensions(),
                               i) &&
        !absl::c_linear_search(dimension_numbers.rhs_batch_dimensions(), i)) {
      dimensions.push_back(rhs.dimensions(i));
      is_dynamic.push_back(rhs.is_dynamic_dimension(i));
    }
  }
  TF_ASSIGN_OR_RETURN(
      PrimitiveType type,
      MaybeUpcast(ShapeUtil::HigherPrecisionElementType(lhs, rhs),
                  preferred_element_type));
  Shape result = ShapeUtil::MakeShape(type, dimensions, is_dynamic);

  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(result));
  VLOG(2) << "inferred dot shape: " << ShapeUtil::HumanString(result);
  return result;
}

/* static */ StatusOr<Shape>
ShapeInference::InferDegenerateDimensionBroadcastShape(HloOpcode operation,
                                                       const Shape& lhs,
                                                       const Shape& rhs) {
  TF_RET_CHECK(lhs.rank() == rhs.rank());

  // The shapes have to be compatible. That is, if some dimension d has a
  // different size in the two shapes, one of them has to be 1 (a "degenerate"
  // dimension). In that case, the output shape has the non-1 dimension size
  // from the lhs/rhs pair in every index.
  std::vector<int64_t> output_dimensions(lhs.rank());
  std::vector<bool> output_dimensions_is_dynamic(lhs.rank());
  for (int64_t i = 0; i < lhs.rank(); ++i) {
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

  // Merge dynamic dimensions from two shapes.
  for (int64_t i = 0; i < rhs.rank(); ++i) {
    if (rhs.is_dynamic_dimension(i) || lhs.is_dynamic_dimension(i)) {
      output_dimensions_is_dynamic[i] = true;
    }
  }

  return ShapeUtil::MakeShape(ShapeUtil::HigherPrecisionElementType(lhs, rhs),
                              output_dimensions, output_dimensions_is_dynamic);
}

/* static */ StatusOr<Shape> ShapeInference::InferInDimBroadcastShape(
    const Shape& smaller_shape, const Shape& larger_shape,
    absl::Span<const int64_t> broadcast_dimensions) {
  if (broadcast_dimensions.empty() && !ShapeUtil::IsScalar(smaller_shape)) {
    // Reject "magic" inference for binops on different shapes, requiring
    // the user to provide an explicit broadcast dimension in this case.
    // See b/25177275 for more details.
    return InvalidArgument("Shapes must be equal rank, but are %s and %s",
                           ShapeUtil::HumanString(smaller_shape),
                           ShapeUtil::HumanString(larger_shape));
  } else if (broadcast_dimensions.size() != smaller_shape.rank()) {
    return InvalidArgument(
        "Size of broadcast_dimensions has to match lower-rank operand's "
        "rank; "
        " lower-rank operand's rank is %d, size of broadcast_dimensions is "
        "%u.",
        smaller_shape.rank(), broadcast_dimensions.size());
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
    int64_t dimension_to_match = broadcast_dimensions.at(i);
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
    int64_t small_dimension_size = smaller_shape.dimensions(i);
    int64_t large_dimension_size = larger_shape.dimensions(dimension_to_match);
    bool small_is_dynamic = smaller_shape.is_dynamic_dimension(i);
    bool large_is_dynamic =
        larger_shape.is_dynamic_dimension(dimension_to_match);
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
    if (small_is_dynamic != large_is_dynamic) {
      if (small_dimension_size == large_dimension_size ||
          (small_dimension_size == 1 && !small_is_dynamic) ||
          (large_dimension_size == 1 && !large_is_dynamic)) {
        // Do nothing. It's OK when the size-1 dimension is not static.
      } else {
        return InvalidArgument(
            "Broadcast dimension %d dynamism mismatch: %s and %s.", i,
            ShapeUtil::HumanString(smaller_shape),
            ShapeUtil::HumanString(larger_shape));
      }
    }
    // Make sure the broadcast dimensions are listed in a strictly increasing
    // order.
    if (i > 0 && broadcast_dimensions.at(i - 1) >= dimension_to_match) {
      return InvalidArgument(
          "Broadcast dimensions order is wrong: %d comes after %d.",
          dimension_to_match, broadcast_dimensions.at(i - 1));
    }

    output_shape.set_dimensions(dimension_to_match, small_dimension_size);
    output_shape.set_dynamic_dimension(dimension_to_match, small_is_dynamic);
  }

  return output_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferElementwiseBinaryOpShape(
    HloOpcode operation, const Shape& lhs, const Shape& rhs,
    absl::Span<const int64_t> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(lhs, "lhs of elementwise binary operation"));
  TF_RETURN_IF_ERROR(ExpectArray(rhs, "rhs of elementwise binary operation"));

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) {
    return InvalidArgument(
        "Binary op %s with different element types: %s and %s.",
        HloOpcodeString(operation), ShapeUtil::HumanString(lhs),
        ShapeUtil::HumanString(rhs));
  }

  if (lhs.rank() == rhs.rank()) {
    std::vector<int64_t> identity_dims(lhs.rank());
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
    Shape result = ShapeUtil::ChangeElementType(
        lhs, ShapeUtil::HigherPrecisionElementType(lhs, rhs));

    for (int64_t i = 0; i < rhs.rank(); ++i) {
      if (rhs.is_dynamic_dimension(i)) {
        result.set_dynamic_dimension(i, true);
      }
    }

    return result;

  } else if (lhs.rank() == rhs.rank()) {
    return InferDegenerateDimensionBroadcastShape(operation, lhs, rhs);
  } else {
    // Ranks do not match, so perform InDim broadcasting using
    // broadcast_dimensions. Scalar broadcasting is a special case of this.
    const Shape& larger_shape = lhs.rank() > rhs.rank() ? lhs : rhs;
    const Shape& smaller_shape = lhs.rank() > rhs.rank() ? rhs : lhs;

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
    absl::Span<const int64_t> broadcast_dimensions) {
  VLOG(2) << StrFormat(
      "inferring shape for <%s>(%s, %s) with broadcast_dimensions={%s}",
      HloOpcodeString(opcode), ShapeUtil::HumanStringWithLayout(lhs),
      ShapeUtil::HumanStringWithLayout(rhs),
      StrJoin(broadcast_dimensions, ", "));

  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));

  TF_RETURN_IF_ERROR(ExpectArray(
      lhs, absl::StrCat("lhs of binary operation ", HloOpcodeString(opcode))));
  TF_RETURN_IF_ERROR(ExpectArray(
      rhs, absl::StrCat("rhs of binary operation ", HloOpcodeString(opcode))));
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
      return InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                           broadcast_dimensions);

    case HloOpcode::kSubtract:
    case HloOpcode::kAtan2:
    case HloOpcode::kPower:
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
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
      } else if (lhs.element_type() == F64 && rhs.element_type() == F64) {
        return ShapeUtil::ChangeElementType(shape, C128);
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
    case HloOpcode::kCompare: {
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
        for (int64_t operand = 1; operand < operand_shapes.size(); ++operand) {
          if (!ShapeUtil::SameDimensions(*operand_shapes[0],
                                         *operand_shapes[operand])) {
            return InvalidArgument(
                "Sort keys and values dimensions must match. "
                "Keys shape is: %s\n, Values shape (operand index %lld) is: %s",
                ShapeUtil::HumanString(*operand_shapes[0]), operand,
                ShapeUtil::HumanString(*operand_shapes[operand]));
          }
        }
        return ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
      }
      return InvalidArgument("Unexpected number of operands for sort");
    }
    default:
      return InvalidArgument("Unknown operation %s.", HloOpcodeString(opcode));
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferMapShape(
    absl::Span<const Shape* const> arg_shapes, const ProgramShape& to_apply,
    absl::Span<const int64_t> dimensions) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Map expects at least one argument.");
  }

  // All arguments must have the same shape ignoring the element types.
  const Shape* arg_shape = arg_shapes[0];
  for (size_t i = 1; i < arg_shapes.size(); ++i) {
    TF_RETURN_IF_ERROR(ExpectArray(*arg_shapes[i], "operand of map"));

    if (ShapeUtil::CompatibleIgnoringElementType(*arg_shapes[i], *arg_shape)) {
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

    std::vector<std::string> pieces;
    pieces.reserve(arg_shapes.size());
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
                                                       *arg_shapes[i])) {
      return InvalidArgument(
          "Mapped computation's parameter type has to match argument element "
          "type; got parameter %d shape: %s, argument shape: %s.",
          i, ShapeUtil::HumanString(parameter_shape),
          ShapeUtil::HumanString(*arg_shape));
    }
  }

  return ShapeUtil::MakeShape(output_shape.element_type(),
                              arg_shape->dimensions());
}

/* static */ StatusOr<Shape> ShapeInference::InferBatchNormTrainingShape(
    const Shape& operand_shape, const Shape& scale_shape,
    const Shape& offset_shape, int64_t feature_index) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand of batch norm training"));
  TF_RETURN_IF_ERROR(
      ExpectArray(offset_shape, "offset input of batch norm training"));
  TF_RETURN_IF_ERROR(
      ExpectArray(scale_shape, "scale input of batch norm training"));

  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(operand_shape) ==
               OkStatus());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(offset_shape) ==
               OkStatus());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(scale_shape) ==
               OkStatus());

  if (feature_index >= operand_shape.rank()) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-training to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %d, and rank %d.",
        feature_index, operand_shape.rank());
  }

  if (feature_index < 0) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-training to "
        "be a non-negative number, got %d.",
        feature_index);
  }

  if (operand_shape.rank() < 1) {
    return InvalidArgument(
        "Expected the rank of operand to "
        "batch-norm-training to be at least 1; got %d.",
        operand_shape.rank());
  }

  if (offset_shape.rank() != 1) {
    return InvalidArgument(
        "Offset input of batch-norm-training must have"
        " rank 1, but has rank %d.",
        offset_shape.rank());
  }

  if (scale_shape.rank() != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-training must have"
        " rank 1, but has rank %d.",
        scale_shape.rank());
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

  const int64_t feature_count = operand_shape.dimensions(feature_index);
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

  return ShapeUtil::MakeTupleShapeWithPtrs({&operand_shape,
                                            &output_shape_for_mean_and_var,
                                            &output_shape_for_mean_and_var});
}

/* static */ StatusOr<Shape> ShapeInference::InferBatchNormInferenceShape(
    const Shape& operand_shape, const Shape& scale_shape,
    const Shape& offset_shape, const Shape& mean_shape,
    const Shape& variance_shape, int64_t feature_index) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand of batch norm inference"));
  TF_RETURN_IF_ERROR(
      ExpectArray(offset_shape, "offset input of batch norm inference"));
  TF_RETURN_IF_ERROR(
      ExpectArray(scale_shape, "scale input of batch norm inference"));

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(operand_shape));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(offset_shape));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(scale_shape));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(mean_shape));
  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(variance_shape));

  if (feature_index >= operand_shape.rank()) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-inference to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %d, and rank %d.",
        feature_index, operand_shape.rank());
  }

  if (feature_index < 0) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-inference to "
        "be a non-negative number, got %d.",
        feature_index);
  }

  if (operand_shape.rank() < 1) {
    return InvalidArgument(
        "Expected the rank of operand to "
        "batch-norm-inference to be at least 1; got %d.",
        operand_shape.rank());
  }

  if (offset_shape.rank() != 1) {
    return InvalidArgument(
        "Offset input of batch-norm-inference must have"
        " rank 1, but has rank %d.",
        offset_shape.rank());
  }

  if (scale_shape.rank() != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-inference must have"
        " rank 1, but has rank %d.",
        scale_shape.rank());
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

  const int64_t feature_count = operand_shape.dimensions(feature_index);
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
    const Shape& output_grad_shape, int64_t feature_index) {
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

  if (feature_index >= operand_shape.rank()) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-grad to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %d, and rank %d.",
        feature_index, operand_shape.rank());
  }

  if (operand_shape.rank() != output_grad_shape.rank()) {
    return InvalidArgument(
        "Expected operand_shape of batch-norm-grad to have the same rank as"
        " output_grad_shape; got rank(oprand_shape) %d, and"
        " rank(output_grad_shape) %d.",
        operand_shape.rank(), output_grad_shape.rank());
  }

  if (mean_shape.rank() != 1) {
    return InvalidArgument(
        "Mean input of batch-norm-grad must have"
        " rank 1, but has rank %d.",
        mean_shape.rank());
  }

  if (scale_shape.rank() != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-grad must have"
        " rank 1, but has rank %d.",
        scale_shape.rank());
  }

  if (var_shape.rank() != 1) {
    return InvalidArgument(
        "Var input of batch-norm-grad must have"
        " rank 1, but has rank %d.",
        var_shape.rank());
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

  const int64_t feature_count = operand_shape.dimensions(feature_index);

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
  for (int64_t i = 0; i < operand_shape.rank(); ++i) {
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

  return ShapeUtil::MakeTupleShapeWithPtrs(
      {&operand_shape, &feature_shape, &feature_shape});
}

/* static */ StatusOr<Shape> ShapeInference::InferConvolveShape(
    const Shape& lhs, const Shape& rhs, int64_t feature_group_count,
    int64_t batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dnums,
    absl::optional<PrimitiveType> preferred_element_type) {
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

  if (batch_group_count > 1 && feature_group_count > 1) {
    return InvalidArgument(
        "both batch_group_count %d and feature_group_count %d cannot be "
        "greater than 1",
        batch_group_count, feature_group_count);
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
  if (lhs.rank() != num_dims) {
    return InvalidArgument(
        "The LHS argument to a convolution should have rank %d; lhs: %s.",
        num_dims, ShapeUtil::HumanString(lhs));
  }
  if (rhs.rank() != num_dims) {
    return InvalidArgument(
        "The RHS argument to a convolution should have rank %d; rhs: %s.",
        num_dims, ShapeUtil::HumanString(rhs));
  }
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));

  // Verifies that the input and window dimensions are a permutation of
  // the dimension numbers.
  std::vector<int64_t> input_dnums(num_dims);
  input_dnums[0] = dnums.input_batch_dimension();
  input_dnums[1] = dnums.input_feature_dimension();
  std::copy(dnums.input_spatial_dimensions().begin(),
            dnums.input_spatial_dimensions().end(), input_dnums.begin() + 2);
  absl::c_sort(input_dnums);

  std::vector<int64_t> window_dnums(num_dims);
  window_dnums[0] = dnums.kernel_input_feature_dimension();
  window_dnums[1] = dnums.kernel_output_feature_dimension();
  std::copy(dnums.kernel_spatial_dimensions().begin(),
            dnums.kernel_spatial_dimensions().end(), window_dnums.begin() + 2);
  absl::c_sort(window_dnums);

  std::vector<int64_t> output_dnums(num_dims);
  output_dnums[0] = dnums.output_batch_dimension();
  output_dnums[1] = dnums.output_feature_dimension();
  std::copy(dnums.output_spatial_dimensions().begin(),
            dnums.output_spatial_dimensions().end(), output_dnums.begin() + 2);
  absl::c_sort(output_dnums);

  std::vector<int64_t> expected_dnums(num_dims);
  std::iota(expected_dnums.begin(), expected_dnums.end(), 0);

  const auto in_range = [num_dims](int64_t i) {
    return 0 <= i && i < num_dims;
  };
  if (!absl::c_all_of(input_dnums, in_range) ||
      !absl::c_all_of(window_dnums, in_range) ||
      !absl::c_all_of(output_dnums, in_range)) {
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

  std::vector<int64_t> input_spatial_dims(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_spatial_dims[i] = lhs.dimensions(dnums.input_spatial_dimensions(i));
  }
  const int64_t input_features =
      lhs.dimensions(dnums.input_feature_dimension());
  const int64_t input_batch = lhs.dimensions(dnums.input_batch_dimension());

  std::vector<int64_t> kernel_spatial_dims(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    kernel_spatial_dims[i] = rhs.dimensions(dnums.kernel_spatial_dimensions(i));
  }
  const int64_t kernel_input_features =
      rhs.dimensions(dnums.kernel_input_feature_dimension());
  const int64_t kernel_output_features =
      rhs.dimensions(dnums.kernel_output_feature_dimension());

  if (kernel_output_features % batch_group_count != 0) {
    return InvalidArgument(
        "Expected output feature dimension size (value %d) to be a multiple of "
        "batch group count %d; got <conv>(%s, %s)\n"
        "Dimension numbers: {%s}.",
        kernel_output_features, batch_group_count, ShapeUtil::HumanString(lhs),
        ShapeUtil::HumanString(rhs), dnums.DebugString());
  }

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
    // A depthwise/grouped filter has the shape
    // [space0, .. spaceN, GROUP_SIZE, NUM_OUTPUT_FEATURES]. When
    // [space0, .. spaceN, GROUP_SIZE] is convolved with the input, a shape
    // [space0, .. spaceN, feature_group_count] is formed. Therefore, the output
    // feature count (which is equal to kernel output features) has to be a
    // multiple of feature_group_count.
    return InvalidArgument(
        "Expected output feature dimension (value %d) to be divisible by "
        "feature_group_count (value %d); "
        "got <conv>(%s, %s)\n"
        "Dimension numbers: {%s}.",
        kernel_output_features, feature_group_count,
        ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs),
        dnums.DebugString());
  }

  if (input_batch % batch_group_count != 0) {
    return InvalidArgument(
        "Expected input batch dimension (value %d) to be divisible by "
        "batch_group_count (value %d); "
        "got <conv>(%s, %s)\n"
        "Dimension numbers: {%s}.",
        input_batch, batch_group_count, ShapeUtil::HumanString(lhs),
        ShapeUtil::HumanString(rhs), dnums.DebugString());
  }

  std::vector<int64_t> window_dims(num_spatial_dims);
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
      InferWindowOutputShape(base_shape, window, lhs.element_type()));

  std::vector<int64_t> dimensions(num_dims);
  dimensions[dnums.output_batch_dimension()] = input_batch / batch_group_count;
  dimensions[dnums.output_feature_dimension()] = kernel_output_features;

  for (int i = 0; i < num_spatial_dims; ++i) {
    dimensions[dnums.output_spatial_dimensions(i)] =
        window_output_shape.dimensions(i);
  }
  std::vector<bool> is_dynamic(num_dims);
  for (int i = 0; i < num_dims; i++) {
    if (lhs.is_dynamic_dimension(i)) {
      if (i == dnums.input_batch_dimension()) {
        is_dynamic[dnums.output_batch_dimension()] = true;
      } else if (i == dnums.input_feature_dimension()) {
        // Input feature dimension is a contracting dimension, which does not
        // affect the output dimension size. So we need to do nothing.
      } else {
        for (int64_t j = 0; j < dnums.output_spatial_dimensions_size(); ++j) {
          if (i == dnums.input_spatial_dimensions(j)) {
            // i is a spatial dimension, find corresponding output spatial
            // dimension.
            is_dynamic[dnums.output_spatial_dimensions(j)] = true;
          }
        }
      }
    }
    if (rhs.is_dynamic_dimension(i)) {
      if (i == dnums.kernel_input_feature_dimension()) {
        // Kernel feature dimension does not affect the output dimension size.
        // So we need to do nothing.
      } else if (i == dnums.kernel_output_feature_dimension()) {
        return InvalidArgument(
            "Dynamic output feature dim on convolution kernel is not "
            "supported: rhs shape is %s ",
            rhs.ToString());
      } else {
        for (int64_t j = 0; j < dnums.kernel_spatial_dimensions_size(); ++j) {
          if (i == dnums.kernel_spatial_dimensions(j)) {
            // i is a spatial dimension, find corresponding output spatial
            // dimension.
            is_dynamic[dnums.output_spatial_dimensions(j)] = true;
          }
        }
      }
    }
  }
  TF_ASSIGN_OR_RETURN(
      PrimitiveType type,
      MaybeUpcast(ShapeUtil::HigherPrecisionElementType(lhs, rhs),
                  preferred_element_type));
  return ShapeUtil::MakeShape(type, dimensions, is_dynamic);
}

/* static */ StatusOr<Shape> ShapeInference::InferFftShape(
    const Shape& in, const FftType fft_type,
    const absl::Span<const int64_t> fft_length) {
  const int64_t fft_rank = fft_length.size();
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
      if (!primitive_util::IsComplexType(in.element_type())) {
        return InvalidArgument("%s requires complex input type, found %s.",
                               FftType_Name(fft_type),
                               PrimitiveType_Name(in.element_type()));
      }
      RET_CHECK_RANK(in);
      return in;
    case RFFT: {
      if (in.element_type() != F32 && in.element_type() != F64) {
        return InvalidArgument("RFFT requires F32 or F64 input type, found %s.",
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
      Shape result = ShapeUtil::ChangeElementType(
          in, in.element_type() == F32 ? C64 : C128);
      // Preserve the size of zero-sized dimensions.
      if (fft_length[fft_rank - 1] != 0) {
        result.set_dimensions(result.dimensions_size() - 1,
                              fft_length[fft_rank - 1] / 2 + 1);
      }
      return result;
    }
    case IRFFT: {
      if (!primitive_util::IsComplexType(in.element_type())) {
        return InvalidArgument("IRFFT requires complex input type, found %s.",
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
      // The size of zero-sized dimensions is preserved.
      if ((in.dimensions(in.dimensions_size() - 1) != 0 ||
           fft_length[fft_rank - 1] != 0) &&
          in.dimensions(in.dimensions_size() - 1) !=
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

/* static */ StatusOr<Shape> ShapeInference::InferTriangularSolveShape(
    const Shape& a, const Shape& b, const TriangularSolveOptions& options) {
  if ((!ShapeUtil::ElementIsFloating(a) && !ShapeUtil::ElementIsComplex(a)) ||
      a.element_type() != b.element_type()) {
    return InvalidArgument(
        "Expected element types in shape to be floating or complex and "
        "identical for TriangularSolve; got %s and %s.",
        PrimitiveType_Name(a.element_type()),
        PrimitiveType_Name(b.element_type()));
  }
  if (a.rank() < 2) {
    return InvalidArgument(
        "The 'a' argument to TriangularSolve must have rank >= 2, got shape %s",
        a.ToString());
  }
  if (b.rank() != a.rank()) {
    return InvalidArgument(
        "Arguments to triangular solve must have equal rank; got %s and %s.",
        b.ToString(), a.ToString());
  }
  if (a.dimensions(a.rank() - 2) != a.dimensions(a.rank() - 1)) {
    return InvalidArgument(
        "The two minor dimensions of 'a' must have equal size, got %s.",
        a.ToString());
  }
  if (a.dimensions(a.rank() - 1) !=
      b.dimensions(b.rank() - (options.left_side() ? 2 : 1))) {
    return InvalidArgument(
        "The shared dimension of 'a' and 'b' does not match, got shapes %s and "
        "%s",
        a.ToString(), b.ToString());
  }
  absl::Span<const int64_t> a_batch_dims(a.dimensions());
  absl::Span<const int64_t> b_batch_dims(b.dimensions());
  a_batch_dims.remove_suffix(2);
  b_batch_dims.remove_suffix(2);
  if (a_batch_dims != b_batch_dims) {
    return InvalidArgument(
        "The leading batch dimensions of the arguments to triangular solve "
        "must be equal; got %s and %s.",
        b.ToString(), a.ToString());
  }
  if (!TriangularSolveOptions_Transpose_IsValid(options.transpose_a()) ||
      options.transpose_a() == TriangularSolveOptions::TRANSPOSE_INVALID) {
    return InvalidArgument(
        "Invalid transpose option value for triangular solve (%d).\n",
        options.transpose_a());
  }
  return b;
}

/* static */ StatusOr<Shape> ShapeInference::InferCholeskyShape(
    const Shape& a) {
  if (!ShapeUtil::ElementIsFloating(a) && !ShapeUtil::ElementIsComplex(a)) {
    return InvalidArgument(
        "Expected element type in shape to be floating or complex for "
        "Cholesky; got %s.",
        PrimitiveType_Name(a.element_type()));
  }
  if (a.rank() < 2) {
    return InvalidArgument(
        "The 'a' argument to Cholesky must have rank >= 2, got shape %s",
        a.ToString());
  }
  if (a.dimensions(a.rank() - 2) != a.dimensions(a.rank() - 1)) {
    return InvalidArgument(
        "The two minor dimensions of 'a' must have equal size, got %s.",
        a.ToString());
  }
  return a;
}

/* static */ StatusOr<Shape> ShapeInference::InferAllGatherShape(
    absl::Span<const Shape* const> operand_shapes, int64_t all_gather_dimension,
    int64_t shard_count) {
  TF_RET_CHECK(all_gather_dimension >= 0);
  TF_RET_CHECK(shard_count > 0);

  std::vector<Shape> output_shapes;
  output_shapes.reserve(operand_shapes.size());
  for (const Shape* operand_shape : operand_shapes) {
    TF_RET_CHECK(all_gather_dimension < operand_shape->rank());
    TF_RETURN_IF_ERROR(ExpectArray(*operand_shape, "operand of all-gather"));

    Shape output_shape = *operand_shape;
    output_shape.set_dimensions(
        all_gather_dimension,
        shard_count * output_shape.dimensions(all_gather_dimension));
    output_shapes.push_back(output_shape);
  }
  if (output_shapes.size() == 1) {
    return output_shapes[0];
  }
  return ShapeUtil::MakeTupleShape(output_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferAllGatherStartShape(
    absl::Span<const Shape* const> operand_shapes, int64_t all_gather_dimension,
    int64_t shard_count) {
  TF_ASSIGN_OR_RETURN(
      Shape ag_shape,
      InferAllGatherShape(operand_shapes, all_gather_dimension, shard_count));
  Shape input_shape;
  if (operand_shapes.size() == 1) {
    input_shape = *operand_shapes[0];
  } else {
    input_shape = ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
  }
  return ShapeUtil::MakeTupleShapeWithPtrs({&input_shape, &ag_shape});
}

/* static */ StatusOr<Shape> ShapeInference::InferAllGatherDoneShape(
    const Shape& all_gather_start_shape) {
  return ShapeUtil::GetTupleElementShape(all_gather_start_shape, 1);
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
  return ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceScatterShape(
    absl::Span<const Shape* const> operand_shapes, int64_t scatter_dimension,
    int64_t shard_count) {
  TF_RET_CHECK(scatter_dimension >= 0);
  TF_RET_CHECK(shard_count > 0);

  std::vector<Shape> output_shapes;
  output_shapes.reserve(operand_shapes.size());
  for (const Shape* operand_shape : operand_shapes) {
    TF_RET_CHECK(scatter_dimension < operand_shape->rank());
    TF_RETURN_IF_ERROR(
        ExpectArray(*operand_shape, "operand of reduce-scatter"));

    int64_t scatter_dim_input_size =
        operand_shape->dimensions(scatter_dimension);
    if (scatter_dim_input_size % shard_count != 0) {
      return InvalidArgument(
          "ReduceScatter operand scatter dimension size %d must be "
          "dividable by shard_count "
          "%d.",
          scatter_dim_input_size, shard_count);
    }

    Shape output_shape = *operand_shape;
    output_shape.set_dimensions(scatter_dimension,
                                scatter_dim_input_size / shard_count);
    output_shapes.push_back(output_shape);
  }

  if (output_shapes.size() == 1) {
    return output_shapes[0];
  }
  return ShapeUtil::MakeTupleShape(output_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferAllReduceStartShape(
    absl::Span<const Shape* const> operand_shapes) {
  return InferAllReduceShape(operand_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferAllReduceDoneShape(
    const Shape& operand_shape) {
  // The returned value from AllReduceDone is the operand forwarded.
  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferAllToAllShape(
    const Shape& shape, int64_t split_dimension, int64_t concat_dimension,
    int64_t split_count) {
  TF_RET_CHECK(split_count > 0);
  if (split_dimension >= shape.rank() || split_dimension < 0) {
    return InvalidArgument(
        "AllToAll split_dimension %d is out-of-bounds in shape %s.",
        split_dimension, ShapeUtil::HumanString(shape));
  }
  if (concat_dimension >= shape.rank() || concat_dimension < 0) {
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
  std::vector<int64_t> new_dimensions(shape.dimensions().begin(),
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
    if (!Shape::Equal().IgnoreMemorySpaceInLayout()(*operand_shapes[0],
                                                    *operand_shapes[i])) {
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
    absl::Span<const Shape* const> operand_shapes) {
  if (operand_shapes.size() == 1) {
    TF_RETURN_IF_ERROR(
        ExpectArray(*(operand_shapes[0]), "operand of collective-permute"));
    return *(operand_shapes[0]);
  } else {
    TF_RET_CHECK(operand_shapes.size() == 4);
    return *(operand_shapes[1]);
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferCollectivePermuteStartShape(
    absl::Span<const Shape* const> operand_shapes) {
  const Shape u32_scalar = ShapeUtil::MakeShape(U32, {});
  if (operand_shapes.size() == 1) {
    TF_RETURN_IF_ERROR(ExpectArray(*(operand_shapes[0]),
                                   "operand of collective-permute-start"));
    return ShapeUtil::MakeTupleShapeWithPtrs(
        {operand_shapes[0], operand_shapes[0], &u32_scalar, &u32_scalar});
  } else {
    TF_RET_CHECK(operand_shapes.size() == 4);
    return ShapeUtil::MakeTupleShapeWithPtrs(
        {operand_shapes[0], operand_shapes[1], &u32_scalar, &u32_scalar});
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferCollectivePermuteDoneShape(
    const Shape& operand_shape) {
  TF_RET_CHECK(operand_shape.IsTuple());
  return ShapeUtil::GetTupleElementShape(operand_shape, 1);
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceShape(
    absl::Span<const Shape* const> arg_shapes,
    absl::Span<const int64_t> dimensions_to_reduce,
    const ProgramShape& to_apply) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Reduce must have at least 2 arguments, has 0");
  }
  if (arg_shapes.size() % 2) {
    return InvalidArgument(
        "Reduce must have an even number of arguments, has %lu",
        arg_shapes.size());
  }
  int64_t num_reduced_args = arg_shapes.size() / 2;
  auto reduced_args = arg_shapes.subspan(0, num_reduced_args);
  // Check that all of the reduced tensors have the same dimensions. The element
  // types may be different.
  for (int64_t i = 1; i < num_reduced_args; ++i) {
    if (!ShapeUtil::SameDimensions(*reduced_args[0], *reduced_args[i])) {
      return InvalidArgument(
          "All reduced tensors must have the same dimension. Tensor 0 has "
          "shape %s, Tensor %d has shape %s",
          ShapeUtil::HumanString(*reduced_args[0]), i,
          ShapeUtil::HumanString(*reduced_args[i]));
    }
  }
  // Check that the dimensions to reduce are in-bounds for the given shape.
  // We've already verified all reduced tensors have the same dimensions, so it
  // doesn't matter which one we choose.
  const Shape& arg = *reduced_args[0];
  for (int64_t dimension : dimensions_to_reduce) {
    if (dimension >= arg.rank() || dimension < 0) {
      return InvalidArgument("Reducing out-of-bounds dimension %d in shape %s.",
                             dimension, ShapeUtil::HumanString(arg));
    }
  }

  auto init_values = arg_shapes.subspan(num_reduced_args, arg_shapes.size());
  std::vector<PrimitiveType> element_types;
  element_types.reserve(reduced_args.size());
  for (const Shape* arg : reduced_args) {
    element_types.push_back(arg->element_type());
  }
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply, init_values, element_types,
                                        num_reduced_args));

  absl::flat_hash_set<int64_t> dimensions_to_reduce_set;
  for (int64_t dim_to_reduce : dimensions_to_reduce) {
    if (!dimensions_to_reduce_set.insert(dim_to_reduce).second) {
      return InvalidArgument("Duplicate reduction dimension: %d",
                             dim_to_reduce);
    }
  }

  std::vector<int64_t> new_dimensions;
  std::vector<bool> new_is_dynamic;
  for (int i = 0; i < arg.rank(); ++i) {
    if (dimensions_to_reduce_set.find(i) == dimensions_to_reduce_set.end()) {
      new_dimensions.push_back(arg.dimensions(i));
      new_is_dynamic.push_back(arg.is_dynamic_dimension(i));
    }
  }

  if (ShapeUtil::IsScalar(to_apply.result())) {
    return ShapeUtil::MakeShape(to_apply.result().element_type(),
                                new_dimensions, new_is_dynamic);
  } else {
    std::vector<Shape> result_subshapes;
    const auto& tuple_shapes = to_apply.result().tuple_shapes();
    result_subshapes.reserve(tuple_shapes.size());
    for (const Shape& subshape : tuple_shapes) {
      result_subshapes.push_back(ShapeUtil::MakeShape(
          subshape.element_type(), new_dimensions, new_is_dynamic));
    }
    return ShapeUtil::MakeTupleShape(result_subshapes);
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceWindowShape(
    const Shape& operand_shape, const Shape& init_value_shape,
    const Window& window, const ProgramShape& to_apply_shape) {
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply_shape, {&init_value_shape},
                                        {operand_shape.element_type()},
                                        /*inputs=*/1));
  return InferReduceWindowShape(operand_shape, init_value_shape, window);
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceWindowShape(
    absl::Span<const Shape* const> operands,
    absl::Span<const Shape* const> init_values, const Window& window,
    const ProgramShape& to_apply_shape) {
  auto number_of_input = operands.size();
  // Check that all of the reduced tensors have the same dimensions. The element
  // types may be different.
  for (int64_t i = 1; i < number_of_input; ++i) {
    if (!ShapeUtil::SameDimensions(*operands[0], *operands[i])) {
      return InvalidArgument(
          "All reduced tensors must have the same dimension. Tensor 0 has "
          "shape %s, Tensor %d has shape %s",
          ShapeUtil::HumanString(*operands[0]), i,
          ShapeUtil::HumanString(*operands[i]));
    }
  }
  std::vector<PrimitiveType> operand_element_type_vec;
  operand_element_type_vec.reserve(operands.size());
  for (const Shape* s : operands) {
    operand_element_type_vec.push_back(s->element_type());
  }
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply_shape, init_values,
                                        operand_element_type_vec,
                                        /*inputs=*/number_of_input));
  std::vector<Shape> output_shape_vec;
  const size_t n = operands.size();
  output_shape_vec.reserve(n);
  for (size_t i = 0; i < operands.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        auto cur_output_shape,
        InferReduceWindowShape(*operands[i], *init_values[i], window));
    output_shape_vec.push_back(cur_output_shape);
  }
  if (ShapeUtil::IsScalar(to_apply_shape.result())) {
    CHECK_EQ(output_shape_vec.size(), 1);
    return output_shape_vec[0];
  } else {
    return ShapeUtil::MakeTupleShape(output_shape_vec);
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceWindowShape(
    const Shape& operand_shape, const Shape& init_value_shape,
    const Window& window) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of reduce-window"));
  return InferWindowOutputShape(operand_shape, window,
                                init_value_shape.element_type());
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
                                             operand_shape.element_type()));
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
    const Shape& shape, int64_t dimension) {
  if (dimension < 0 || dimension >= shape.rank()) {
    return InvalidArgument("GetDimensionSize dimension out of bounds: %d.",
                           dimension);
  }

  // TODO(b/119580730): Remove this restriction when very large dimension size
  // is needed.
  if (shape.dimensions(dimension) > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument(
        "GetDimensionSize's input shape is %s, the %dth dimension exceeds the "
        "INT_MAX limit.",
        ShapeUtil::HumanString(shape), dimension);
  }

  return ShapeUtil::MakeShape(S32, {});
}

/* static */ StatusOr<Shape> ShapeInference::InferSetDimensionSizeShape(
    const Shape& shape, const Shape& val_shape, int64_t dimension) {
  if (dimension < 0 || dimension >= shape.rank()) {
    return InvalidArgument("SetDimensionSize dimension out of bounds: %d.",
                           dimension);
  }

  if (val_shape.rank() != 0 || val_shape.element_type() != S32) {
    return InvalidArgument(
        "SetDimensionSize's value has to be S32 scalar, got %s",
        val_shape.ToString());
  }
  // TODO(b/119580730): Remove this restriction when very large dimension size
  // is needed.
  if (shape.dimensions(dimension) > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument(
        "SetDimensionSize's input shape is %s, the %dth dimension exceeds the "
        "INT_MAX limit.",
        ShapeUtil::HumanString(shape), dimension);
  }

  Shape result = shape;
  result.set_dynamic_dimension(dimension, true);
  return result;
}

/* static */ StatusOr<Window> ShapeInference::InferWindowFromDimensions(
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation) {
  const auto verify_size = [&](const size_t x, const char* x_name) {
    if (x == 0 || x == window_dimensions.size()) {
      return OkStatus();
    } else {
      return InvalidArgument(
          "%s", absl::StrCat(
                    "Window has different number of window dimensions than of ",
                    x_name,
                    "\nNumber of window dimensions: ", window_dimensions.size(),
                    "\nNumber of ", x_name, ": ", x, "\n"));
    }
  };
  TF_RETURN_IF_ERROR(verify_size(window_strides.size(), "window strides"));
  TF_RETURN_IF_ERROR(verify_size(padding.size(), "padding entries"));
  TF_RETURN_IF_ERROR(verify_size(lhs_dilation.size(), "lhs dilation factors"));
  TF_RETURN_IF_ERROR(verify_size(rhs_dilation.size(), "rhs dilation factors"));

  Window window;
  for (size_t i = 0; i < window_dimensions.size(); i++) {
    auto dim = window.add_dimensions();
    dim->set_size(window_dimensions[i]);
    if (!window_strides.empty()) {
      dim->set_stride(window_strides[i]);
    } else {
      dim->set_stride(1);
    }
    if (!padding.empty()) {
      dim->set_padding_low(padding[i].first);
      dim->set_padding_high(padding[i].second);
    } else {
      dim->set_padding_low(0);
      dim->set_padding_high(0);
    }
    if (!lhs_dilation.empty()) {
      dim->set_base_dilation(lhs_dilation[i]);
    } else {
      dim->set_base_dilation(1);
    }
    if (!rhs_dilation.empty()) {
      dim->set_window_dilation(rhs_dilation[i]);
    } else {
      dim->set_window_dilation(1);
    }
    dim->set_window_reversal(false);
  }
  return window;
}

/* static */ StatusOr<Shape> ShapeInference::InferSliceShape(
    const Shape& arg, absl::Span<const int64_t> starts,
    absl::Span<const int64_t> limits, absl::Span<const int64_t> strides) {
  auto error = [&](const std::string& message) {
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

  if (starts.size() != arg.rank()) {
    return InvalidArgument(
        "Slice index count does not match argument rank: %u vs %d.",
        starts.size(), arg.rank());
  }

  std::vector<int64_t> sizes;
  const auto starts_size = starts.size();
  sizes.reserve(starts_size);
  for (int64_t dimension = 0; dimension < starts_size; ++dimension) {
    int64_t start_index = starts[dimension];
    int64_t limit_index = limits[dimension];
    int64_t stride = strides[dimension];
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

  std::vector<bool> is_dynamic(arg.rank());
  for (int64_t i = 0; i < arg.dimensions_size(); ++i) {
    // Slicing 1 out of a dynamic dimension eliminates the dynamic dimension.
    if (sizes[i] == 1) {
      continue;
    }
    is_dynamic[i] = arg.is_dynamic_dimension(i);
  }

  return ShapeUtil::MakeShape(arg.element_type(), sizes, is_dynamic);
}

/* static */ StatusOr<Shape> ShapeInference::InferDynamicSliceShape(
    const Shape& operand_shape, absl::Span<const Shape> start_index_shapes,
    absl::Span<const int64_t> slice_sizes, bool allow_scalar_indices) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of dynamic slice"));
  auto number_of_indices = start_index_shapes.size();
  // TODO(b/118437727): Remove this path.
  if (!allow_scalar_indices ||
      (number_of_indices >= 1 && start_index_shapes[0].rank() == 1)) {
    if (number_of_indices != 1) {
      return InvalidArgument(
          "Dynamic slice should have exactly 1 index operand, has %d.",
          number_of_indices);
    }

    const Shape& start_indices_shape = start_index_shapes[0];
    VLOG(2) << StrFormat(
        "slicing shape %s at dynamic start_indices %s with slice_sizes={%s}",
        ShapeUtil::HumanString(operand_shape),
        ShapeUtil::HumanString(start_indices_shape),
        StrJoin(slice_sizes, ", "));

    TF_RETURN_IF_ERROR(
        ExpectArray(start_indices_shape, "start indices of dynamic slice"));

    if (start_indices_shape.rank() != 1) {
      return InvalidArgument(
          "Dynamic slice start indices of rank %d must be rank1.",
          start_indices_shape.rank());
    }

    if (!ShapeUtil::ElementIsIntegral(start_indices_shape)) {
      return InvalidArgument(
          "Dynamic slice start indices must be of integral type.");
    }

    const int64_t start_num_dims = start_indices_shape.dimensions(0);
    if (operand_shape.rank() != start_num_dims) {
      return InvalidArgument(
          "Dynamic slice start number of dimensions %d (%s) must match rank "
          "%d of slice input (%s).",
          start_num_dims, ShapeUtil::HumanString(start_indices_shape),
          operand_shape.rank(), ShapeUtil::HumanString(operand_shape));
    }
  } else {
    VLOG(2) << StrFormat("slicing shape %s a with slice_sizes={%s}",
                         ShapeUtil::HumanString(operand_shape),
                         StrJoin(slice_sizes, ", "));

    if (operand_shape.rank() != number_of_indices) {
      return InvalidArgument(
          "Dynamic slice start number of dimensions %d must match rank "
          "%d of slice input (%s).",
          number_of_indices, operand_shape.rank(),
          ShapeUtil::HumanString(operand_shape));
    }

    if (number_of_indices > 0) {
      const Shape& first_index_shape = start_index_shapes[0];
      if (!ShapeUtil::IsScalar(first_index_shape)) {
        return InvalidArgument("Dynamic slice indices must be scalar, not %s.",
                               ShapeUtil::HumanString(first_index_shape));
      }
      if (!ShapeUtil::ElementIsIntegral(first_index_shape)) {
        return InvalidArgument(
            "Dynamic slice start indices must be of integral type.");
      }
      for (const Shape& index_shape : start_index_shapes) {
        if (!ShapeUtil::Compatible(first_index_shape, index_shape)) {
          return InvalidArgument(
              "Dynamic slice start indices must all have the same shape, got "
              "mismatching indices with shapes %s and %s.",
              ShapeUtil::HumanString(first_index_shape),
              ShapeUtil::HumanString(index_shape));
        }
      }
    }
  }

  if (slice_sizes.size() != operand_shape.rank()) {
    return InvalidArgument(
        "Dynamic slice index count does not match argument rank: %u vs %d.",
        slice_sizes.size(), operand_shape.rank());
  }

  for (int64_t dim = 0; dim < slice_sizes.size(); ++dim) {
    const int64_t input_dim_size = operand_shape.dimensions(dim);
    const int64_t slice_dim_size = slice_sizes[dim];
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
    absl::Span<const Shape> start_index_shapes, bool allow_scalar_indices) {
  TF_RETURN_IF_ERROR(
      ExpectArray(operand_shape, "operand of dynamic update slice"));
  TF_RETURN_IF_ERROR(
      ExpectArray(update_shape, "update of dynamic update slice"));

  auto number_of_indices = start_index_shapes.size();
  // TODO(b/118437727): Remove this path.
  if (!allow_scalar_indices ||
      (number_of_indices >= 1 && start_index_shapes[0].rank() == 1)) {
    if (number_of_indices != 1) {
      return InvalidArgument(
          "Dynamic update slice should have exactly 1 index operand, has %d.",
          number_of_indices);
    }
    const Shape& start_indices_shape = start_index_shapes[0];
    TF_RETURN_IF_ERROR(ExpectArray(start_indices_shape,
                                   "start indices of dynamic update slice"));

    VLOG(2) << StrFormat(
        "updating slice of shape %s at dynamic start_indices %s with update "
        "shape %s",
        ShapeUtil::HumanString(operand_shape),
        ShapeUtil::HumanString(start_indices_shape),
        ShapeUtil::HumanString(update_shape));

    if (start_indices_shape.rank() != 1) {
      return InvalidArgument(
          "Dynamic update slice start indices of rank %d must be rank1.",
          start_indices_shape.rank());
    }

    if (!ShapeUtil::ElementIsIntegral(start_indices_shape)) {
      return InvalidArgument(
          "Dynamic update slice start indices must be of integral type.");
    }

    const int64_t start_num_dims = start_indices_shape.dimensions(0);
    if (operand_shape.rank() != start_num_dims) {
      return InvalidArgument(
          "Dynamic update slice start number of dimensions %d (%s) must match "
          "rank %d of slice input (%s).",
          start_num_dims, ShapeUtil::HumanString(start_indices_shape),
          operand_shape.rank(), ShapeUtil::HumanString(operand_shape));
    }
  } else {
    VLOG(2) << StrFormat("updating slice of shape %s with update shape %s",
                         ShapeUtil::HumanString(operand_shape),
                         ShapeUtil::HumanString(update_shape));

    if (operand_shape.rank() != number_of_indices) {
      return InvalidArgument(
          "Dynamic update slice start number of dimensions %d must match "
          "rank %d of slice input (%s).",
          number_of_indices, operand_shape.rank(),
          ShapeUtil::HumanString(operand_shape));
    }

    if (number_of_indices > 0) {
      const Shape& first_index_shape = start_index_shapes[0];
      if (!ShapeUtil::IsScalar(first_index_shape)) {
        return InvalidArgument(
            "Dynamic update slice indices must be scalar, not %s.",
            ShapeUtil::HumanString(first_index_shape));
      }
      if (!ShapeUtil::ElementIsIntegral(first_index_shape)) {
        return InvalidArgument(
            "Dynamic update slice start indices must be of integral type.");
      }
      for (const Shape& index_shape : start_index_shapes) {
        if (!ShapeUtil::Compatible(first_index_shape, index_shape)) {
          return InvalidArgument(
              "Dynamic update slice start indices must all have the same "
              "shape, got mismatching indices with shapes %s and %s.",
              ShapeUtil::HumanString(first_index_shape),
              ShapeUtil::HumanString(index_shape));
        }
      }
    }
  }

  if (update_shape.rank() != operand_shape.rank()) {
    return InvalidArgument(
        "Dynamic update slice update rank does not match argument rank: "
        "%d vs %d.",
        update_shape.rank(), operand_shape.rank());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(operand_shape,
                                                     update_shape)) {
    return InvalidArgument(
        "Dynamic update slice update element type does not match argument. "
        "operand.element_type: %s vs update.element_type: %s.",
        PrimitiveType_Name(operand_shape.element_type()),
        PrimitiveType_Name(update_shape.element_type()));
  }

  for (int64_t dim = 0; dim < operand_shape.rank(); ++dim) {
    const int64_t input_dim_size = operand_shape.dimensions(dim);
    const int64_t update_dim_size = update_shape.dimensions(dim);
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

  auto result_shape = operand_shape;

  // If any of the operand shape is dynamic, the result dimension is also
  // dynamic.
  // If update shape is dynamic, only propagate dynamic dimension to result if
  // the update is a full update (update_shape[i] == operand_shape[i]).
  for (int64_t i = 0; i < update_shape.rank(); ++i) {
    if (operand_shape.is_dynamic_dimension(i)) {
      result_shape.set_dynamic_dimension(i, true);
    }

    if (update_shape.is_dynamic_dimension(i) &&
        update_shape.dimensions(i) == operand_shape.dimensions(i)) {
      // When update/replace a full dimension, propagate dynamic dimension to
      // the result.
      result_shape.set_dynamic_dimension(i, true);
    }
  }

  return result_shape;
}

/*static */ StatusOr<Shape> ShapeInference::InferReverseShape(
    const Shape& operand_shape, absl::Span<const int64_t> dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of reverse"));
  if (!AllUnique(dimensions)) {
    return InvalidArgument("a dimension number is duplicated in reverse");
  }
  for (int64_t dimension : dimensions) {
    if (dimension >= operand_shape.rank() || dimension < 0) {
      return InvalidArgument(
          "One of the reverse dimensions (%d) is out-of-bounds in shape %s.",
          dimension, ShapeUtil::HumanString(operand_shape));
    }
  }
  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferGetTupleElementShape(
    const Shape& arg, int64_t index) {
  if (!arg.IsTuple()) {
    return InvalidArgument(
        "Cannot infer shape: attempting to index into non-tuple: %s.",
        ShapeUtil::HumanString(arg));
  }

  if (index < 0 || index >= arg.tuple_shapes_size()) {
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
  if (!ShapeUtil::Compatible(condition.result(),
                             ShapeUtil::MakeShape(PRED, {}))) {
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
    const Shape& branch_index,
    absl::Span<const ProgramShape> branch_computations,
    absl::Span<const Shape> branch_operands) {
  if (!ShapeUtil::Compatible(branch_index, ShapeUtil::MakeShape(PRED, {})) &&
      !ShapeUtil::Compatible(branch_index, ShapeUtil::MakeShape(S32, {}))) {
    return InvalidArgument("branch_index must be bool or int32_t; got %s.",
                           ShapeUtil::HumanString(branch_index));
  }
  if (branch_index.element_type() == PRED) {
    TF_RET_CHECK(2 == branch_computations.size());
  } else {
    TF_RET_CHECK(!branch_computations.empty());
  }
  TF_RET_CHECK(branch_computations.size() == branch_operands.size());
  Shape result = branch_computations[0].result();
  for (int j = 0; j < branch_computations.size(); ++j) {
    if (branch_computations[j].parameters_size() != 1) {
      return InvalidArgument(
          "branch computation %d must take 1 argument; got %d.", j,
          branch_computations[j].parameters_size());
    }
    if (!ShapeUtil::Compatible(branch_computations[j].parameters(0),
                               branch_operands[j])) {
      auto shape_string = [&]() {
        return StrFormat("operand: %s; computation: %s",
                         ShapeUtil::HumanString(branch_operands[j]),
                         ShapeUtil::HumanString(branch_computations[j]));
      };
      return InvalidArgument(
          "branch operand %d must match the shape of the only parameter of "
          "branch computation %d: got %s.",
          j, j, shape_string());
    }

    if (!ShapeUtil::Compatible(branch_computations[0].result(),
                               branch_computations[j].result())) {
      auto shape_string = [&]() {
        return StrFormat(
            "branch 0 computation result: %s; branch %d computation result: %s",
            ShapeUtil::HumanString(branch_computations[0].result()), j,
            ShapeUtil::HumanString(branch_computations[j].result()));
      };
      return InvalidArgument(
          "the result of branch 0 computation and branch %d computation must "
          "have the same shape: got %s.",
          j, shape_string());
    }
  }
  // For each subshape, If any of the branch is dynamic, we say result is
  // dynamic:
  //
  //   true_branch  (s32[<=4])
  //   false_branch (s32[4])
  //
  // Result is s32[<=4].
  ShapeUtil::ForEachMutableSubshape(
      &result, [&](Shape* subshape, const ShapeIndex& index) {
        if (!subshape->IsArray()) {
          return;
        }
        for (int j = 0; j < branch_computations.size(); ++j) {
          auto branch_subshape =
              ShapeUtil::GetSubshape(branch_computations[j].result(), index);
          for (int64_t i = 0; i < branch_subshape.rank(); ++i) {
            if (branch_subshape.is_dynamic_dimension(i)) {
              subshape->set_dynamic_dimension(i, true);
            }
          }
        }
      });

  return result;
}

/* static */ StatusOr<Shape> ShapeInference::InferBroadcastShape(
    const Shape& operand, absl::Span<const int64_t> broadcast_sizes) {
  TF_RETURN_IF_ERROR(ExpectArray(operand, "operand of broadcast"));
  for (int64_t size : broadcast_sizes) {
    if (size < 0) {
      return InvalidArgument("Broadcast with negative dimension size %d.",
                             size);
    }
  }

  std::vector<int64_t> dimensions(operand.dimensions_size() +
                                  broadcast_sizes.size());
  std::copy(broadcast_sizes.begin(), broadcast_sizes.end(), dimensions.begin());
  std::copy(operand.dimensions().begin(), operand.dimensions().end(),
            dimensions.begin() + broadcast_sizes.size());

  Shape result = ShapeUtil::MakeShape(operand.element_type(), dimensions);
  for (int64_t i = 0; i < operand.dimensions_size(); ++i) {
    result.set_dynamic_dimension(broadcast_sizes.size() + i,
                                 operand.is_dynamic_dimension(i));
  }
  return result;
}

/* static */ StatusOr<Shape> ShapeInference::InferBroadcastShape(
    const Shape& operand_shape, const Shape& output_shape,
    absl::Span<const int64_t> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of broadcast"));
  TF_RETURN_IF_ERROR(ExpectArray(output_shape, "operand of broadcast"));
  const int64_t operand_rank = operand_shape.rank();
  const int64_t output_rank = output_shape.rank();
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
  for (int64_t i = 0; i < operand_rank; i++) {
    if (broadcast_dimensions[i] < 0 || broadcast_dimensions[i] >= output_rank) {
      return InvalidArgument("Broadcast dimension %lld is out of bound",
                             broadcast_dimensions[i]);
    }
    if (operand_shape.dimensions(i) !=
            output_shape.dimensions(broadcast_dimensions[i]) &&
        operand_shape.dimensions(i) != 1) {
      return InvalidArgument(
          "Input dimension should be either 1 or equal to the output dimension "
          "it is broadcasting into; the %lldth operand dimension is %lld, the "
          "%lldth output dimension is %lld.",
          i, operand_shape.dimensions(i), broadcast_dimensions[i],
          output_shape.dimensions(broadcast_dimensions[i]));
    }
    if (operand_shape.is_dynamic_dimension(i) !=
        output_shape.is_dynamic_dimension(broadcast_dimensions[i])) {
      return InvalidArgument(
          "Broadcast input and output dynamism mismatch: %s and %s",
          operand_shape.ToString(), output_shape.ToString());
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

/* static */ StatusOr<Shape> ShapeInference::InferDynamicReshapeShape(
    const Shape& operand, absl::Span<const Shape* const> dim_size_shapes,
    absl::Span<const int64_t> new_size_bounds,
    const std::vector<bool>& dims_are_dynamic) {
  if (new_size_bounds.size() != dims_are_dynamic.size()) {
    return InvalidArgument(
        "DynamicReshape has to have the same number of elements in new_sizes "
        "(%d) and dims_are_dynamic (%d)",
        new_size_bounds.size(), dims_are_dynamic.size());
  }

  for (const Shape* dim_size_shape : dim_size_shapes) {
    if (dim_size_shape->element_type() != S32 && dim_size_shape->rank() != 0) {
      return InvalidArgument(
          "DynamicReshape's dim size has to be scalar S32, got (%s): ",
          dim_size_shape->ToString());
    }
  }

  Shape inferred_shape = ShapeUtil::MakeShape(
      operand.element_type(), new_size_bounds, dims_are_dynamic);
  if (ShapeUtil::ElementsIn(operand) != ShapeUtil::ElementsIn(inferred_shape)) {
    return InvalidArgument(
        "Reshape operation has mismatched element counts: from=%d (%s) "
        "to=%d (%s).",
        ShapeUtil::ElementsIn(operand), ShapeUtil::HumanString(operand),
        ShapeUtil::ElementsIn(inferred_shape),
        ShapeUtil::HumanString(inferred_shape));
  }
  return inferred_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferReshapeShape(
    const Shape& operand, absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> new_sizes, int64_t inferred_dimension) {
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

  std::vector<int64_t> indices(operand.rank());
  std::iota(indices.begin(), indices.end(), 0);
  if (dimensions.size() != operand.rank() ||
      !std::is_permutation(dimensions.begin(), dimensions.end(),
                           indices.begin())) {
    return InvalidArgument(
        "Reshape dimensions [%s] are not a permutation of the operand "
        "dimensions (operand shape is %s).",
        StrJoin(dimensions, ","), ShapeUtil::HumanString(operand));
  }

  // Propagate dynamic dimension.
  auto common_factors = CommonFactors(operand.dimensions(), new_sizes);
  for (int64_t input_dim = 0; input_dim < operand.rank(); ++input_dim) {
    if (!operand.is_dynamic_dimension(input_dim)) {
      continue;
    }

    std::string reshape_debug_str = absl::StrFormat(
        "output: %s, input: %s, input_dim: "
        "%lld",
        ShapeUtil::HumanString(inferred_shape), ShapeUtil::HumanString(operand),
        input_dim);

    int64_t input_dim_start = -1;
    int64_t input_dim_end = -1;
    int64_t output_dim_start = -1;
    int64_t output_dim_end = -1;
    // Find common_factors that the input_dim belongs to.
    for (int64_t i = 0; i < common_factors.size() - 1; ++i) {
      auto start = common_factors[i];
      auto end = common_factors[i + 1];
      if (input_dim >= start.first && input_dim < end.first) {
        input_dim_start = start.first;
        input_dim_end = end.first;
        output_dim_start = start.second;
        output_dim_end = end.second;
        break;
      }
    }
    if ((input_dim_end - input_dim_start) > 1 &&
        (output_dim_end - output_dim_start) > 1) {
      // We don't support the case when a dynamic dimension is both combined
      // with and splitted into other dimensions:
      //
      //  [x, yz]
      //     | Reshape
      //  [xy, z]
      return Unimplemented(
          "Dynamic input dimension to reshape that is both splitted and "
          "combined is not supported: %s",
          reshape_debug_str);
    }

    for (auto common_factor : common_factors) {
      //
      // For reshapes like:
      //  [<=5]
      //    | Reshape
      //  [1, 5]
      //
      //  The input dynamic dimension can go into either dynamic dimensions.
      //  However, the return value of common factors only returns
      //  input: 5
      //  output: 5
      //
      //  We need to expand common factor to include degenerated output
      //  dimensions:
      //  input: 5
      //  output: 1, 5
      //
      //  such that in the logic later on we can consider both dimensions as
      //  candidate.
      if (common_factor.first == input_dim_start) {
        output_dim_start = std::min(output_dim_start, common_factor.second);
      }
      if (common_factor.first == input_dim_end) {
        output_dim_end = std::max(output_dim_end, common_factor.second);
      }
    }

    // Calculate output dynamic reshape dimension.
    int64_t output_dynamic_dimension = -1;

    if (operand.dimensions(input_dim) == 1 && !new_sizes.empty()) {
      // If dynamic dimension is size 1, it can only be most-major or
      // most-minor.
      if (input_dim == 0) {
        output_dynamic_dimension = 0;
      }
      if (input_dim == operand.rank() - 1) {
        output_dynamic_dimension = new_sizes.size() - 1;
      }

      if (output_dynamic_dimension == -1) {
        return Unimplemented(
            "Dynamic degenerated dimension that's not most-minor nor "
            "most-major is not supported: %s",
            reshape_debug_str);
      }
    }

    if (output_dynamic_dimension == -1 &&
        output_dim_end - output_dim_start == 1) {
      // Only one possible output dimension.
      output_dynamic_dimension = output_dim_start;
    }
    if (output_dynamic_dimension == -1 &&
        output_dim_end - output_dim_start > 1) {
      // Multiple outputs can be dynamic, use "inferred_dimension" to tie-break.
      output_dynamic_dimension = inferred_dimension;
    }

    if (output_dynamic_dimension != -1) {
      // TODO(yunxing): Turn this into a CHECK.
      inferred_shape.set_dynamic_dimension(output_dynamic_dimension, true);
    } else {
      std::vector<int64_t> output_non_degenerated;
      output_non_degenerated.reserve(output_dim_end);
      for (int64_t i = output_dim_start; i < output_dim_end; ++i) {
        if (new_sizes[i] != 1) {
          output_non_degenerated.push_back(i);
        }
      }
      if (output_non_degenerated.size() == 1) {
        inferred_shape.set_dynamic_dimension(output_non_degenerated[0], true);
      }
    }
  }

  return inferred_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferTransposeShape(
    const Shape& operand, absl::Span<const int64_t> dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(operand, "transpose"));

  if (dimensions.size() != operand.rank() || !IsPermutation(dimensions)) {
    return InvalidArgument(
        "Transpose dimensions [%s] are not a permutation of the operand "
        "dimensions (operand shape is %s).",
        StrJoin(dimensions, ","), ShapeUtil::HumanString(operand));
  }

  // Permute(dimensions,input) computes output[dimensions[i]]=input[i]. However,
  // we need output[i]=input[dimensions[i]] which is
  // Permute(Inverse(dimensions),input).
  return ShapeUtil::PermuteDimensions(dimensions, operand);
}

/* static */ StatusOr<Shape> ShapeInference::InferClampShape(
    const Shape& min, const Shape& operand, const Shape& max) {
  TF_RETURN_IF_ERROR(ExpectArray(min, "clamp min"));
  TF_RETURN_IF_ERROR(ExpectArray(operand, "clamp operand"));
  TF_RETURN_IF_ERROR(ExpectArray(max, "clamp max"));

  if (!ShapeUtil::CompatibleIgnoringFpPrecision(min, operand) ||
      !ShapeUtil::CompatibleIgnoringFpPrecision(max, operand)) {
    return InvalidArgument(
        "Clamp with different shapes: %s, %s, %s.", ShapeUtil::HumanString(min),
        ShapeUtil::HumanString(operand), ShapeUtil::HumanString(max));
  }
  return operand;
}

/* static */ StatusOr<Shape> ShapeInference::InferSelectShape(
    const Shape& pred, const Shape& on_true, const Shape& on_false) {
  TF_RETURN_IF_ERROR(ExpectArray(pred, "select pred"));
  TF_RETURN_IF_ERROR(ExpectArray(on_true, "select on-true"));
  TF_RETURN_IF_ERROR(ExpectArray(on_false, "select on-false"));

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
  if (!Shape::Equal()
           .IgnoreElementType()
           .IgnoreLayout()
           .IgnoreDynamicDimension()(pred, on_true)) {
    return InvalidArgument(
        "Operands to select and predicate must be the same shape; got %s and "
        "%s.",
        ShapeUtil::HumanString(on_true), ShapeUtil::HumanString(pred));
  }

  return ShapeUtil::ChangeElementType(
      pred, ShapeUtil::HigherPrecisionElementType(on_true, on_false));
}

/* static */ StatusOr<Shape> ShapeInference::InferCallShape(
    absl::Span<const Shape* const> arg_shapes, const ProgramShape& to_apply) {
  // The applied function's arity equals the number of arguments.
  if (arg_shapes.size() != to_apply.parameters_size()) {
    std::string computation_signature = ShapeUtil::HumanString(to_apply);
    std::string argument_shapes =
        StrJoin(arg_shapes, ", ", [](std::string* out, const Shape* shape) {
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
    const Shape& input_shape, absl::Span<const int64_t> start_indices_shape,
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

  const int64_t output_offset_dim_count = dim_numbers.offset_dims_size();
  const int64_t output_shape_rank =
      output_offset_dim_count + start_indices_shape.size() - 1;

  for (int i = 0; i < dim_numbers.offset_dims_size(); ++i) {
    int64_t offset_dim = dim_numbers.offset_dims(i);
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
    int64_t operand_dim_for_start_index_i = dim_numbers.start_index_map(i);
    if (operand_dim_for_start_index_i < 0 ||
        operand_dim_for_start_index_i >= input_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid start_index_map; domain is [0, %d), got: %d->%d.",
          input_shape.dimensions_size(), i, operand_dim_for_start_index_i);
    }
  }

  std::vector<int64_t> sorted_start_index_map(
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

  for (int64_t collapsed_dim : dim_numbers.collapsed_slice_dims()) {
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

  return OkStatus();
}

/*static*/ StatusOr<Shape> ShapeInference::InferGatherShape(
    const Shape& input_shape, const Shape& start_indices_shape,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64_t> slice_sizes) {
  TF_RETURN_IF_ERROR(
      ExpectArray(input_shape, "input tensor operand of gather op"));
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

  std::vector<int64_t> expanded_start_indices_shape;
  // Also tracks if an output dimension is dynamic.
  std::vector<bool> expanded_start_indices_shape_dynamic_dimensions;
  expanded_start_indices_shape.reserve(start_indices_shape.dimensions_size());
  expanded_start_indices_shape_dynamic_dimensions.reserve(
      start_indices_shape.dimensions_size());
  absl::c_copy(start_indices_shape.dimensions(),
               std::back_inserter(expanded_start_indices_shape));
  absl::c_copy(
      start_indices_shape.dynamic_dimensions(),
      std::back_inserter(expanded_start_indices_shape_dynamic_dimensions));
  if (expanded_start_indices_shape.size() ==
      gather_dim_numbers.index_vector_dim()) {
    expanded_start_indices_shape.push_back(1);
    expanded_start_indices_shape_dynamic_dimensions.push_back(false);
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
    int64_t slice_size = slice_sizes[i];
    int64_t corresponding_input_size = input_shape.dimensions(i);
    if (slice_size < 0 || slice_size > corresponding_input_size) {
      return InvalidArgument(
          "Slice size at index %d in gather op is out of range, must be "
          "within [0, %d), got %d.",
          i, corresponding_input_size + 1, slice_size);
    }
  }

  for (int i = 0; i < gather_dim_numbers.collapsed_slice_dims_size(); i++) {
    if (slice_sizes[gather_dim_numbers.collapsed_slice_dims(i)] > 1) {
      return InvalidArgument(
          "Gather op can only collapse slice dims with bound 1 or 0, but bound "
          "is %d for index %d at position %d.",
          slice_sizes[gather_dim_numbers.collapsed_slice_dims(i)],
          gather_dim_numbers.collapsed_slice_dims(i), i);
    }
  }

  int64_t result_rank = gather_dim_numbers.offset_dims_size() +
                        (expanded_start_indices_shape.size() - 1);
  int64_t offset_dims_seen = 0;
  int64_t gather_dims_seen = 0;
  std::vector<int64_t> output_dim_bounds;
  output_dim_bounds.reserve(result_rank);

  std::vector<bool> output_dim_is_dynamic;
  output_dim_is_dynamic.reserve(result_rank);
  for (int64_t i = 0; i < result_rank; i++) {
    int64_t current_bound;
    bool dim_dynamic = false;
    bool is_window_index =
        absl::c_binary_search(gather_dim_numbers.offset_dims(), i);
    if (is_window_index) {
      while (absl::c_binary_search(gather_dim_numbers.collapsed_slice_dims(),
                                   offset_dims_seen)) {
        offset_dims_seen++;
      }
      // Gathering an entire dynamic dimension creates dynamic dimension.
      //
      // e.g.,:
      //
      // gather(input: [1,<=2,1], slice_sizes={1,2,1})
      //
      // creates
      //
      // [<=2, 1]
      if (slice_sizes[offset_dims_seen] ==
          input_shape.dimensions(offset_dims_seen)) {
        dim_dynamic = input_shape.is_dynamic_dimension(offset_dims_seen);
      }
      current_bound = slice_sizes[offset_dims_seen++];
    } else {
      if (gather_dims_seen == gather_dim_numbers.index_vector_dim()) {
        gather_dims_seen++;
      }
      // Forward dynamic dimensions from indices.
      dim_dynamic =
          expanded_start_indices_shape_dynamic_dimensions[gather_dims_seen];

      current_bound = expanded_start_indices_shape[gather_dims_seen++];
    }
    output_dim_is_dynamic.push_back(dim_dynamic);
    output_dim_bounds.push_back(current_bound);
  }

  return ShapeUtil::MakeShape(input_shape.element_type(), output_dim_bounds,
                              output_dim_is_dynamic);
}

namespace {

Status ValidateScatterDimensionNumbers(
    const Shape& operand_shape, absl::Span<const int64_t> scatter_indices_shape,
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
  const int64_t updates_rank = updates_shape.rank();
  for (int64_t window_dim : dim_numbers.update_window_dims()) {
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
  for (int64_t inserted_dim : dim_numbers.inserted_window_dims()) {
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
  if (window_size != operand_shape.rank()) {
    return InvalidArgument(
        "Scatter op has window of size %d; doesn't match operand of rank %d.",
        window_size, operand_shape.rank());
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
    int64_t scatter_dim_to_operand_dim =
        dim_numbers.scatter_dims_to_operand_dims(i);
    if (scatter_dim_to_operand_dim < 0 ||
        scatter_dim_to_operand_dim >= operand_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid scatter_dims_to_operand_dims mapping; domain is [0, %d), "
          "got: %d->%d.",
          operand_shape.dimensions_size(), i, scatter_dim_to_operand_dim);
    }
  }
  std::vector<int64_t> sorted_scatter_dims_to_operand_dims(
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

  return OkStatus();
}

}  // namespace

/*static*/ StatusOr<Shape> ShapeInference::InferScatterShape(
    absl::Span<const Shape* const> arg_shapes,
    const ProgramShape& to_apply_shape,
    const ScatterDimensionNumbers& scatter_dim_numbers) {
  const int64_t operand_count = arg_shapes.size() / 2;
  if (operand_count * 2 + 1 != arg_shapes.size()) {
    return InvalidArgument(
        "Invalid argument count of scatter op: Expected %d, saw %d",
        operand_count * 2 + 1, arg_shapes.size());
  }

  const Shape& scatter_indices_shape = *arg_shapes[operand_count];
  TF_RETURN_IF_ERROR(
      ExpectArray(scatter_indices_shape, "scatter indices of scatter op"));
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

  std::vector<int64_t> expanded_scatter_indices_shape =
      SpanToVector(scatter_indices_shape.dimensions());
  if (expanded_scatter_indices_shape.size() ==
      scatter_dim_numbers.index_vector_dim()) {
    expanded_scatter_indices_shape.push_back(1);
  }

  auto operand_shapes = arg_shapes.first(operand_count);
  auto updates_shapes = arg_shapes.last(operand_count);
  for (int64_t operand_i = 0; operand_i < operand_count; ++operand_i) {
    const Shape& operand_shape = *operand_shapes[operand_i];
    const Shape& updates_shape = *updates_shapes[operand_i];
    TF_RETURN_IF_ERROR(ExpectArray(
        operand_shape, absl::StrCat("operand ", operand_i, " of scatter op")));
    TF_RETURN_IF_ERROR(ExpectArray(
        updates_shape, absl::StrCat("updates ", operand_i, " of scatter op")));

    int64_t inserted_dims_seen = 0;
    std::vector<int64_t> max_update_slice_sizes;
    const auto dimensions_size = operand_shape.dimensions_size();
    max_update_slice_sizes.reserve(dimensions_size);
    for (int i = 0; i < dimensions_size; ++i) {
      if (inserted_dims_seen <
              scatter_dim_numbers.inserted_window_dims_size() &&
          scatter_dim_numbers.inserted_window_dims(inserted_dims_seen) == i) {
        ++inserted_dims_seen;
      } else {
        max_update_slice_sizes.push_back(operand_shape.dimensions(i));
      }
    }
    int64_t expected_updates_rank =
        expanded_scatter_indices_shape.size() - 1 +
        scatter_dim_numbers.update_window_dims_size();
    if (updates_shape.rank() != expected_updates_rank) {
      return InvalidArgument("Updates tensor must be of rank %d; got %d.",
                             expected_updates_rank, updates_shape.rank());
    }

    TF_RETURN_IF_ERROR(ValidateScatterDimensionNumbers(
        operand_shape, expanded_scatter_indices_shape, updates_shape,
        scatter_dim_numbers));

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

    int64_t scatter_dims_seen = 0;
    for (int64_t i = 0; i < updates_shape.rank(); ++i) {
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
  }

  // Check if the update computation has a proper shape as a reduction.
  absl::InlinedVector<Shape, 1> init_element_shapes;
  absl::InlinedVector<const Shape*, 1> init_element_shape_ptrs;
  absl::InlinedVector<PrimitiveType, 1> updates_element_types;
  init_element_shapes.reserve(operand_count);
  init_element_shape_ptrs.reserve(operand_count);
  updates_element_types.reserve(operand_count);
  for (int64_t i = 0; i < operand_count; ++i) {
    init_element_shapes.push_back(
        ShapeUtil::MakeShape(operand_shapes[i]->element_type(), {}));
    init_element_shape_ptrs.push_back(&init_element_shapes.back());
    updates_element_types.push_back(updates_shapes[i]->element_type());
  }
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply_shape, init_element_shape_ptrs,
                                        updates_element_types, operand_count));

  return operand_count == 1 ? *operand_shapes[0]
                            : ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
}

}  // namespace xla
