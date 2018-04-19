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

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::str_util::Join;
using tensorflow::strings::Printf;

namespace xla {

namespace {

// Return the UnaryOperation proto enum value associated with the given HLO
// opcode.
UnaryOperation OpcodeToUnaryOperation(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAbs:
      return UNOP_ABS;
    case HloOpcode::kCeil:
      return UNOP_CEIL;
    case HloOpcode::kCos:
      return UNOP_COS;
    case HloOpcode::kExp:
      return UNOP_EXP;
    case HloOpcode::kFloor:
      return UNOP_FLOOR;
    case HloOpcode::kImag:
      return UNOP_IMAG;
    case HloOpcode::kIsFinite:
      return UNOP_IS_FINITE;
    case HloOpcode::kLog:
      return UNOP_LOG;
    case HloOpcode::kNot:
      return UNOP_NOT;
    case HloOpcode::kNegate:
      return UNOP_NEGATE;
    case HloOpcode::kReal:
      return UNOP_REAL;
    case HloOpcode::kRoundNearestAfz:
      return UNOP_ROUND_NEAREST_AFZ;
    case HloOpcode::kSign:
      return UNOP_SIGN;
    case HloOpcode::kSin:
      return UNOP_SIN;
    case HloOpcode::kSort:
      return UNOP_SORT;
    case HloOpcode::kTanh:
      return UNOP_TANH;
    default:
      LOG(FATAL) << "Unhandled opcode for conversion to unary operation: "
                 << opcode;
  }
}

// Return the BinaryOperation proto enum value associated with the given HLO
// opcode.
BinaryOperation OpcodeToBinaryOperation(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAtan2:
      return BINOP_ATAN2;
    case HloOpcode::kComplex:
      return BINOP_COMPLEX;
    case HloOpcode::kMultiply:
      return BINOP_MUL;
    case HloOpcode::kAdd:
      return BINOP_ADD;
    case HloOpcode::kSubtract:
      return BINOP_SUB;
    case HloOpcode::kDivide:
      return BINOP_DIV;
    case HloOpcode::kEq:
      return BINOP_EQ;
    case HloOpcode::kGe:
      return BINOP_GE;
    case HloOpcode::kGt:
      return BINOP_GT;
    case HloOpcode::kLe:
      return BINOP_LE;
    case HloOpcode::kLt:
      return BINOP_LT;
    case HloOpcode::kNe:
      return BINOP_NE;
    case HloOpcode::kMaximum:
      return BINOP_MAX;
    case HloOpcode::kMinimum:
      return BINOP_MIN;
    case HloOpcode::kPower:
      return BINOP_POW;
    case HloOpcode::kRemainder:
      return BINOP_REM;
    case HloOpcode::kOr:
      return BINOP_OR;
    case HloOpcode::kAnd:
      return BINOP_AND;
    case HloOpcode::kShiftLeft:
      return BINOP_SHIFT_LEFT;
    case HloOpcode::kShiftRightArithmetic:
      return BINOP_SHIFT_RIGHT_ARITHMETIC;
    case HloOpcode::kShiftRightLogical:
      return BINOP_SHIFT_RIGHT_LOGICAL;
    default:
      LOG(FATAL) << "unhandled opcode " << opcode;
  }
}

// Return the TernaryOperation proto enum value associated with the given HLO
// opcode.
TernaryOperation OpcodeToTernaryOperation(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kClamp:
      return TRIOP_CLAMP;
    case HloOpcode::kSelect:
      return TRIOP_SELECT;
    default:
      LOG(FATAL) << "unhandled opcode " << opcode;
  }
}

// Return the VariadicOperation proto enum value associated with the given HLO
// opcode.
VariadicOperation OpcodeToVariadicOperation(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kTuple:
      return VAROP_TUPLE;
    default:
      LOG(FATAL) << "unhandled opcode " << opcode;
  }
}

// Returns true if no element is present in slice more than once.
bool AllUnique(tensorflow::gtl::ArraySlice<int64> slice) {
  return std::set<int64>(slice.begin(), slice.end()).size() == slice.size();
}

tensorflow::Status ExpectNotTupleOrOpaque(const Shape& shape,
                                          tensorflow::StringPiece op_type) {
  if (ShapeUtil::IsTuple(shape)) {
    return InvalidArgument("Expected non-tuple argument for %s, but got %s.",
                           op_type.ToString().c_str(),
                           ShapeUtil::HumanString(shape).c_str());
  } else if (ShapeUtil::IsOpaque(shape)) {
    return InvalidArgument("Expected non-opaque argument for %s, but got %s.",
                           op_type.ToString().c_str(),
                           ShapeUtil::HumanString(shape).c_str());
  } else {
    return tensorflow::Status::OK();
  }
}

tensorflow::Status VerifyReducerShape(const ProgramShape& reducer_shape,
                                      const Shape& init_value_shape,
                                      const PrimitiveType& input_element_type) {
  if (reducer_shape.parameters_size() != 2) {
    return InvalidArgument(
        "Reduction function must take 2 parameters, but "
        "takes %d parameter(s).",
        reducer_shape.parameters_size());
  }

  const Shape& accumulator_shape = reducer_shape.result();
  if (ShapeUtil::Rank(accumulator_shape) != 0) {
    return InvalidArgument(
        "Reduction function must have rank 0 (rank %lld reduction function "
        "given).",
        ShapeUtil::Rank(accumulator_shape));
  }

  // Check that the accumulator can be passed in as the first argument.
  // Note: comparing here and below with Compatible since we don't care about
  // layout in scalars - see b/26668201 for a longer-term vision.
  if (!ShapeUtil::Compatible(accumulator_shape, reducer_shape.parameters(0))) {
    return InvalidArgument(
        "Reduction function's first parameter shape differs from the "
        "result shape: %s vs %s",
        ShapeUtil::HumanString(reducer_shape.parameters(0)).c_str(),
        ShapeUtil::HumanString(accumulator_shape).c_str());
  }

  // Check that init_value's shape is suitable for reducer_shape.
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(accumulator_shape,
                                                init_value_shape)) {
    return InvalidArgument(
        "Reduction function's accumulator shape differs from the "
        "init_value shape: %s vs %s",
        ShapeUtil::HumanString(accumulator_shape).c_str(),
        ShapeUtil::HumanString(init_value_shape).c_str());
  }

  // Check that the inputs can be passed in as the second argument.
  const Shape& input_element_shape =
      ShapeUtil::MakeShape(input_element_type, {});
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(input_element_shape,
                                                reducer_shape.parameters(1))) {
    return InvalidArgument(
        "Reduction function's second parameter shape differs from the "
        "input type element type: %s vs %s",
        ShapeUtil::HumanString(reducer_shape.parameters(1)).c_str(),
        ShapeUtil::HumanString(input_element_shape).c_str());
  }

  // Currently the accumulator and inputs must be the same type,
  // though that restriction could be relaxed.
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(accumulator_shape,
                                                reducer_shape.parameters(1))) {
    return InvalidArgument(
        "Reduction function's second parameter shape must "
        "match the result shape, but got %s vs %s.",
        ShapeUtil::HumanString(reducer_shape.parameters(1)).c_str(),
        ShapeUtil::HumanString(accumulator_shape).c_str());
  }

  return tensorflow::Status::OK();
}

StatusOr<Shape> InferWindowOutputShape(const Shape& base_shape,
                                       const Window& window,
                                       PrimitiveType element_type,
                                       bool allow_negative_padding) {
  if (window.dimensions_size() != ShapeUtil::Rank(base_shape)) {
    return InvalidArgument(
        "Window has dimension %d but base shape has dimension %lld.",
        window.dimensions_size(), ShapeUtil::Rank(base_shape));
  }

  std::vector<int64> output_dimensions(window.dimensions_size());
  for (int64 i = 0; i < window.dimensions_size(); ++i) {
    const auto& dim = window.dimensions(i);
    if (dim.size() <= 0) {
      return InvalidArgument("Window %s has a non-positive dimension.",
                             window.DebugString().c_str());
    }
    if (dim.stride() <= 0) {
      return InvalidArgument("Window %s has a non-positive stride.",
                             window.DebugString().c_str());
    }
    if (!allow_negative_padding && dim.padding_low() < 0) {
      return InvalidArgument("Window %s has a negative low padding.",
                             window.DebugString().c_str());
    }
    if (!allow_negative_padding && dim.padding_high() < 0) {
      return InvalidArgument("Window %s has a negative high padding.",
                             window.DebugString().c_str());
    }
    if (dim.base_dilation() < 1) {
      return InvalidArgument(
          "Window %s has a non-positive base area dilation factor.",
          window.DebugString().c_str());
    }
    if (dim.window_dilation() < 1) {
      return InvalidArgument(
          "Window %s has a non-positive window dilation factor.",
          window.DebugString().c_str());
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

  return ShapeUtil::MakeShape(element_type, output_dimensions);
}

}  // namespace

/* static */ StatusOr<Shape> ShapeInference::InferUnaryOpShape(
    HloOpcode opcode, const HloInstruction* operand) {
  return InferUnaryOpShape(opcode, operand->shape());
}

/* static */ StatusOr<Shape> ShapeInference::InferUnaryOpShape(
    HloOpcode opcode, const Shape& shape) {
  // There is no copy operation at the proto level, so handle copy explicitly.
  if (opcode == HloOpcode::kCopy) {
    return shape;
  }

  return InferUnaryOpShape(OpcodeToUnaryOperation(opcode), shape);
}

/* static */ StatusOr<Shape> ShapeInference::InferUnaryOpShape(
    UnaryOperation operation, const Shape& arg) {
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(arg, "operand of unary operation"));

  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(arg));
  switch (operation) {
    case UNOP_FLOOR:
    case UNOP_CEIL:
      if (!ShapeUtil::ElementIsFloating(arg)) {
        return InvalidArgument(
            "Expected element type in shape to be floating for floor/ceil "
            "operation; got %s.",
            PrimitiveType_Name(arg.element_type()).c_str());
      }
      return arg;
    case UNOP_COS:
    case UNOP_SIN:
    case UNOP_EXP:
    case UNOP_LOG:
    case UNOP_TANH:
      if (!ShapeUtil::ElementIsFloating(arg) &&
          !ShapeUtil::ElementIsComplex(arg)) {
        return InvalidArgument(
            "Expected element type in shape to be floating or complex for "
            "sin/cos/exp/log/tanh operation; got %s.",
            PrimitiveType_Name(arg.element_type()).c_str());
      }
      return arg;
    case UNOP_REAL:
    case UNOP_IMAG:
      if (!ShapeUtil::ElementIsComplex(arg)) {
        return InvalidArgument(
            "Expected element type in shape to be complex for real/imag "
            "operation; got %s.",
            PrimitiveType_Name(arg.element_type()).c_str());
      }
      return ShapeUtil::ChangeElementType(arg, F32);
    case UNOP_ABS:
      if (ShapeUtil::ElementIsComplex(arg)) {
        return ShapeUtil::ChangeElementType(
            arg, primitive_util::ComplexComponentType(arg.element_type()));
      }
      return arg;
    case UNOP_NEGATE:
    case UNOP_ROUND_NEAREST_AFZ:
    case UNOP_SIGN:
    case UNOP_SORT:
      return arg;

    case UNOP_NOT:
      if (arg.element_type() != PRED &&
          !primitive_util::IsIntegralType(arg.element_type())) {
        return InvalidArgument(
            "Expected pred or an integral element type in argument to Not "
            "operation; got %s.",
            PrimitiveType_Name(arg.element_type()).c_str());
      }
      return arg;

    case UNOP_IS_FINITE:
      if (!ShapeUtil::ElementIsFloating(arg)) {
        return InvalidArgument(
            "Expected element type in shape to be floating point for IsFinite "
            "operation; got %s.",
            PrimitiveType_Name(arg.element_type()).c_str());
      }
      return ShapeUtil::ChangeElementType(arg, PRED);

    default:
      return InvalidArgument(
          "Unknown operation for unary shape inference: \"%s\".",
          UnaryOperation_Name(operation).c_str());
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferConcatOpShape(
    tensorflow::gtl::ArraySlice<const Shape*> arg_shapes,
    const int64 dimension) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Concatenate expects at least one argument.");
  }
  if (dimension < 0 || dimension >= ShapeUtil::Rank(*arg_shapes[0])) {
    return InvalidArgument("Concatenate dimension out of bounds: %lld.",
                           dimension);
  }
  const Shape* arg_shape = nullptr;
  PrimitiveType element_type = PRIMITIVE_TYPE_INVALID;
  for (const Shape* shape : arg_shapes) {
    TF_RETURN_IF_ERROR(
        ExpectNotTupleOrOpaque(*shape, "operand of concatenation"));
    if (!arg_shape) {
      arg_shape = shape;
      element_type = arg_shape->element_type();
      continue;
    }
    if (ShapeUtil::Rank(*arg_shape) != ShapeUtil::Rank(*shape)) {
      return InvalidArgument(
          "Cannot concatenate arrays with different ranks: %lld (%s) vs %lld "
          "(%s).",
          ShapeUtil::Rank(*arg_shape),
          ShapeUtil::HumanString(*arg_shape).c_str(), ShapeUtil::Rank(*shape),
          ShapeUtil::HumanString(*shape).c_str());
    }
    if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(*arg_shape, *shape)) {
      return InvalidArgument(
          "Cannot concatenate arrays with different element types: %s vs %s.",
          PrimitiveType_Name(arg_shape->element_type()).c_str(),
          PrimitiveType_Name(shape->element_type()).c_str());
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
            "the same): %s vs %s in dimension %lld.",
            ShapeUtil::HumanString(*arg_shape).c_str(),
            ShapeUtil::HumanString(*shape).c_str(), dimension);
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
        ShapeUtil::HumanString(operand_shape).c_str(),
        PrimitiveType_Name(new_element_type).c_str());
  }
  if (ShapeUtil::IsTuple(operand_shape) || new_element_type == TUPLE) {
    // Note: we may want to support tuple conversions via this operation in the
    // future, by recursing into the tuple elements to check all sub-conversions
    // are valid. For now we just reject them, though.
    return InvalidArgument(
        "Convert does not allow tuples, so cannot convert from %s to %s.",
        ShapeUtil::HumanString(operand_shape).c_str(),
        PrimitiveType_Name(new_element_type).c_str());
  }

  return ShapeUtil::ChangeElementType(operand_shape, new_element_type);
}

/* static */ StatusOr<Shape> ShapeInference::InferBitcastConvertShape(
    const Shape& operand_shape, PrimitiveType new_element_type) {
  auto old_element_type = operand_shape.element_type();
  if (primitive_util::IsComplexType(old_element_type) !=
      primitive_util::IsComplexType(new_element_type)) {
    return InvalidArgument("Conversion from complex to real type %s => %s.",
                           ShapeUtil::HumanString(operand_shape).c_str(),
                           PrimitiveType_Name(new_element_type).c_str());
  }
  if (ShapeUtil::IsTuple(operand_shape) || new_element_type == TUPLE) {
    // Note: we may want to support tuple conversions via this operation in the
    // future, by recursing into the tuple elements to check all sub-conversions
    // are valid. For now we just reject them, though.
    return InvalidArgument(
        "Cannot convert from or to tuple type; requested conversion: %s => %s.",
        ShapeUtil::HumanString(operand_shape).c_str(),
        PrimitiveType_Name(new_element_type).c_str());
  }
  if (primitive_util::BitWidth(old_element_type) !=
      primitive_util::BitWidth(new_element_type)) {
    return InvalidArgument(
        "Cannot bitcast types with different bit-widths: %s => %s.",
        PrimitiveType_Name(old_element_type).c_str(),
        PrimitiveType_Name(new_element_type).c_str());
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
        PrimitiveType_Name(operand_shape.element_type()).c_str());
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
  if (ShapeUtil::IsTuple(operand_shape)) {
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
        ShapeUtil::HumanString(operand_shape).c_str(),
        padding_config.ShortDebugString().c_str());
  }
  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(operand_shape,
                                                     padding_value_shape)) {
    return InvalidArgument(
        "The element types of the operands to Pad do not match.");
  }
  std::vector<int64> dimensions(ShapeUtil::Rank(operand_shape));
  for (int64 i = 0; i < operand_shape.dimensions_size(); ++i) {
    dimensions[i] = operand_shape.dimensions(i) +
                    padding_config.dimensions(i).edge_padding_low() +
                    padding_config.dimensions(i).edge_padding_high() +
                    std::max<int64>(operand_shape.dimensions(i) - 1, 0LL) *
                        padding_config.dimensions(i).interior_padding();
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
  auto dims_in_range =
      [](const int64 rank, tensorflow::gtl::ArraySlice<int64> contracting_dims,
         tensorflow::gtl::ArraySlice<int64> batch_dims) -> bool {
    auto in_range = [&rank](int64 i) -> bool { return 0 <= i && i < rank; };
    return std::all_of(contracting_dims.begin(), contracting_dims.end(),
                       in_range) &&
           std::all_of(batch_dims.begin(), batch_dims.end(), in_range);
  };

  tensorflow::gtl::ArraySlice<int64> lhs_contracting_dimensions =
      AsInt64Slice(dimension_numbers.lhs_contracting_dimensions());
  tensorflow::gtl::ArraySlice<int64> rhs_contracting_dimensions =
      AsInt64Slice(dimension_numbers.rhs_contracting_dimensions());
  tensorflow::gtl::ArraySlice<int64> lhs_batch_dimensions =
      AsInt64Slice(dimension_numbers.lhs_batch_dimensions());
  tensorflow::gtl::ArraySlice<int64> rhs_batch_dimensions =
      AsInt64Slice(dimension_numbers.rhs_batch_dimensions());

  if (!dims_in_range(ShapeUtil::Rank(lhs), lhs_contracting_dimensions,
                     lhs_batch_dimensions) ||
      !dims_in_range(ShapeUtil::Rank(rhs), rhs_contracting_dimensions,
                     rhs_batch_dimensions)) {
    return InvalidArgument("A dimension number is out of range in Dot: %s.",
                           dimension_numbers.DebugString().c_str());
  }

  // Check that dimension numbers are unique.
  auto dims_unique = [](tensorflow::gtl::ArraySlice<int64> contracting_dims,
                        tensorflow::gtl::ArraySlice<int64> batch_dims) -> bool {
    tensorflow::gtl::FlatSet<int64> dim_set;
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
                           dimension_numbers.DebugString().c_str());
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
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(lhs, "lhs of dot"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(rhs, "rhs of dot"));

  auto fail = [lhs, rhs](const string& addendum) -> Status {
    string message = tensorflow::strings::Printf(
        "Cannot infer shape for dot operation: %s <dot> %s.",
        ShapeUtil::HumanString(lhs).c_str(),
        ShapeUtil::HumanString(rhs).c_str());
    if (!addendum.empty()) {
      message += " " + addendum;
    }
    return InvalidArgument("%s", message.c_str());
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
ShapeInference::InferDegenerateDimensionBroadcastShape(
    BinaryOperation operation, const Shape& lhs, const Shape& rhs) {
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
          BinaryOperation_Name(operation).c_str(),
          ShapeUtil::HumanString(lhs).c_str(),
          ShapeUtil::HumanString(rhs).c_str());
    }
  }
  return ShapeUtil::MakeShape(ShapeUtil::HigherPrecisionElementType(lhs, rhs),
                              output_dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferInDimBroadcastShape(
    BinaryOperation operation, const Shape& smaller_shape,
    const Shape& larger_shape,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  if (broadcast_dimensions.empty() && !ShapeUtil::IsScalar(smaller_shape)) {
    // Reject "magic" inference for binops on different shapes, requiring
    // the user to provide an explicit broadcast dimension in this case.
    // See b/25177275 for more details.
    return InvalidArgument("Automatic shape inference not supported: %s and %s",
                           ShapeUtil::HumanString(smaller_shape).c_str(),
                           ShapeUtil::HumanString(larger_shape).c_str());
  } else if (broadcast_dimensions.size() != ShapeUtil::Rank(smaller_shape)) {
    return InvalidArgument(
        "Size of broadcast_dimensions has to match lower-rank operand's "
        "rank; "
        " lower-rank operand's rank is %lld, size of broadcast_dimensions is "
        "%zu.",
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
          "Broadcast dimension number (%lld) cannot be negative.",
          dimension_to_match);
    }
    if (dimension_to_match >= larger_shape.dimensions_size()) {
      return InvalidArgument(
          "Broadcast dimension number (%lld) too large; higher-rank "
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
          "Broadcast dimension %d mismatch: %lld != %lld; %s and %s.", i,
          small_dimension_size, large_dimension_size,
          ShapeUtil::HumanString(smaller_shape).c_str(),
          ShapeUtil::HumanString(larger_shape).c_str());
    }
    // Make sure the broadcast dimensions are listed in a strictly increasing
    // order.
    if (i > 0 && broadcast_dimensions.at(i - 1) >= dimension_to_match) {
      return InvalidArgument(
          "Broadcast dimensions order is wrong: %lld comes after %lld.",
          dimension_to_match, broadcast_dimensions.at(i - 1));
    }

    output_shape.set_dimensions(dimension_to_match, small_dimension_size);
  }

  return output_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferElementwiseBinaryOpShape(
    BinaryOperation operation, const Shape& lhs, const Shape& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(lhs, "lhs of elementwise binary operation"));
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(rhs, "rhs of elementwise binary operation"));

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) {
    return InvalidArgument(
        "Binary op %s with different element types: %s and %s.",
        BinaryOperation_Name(operation).c_str(),
        ShapeUtil::HumanString(lhs).c_str(),
        ShapeUtil::HumanString(rhs).c_str());
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
    TF_ASSIGN_OR_RETURN(
        Shape indim_broadcast_shape,
        InferInDimBroadcastShape(operation, smaller_shape, larger_shape,
                                 broadcast_dimensions));

    return InferDegenerateDimensionBroadcastShape(
        operation, indim_broadcast_shape, larger_shape);
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferBinaryOpShape(
    HloOpcode opcode, const HloInstruction* lhs, const HloInstruction* rhs) {
  return InferBinaryOpShape(OpcodeToBinaryOperation(opcode), lhs->shape(),
                            rhs->shape(), /*broadcast_dimensions=*/{});
}

/* static */ StatusOr<Shape> ShapeInference::InferBinaryOpShape(
    HloOpcode opcode, const Shape& lhs, const Shape& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return InferBinaryOpShape(OpcodeToBinaryOperation(opcode), lhs, rhs,
                            broadcast_dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferBinaryOpShape(
    BinaryOperation operation, const Shape& lhs, const Shape& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  VLOG(2) << tensorflow::strings::Printf(
      "inferring shape for <%s>(%s, %s) with broadcast_dimensions={%s}",
      BinaryOperation_Name(operation).c_str(),
      ShapeUtil::HumanString(lhs).c_str(), ShapeUtil::HumanString(rhs).c_str(),
      Join(broadcast_dimensions, ", ").c_str());
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));

  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      lhs, tensorflow::strings::StrCat("lhs of binary operation ",
                                       BinaryOperation_Name(operation))));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      rhs, tensorflow::strings::StrCat("rhs of binary operation ",
                                       BinaryOperation_Name(operation))));
  switch (operation) {
    case BINOP_MAX:
    case BINOP_MIN:
    case BINOP_SUB:
    case BINOP_ADD:
    case BINOP_ATAN2:
    case BINOP_POW:
    case BINOP_DIV:
    case BINOP_REM:
    case BINOP_MUL:
    case BINOP_SHIFT_LEFT:
    case BINOP_SHIFT_RIGHT_ARITHMETIC:
    case BINOP_SHIFT_RIGHT_LOGICAL:
      return InferElementwiseBinaryOpShape(operation, lhs, rhs,
                                           broadcast_dimensions);

    case BINOP_COMPLEX: {
      if (!ShapeUtil::ElementIsFloating(lhs)) {
        return InvalidArgument(
            "Expected element type in shape to be floating for complex compose "
            "operation; got %s.",
            PrimitiveType_Name(lhs.element_type()).c_str());
      }
      TF_ASSIGN_OR_RETURN(const Shape& shape,
                          InferElementwiseBinaryOpShape(operation, lhs, rhs,
                                                        broadcast_dimensions));
      if (lhs.element_type() == F32 && rhs.element_type() == F32) {
        return ShapeUtil::ChangeElementType(shape, C64);
      } else {
        return Unimplemented("Complex component type is not implemented.");
      }
    }
    case BINOP_AND:
    case BINOP_OR:
      if (lhs.element_type() != PRED &&
          !primitive_util::IsIntegralType(lhs.element_type())) {
        return InvalidArgument(
            "Expected pred or integral type in argument to and/or operation; "
            "got %s.",
            PrimitiveType_Name(lhs.element_type()).c_str());
      }
      return InferElementwiseBinaryOpShape(operation, lhs, rhs,
                                           broadcast_dimensions);
    case BINOP_EQ:
    case BINOP_GE:
    case BINOP_GT:
    case BINOP_LE:
    case BINOP_LT:
    case BINOP_NE: {
      TF_ASSIGN_OR_RETURN(const Shape& shape,
                          InferElementwiseBinaryOpShape(operation, lhs, rhs,
                                                        broadcast_dimensions));
      return ShapeUtil::ChangeElementType(shape, PRED);
    }
    default:
      return Unimplemented(
          "Binary op shape inference: %s; lhs: %s; rhs: %s is not implemented.",
          BinaryOperation_Name(operation).c_str(),
          lhs.ShortDebugString().c_str(), rhs.ShortDebugString().c_str());
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferTernaryOpShape(
    HloOpcode opcode, const HloInstruction* lhs, const HloInstruction* rhs,
    const HloInstruction* ehs) {
  return InferTernaryOpShape(opcode, lhs->shape(), rhs->shape(), ehs->shape());
}

/* static */ StatusOr<Shape> ShapeInference::InferTernaryOpShape(
    HloOpcode opcode, const Shape& lhs, const Shape& rhs, const Shape& ehs) {
  return InferTernaryOpShape(OpcodeToTernaryOperation(opcode), lhs, rhs, ehs);
}

/* static */ StatusOr<Shape> ShapeInference::InferTernaryOpShape(
    TernaryOperation operation, const Shape& lhs, const Shape& rhs,
    const Shape& ehs) {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(ehs));
  switch (operation) {
    case TRIOP_CLAMP:
      return InferClampShape(lhs, rhs, ehs);
    case TRIOP_SELECT:
      return InferSelectShape(lhs, rhs, ehs);
    default:
      return InvalidArgument("Unknown operation %s.",
                             TernaryOperation_Name(operation).c_str());
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferVariadicOpShape(
    HloOpcode opcode,
    tensorflow::gtl::ArraySlice<const HloInstruction*> operands) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : operands) {
    operand_shapes.push_back(&operand->shape());
  }
  return InferVariadicOpShape(opcode, operand_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferVariadicOpShape(
    HloOpcode opcode,
    tensorflow::gtl::ArraySlice<const Shape*> operand_shapes) {
  return InferVariadicOpShape(OpcodeToVariadicOperation(opcode),
                              operand_shapes);
}

/* static */ StatusOr<Shape> ShapeInference::InferVariadicOpShape(
    VariadicOperation operation,
    tensorflow::gtl::ArraySlice<const Shape*> operand_shapes) {
  for (const Shape* shape : operand_shapes) {
    TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(*shape));
  }
  switch (operation) {
    case VAROP_TUPLE: {
      Shape result = ShapeUtil::MakeTupleShape({});
      for (const Shape* shape : operand_shapes) {
        ShapeUtil::AppendShapeToTuple(*shape, &result);
      }
      return result;
    }
    default:
      return InvalidArgument("Unknown operation %s.",
                             VariadicOperation_Name(operation).c_str());
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferMapShape(
    tensorflow::gtl::ArraySlice<const Shape*> arg_shapes,
    const ProgramShape& to_apply,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  if (arg_shapes.empty()) {
    return InvalidArgument("Map expects at least one argument.");
  }

  // All arguments must have the same shape.
  const Shape* arg_shape = arg_shapes[0];
  for (size_t i = 1; i < arg_shapes.size(); ++i) {
    TF_RETURN_IF_ERROR(
        ExpectNotTupleOrOpaque(*arg_shapes[i], "operand of map"));

    if (ShapeUtil::CompatibleIgnoringFpPrecision(*arg_shapes[i], *arg_shape)) {
      continue;
    }
    if (!ShapeUtil::IsTuple(*arg_shapes[i]) &&
        !ShapeUtil::IsTuple(*arg_shape) &&
        ShapeUtil::SameElementTypeIgnoringFpPrecision(*arg_shapes[i],
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
        Join(pieces, ", ").c_str());
  }

  // Check that dimensions.size == arg_shape.dimensions_size() (we currently
  // only support mapping across all dimensions: i.e. scalar map functions).
  if (dimensions.size() != arg_shape->dimensions_size()) {
    return InvalidArgument(
        "Map applied to a subset of dimensions currently not supported: "
        "arg_dimension_size: %d, requested_map_dimensions_size: %zu.",
        arg_shape->dimensions_size(), dimensions.size());
  }

  // Check that requested map dimensions numbers are monotonically increasing.
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] != i) {
      return InvalidArgument(
          "Map requires monotonically increasing dimension numbers; got: %s.",
          Join(dimensions, ", ").c_str());
    }
  }

  // The applied function's arity equals the number of arguments.
  if (arg_shapes.size() != to_apply.parameters_size()) {
    return InvalidArgument(
        "Map applied function arity must match number of arguments; got: "
        "arity: %d, arguments: %zu.",
        to_apply.parameters_size(), arg_shapes.size());
  }

  // The parameters should all be scalars, and the output too.
  const Shape& output_shape = to_apply.result();
  if (!ShapeUtil::IsScalar(output_shape)) {
    return InvalidArgument(
        "Mapped computation's result has to be a scalar; got: %s.",
        ShapeUtil::HumanString(output_shape).c_str());
  }

  for (int i = 0; i < to_apply.parameters_size(); ++i) {
    const Shape& parameter_shape = to_apply.parameters(i);

    if (!ShapeUtil::IsScalar(parameter_shape)) {
      return InvalidArgument(
          "Mapped computation's parameter has to be a scalar; "
          "got parameter %d shape: %s.",
          i, ShapeUtil::HumanString(parameter_shape).c_str());
    }

    if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(parameter_shape,
                                                       *arg_shape)) {
      return InvalidArgument(
          "Mapped computation's parameter type has to match argument element "
          "type; got parameter %d shape: %s, argument shape: %s.",
          i, ShapeUtil::HumanString(parameter_shape).c_str(),
          ShapeUtil::HumanString(*arg_shape).c_str());
    }
  }

  return ShapeUtil::MakeShape(output_shape.element_type(),
                              AsInt64Slice(arg_shape->dimensions()));
}

/* static */ StatusOr<Shape> ShapeInference::InferBatchNormTrainingShape(
    const Shape& operand_shape, const Shape& scale_shape,
    const Shape& offset_shape, int64 feature_index) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(operand_shape, "operand of batch norm training"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      offset_shape, "offset input of batch norm training"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      scale_shape, "scale input of batch norm training"));

  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(operand_shape) ==
               tensorflow::Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(offset_shape) ==
               tensorflow::Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(scale_shape) ==
               tensorflow::Status::OK());

  if (feature_index >= ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-training to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %lld, and rank %lld.",
        feature_index, ShapeUtil::Rank(operand_shape));
  }

  if (feature_index < 0) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-training to "
        "be a non-negative number, got %lld.",
        feature_index);
  }

  if (ShapeUtil::Rank(operand_shape) < 1) {
    return InvalidArgument(
        "Expected the rank of operand to "
        "batch-norm-training to be at least 1; got %lld.",
        ShapeUtil::Rank(operand_shape));
  }

  if (ShapeUtil::Rank(offset_shape) != 1) {
    return InvalidArgument(
        "Offset input of batch-norm-training must have"
        " rank 1, but has rank %lld.",
        ShapeUtil::Rank(offset_shape));
  }

  if (ShapeUtil::Rank(scale_shape) != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-training must have"
        " rank 1, but has rank %lld.",
        ShapeUtil::Rank(scale_shape));
  }

  if (!ShapeUtil::ElementIsFloating(operand_shape)) {
    return InvalidArgument(
        "The operand to batch-norm-training must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(offset_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-training, "
        "but the shape of offset factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(offset_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(scale_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-training, "
        "but the shape of scale factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(scale_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  const int64 feature_count = operand_shape.dimensions(feature_index);
  Shape output_shape_for_mean_and_var =
      ShapeUtil::MakeShape(operand_shape.element_type(), {feature_count});

  if (ShapeUtil::GetDimension(offset_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of offset factor should be the same as feature count,"
        "but the size of offset factor is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(offset_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(scale_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of scale factor should be the same as feature count,"
        "but the size of scale factor is %lld "
        "and the feature count is %lld.",
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
      ExpectNotTupleOrOpaque(operand_shape, "operand of batch norm inference"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      offset_shape, "offset input of batch norm inference"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      scale_shape, "scale input of batch norm inference"));

  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(operand_shape) ==
               tensorflow::Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(offset_shape) ==
               tensorflow::Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(scale_shape) ==
               tensorflow::Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(mean_shape) ==
               tensorflow::Status::OK());
  TF_RET_CHECK(ShapeUtil::ValidateShapeWithOptionalLayout(variance_shape) ==
               tensorflow::Status::OK());

  if (feature_index >= ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-inference to be "
        "smaller than the rank of operand_shape; "
        "got feature_index %lld, and rank %lld.",
        feature_index, ShapeUtil::Rank(operand_shape));
  }

  if (feature_index < 0) {
    return InvalidArgument(
        "Expected feature_index of batch-norm-inference to "
        "be a non-negative number, got %lld.",
        feature_index);
  }

  if (ShapeUtil::Rank(operand_shape) < 1) {
    return InvalidArgument(
        "Expected the rank of operand to "
        "batch-norm-inference to be at least 1; got %lld.",
        ShapeUtil::Rank(operand_shape));
  }

  if (ShapeUtil::Rank(offset_shape) != 1) {
    return InvalidArgument(
        "Offset input of batch-norm-inference must have"
        " rank 1, but has rank %lld.",
        ShapeUtil::Rank(offset_shape));
  }

  if (ShapeUtil::Rank(scale_shape) != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-inference must have"
        " rank 1, but has rank %lld.",
        ShapeUtil::Rank(scale_shape));
  }

  if (!ShapeUtil::ElementIsFloating(operand_shape)) {
    return InvalidArgument(
        "The operand to batch-norm-inference must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(offset_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of offset factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(offset_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(scale_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of scale factor is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(scale_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(mean_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of mean is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(variance_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for "
        "batch-norm-inference, "
        "but the shape of variance is %s "
        "and the shape of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()).c_str(),
        PrimitiveType_Name(variance_shape.element_type()).c_str());
  }

  const int64 feature_count = operand_shape.dimensions(feature_index);
  Shape output_shape_for_mean_and_var =
      ShapeUtil::MakeShape(operand_shape.element_type(), {feature_count});

  if (ShapeUtil::GetDimension(offset_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of offset factor should be the same as feature count,"
        "but the size of offset factor is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(offset_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(scale_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of scale factor should be the same as feature count,"
        "but the size of scale factor is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(scale_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(mean_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of mean should be the same as feature count,"
        "but the size of mean is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(mean_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(variance_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of variance should be the same as feature count,"
        "but the size of variance is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(variance_shape, 0), feature_count);
  }

  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferBatchNormGradShape(
    const Shape& operand_shape, const Shape& scale_shape,
    const Shape& mean_shape, const Shape& var_shape,
    const Shape& output_grad_shape, int64 feature_index) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(operand_shape, "operand of batch norm grad"));
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(scale_shape, "scale input of batch norm grad"));
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(mean_shape, "mean input of batch norm grad"));
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(var_shape, "var input of batch norm grad"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      output_grad_shape, "output_grad input of batch norm grad"));

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
        "got feature_index %lld, and rank %lld.",
        feature_index, ShapeUtil::Rank(operand_shape));
  }

  if (ShapeUtil::Rank(operand_shape) != ShapeUtil::Rank(output_grad_shape)) {
    return InvalidArgument(
        "Expected operand_shape of batch-norm-grad to have the same rank as"
        " output_grad_shape; got rank(oprand_shape) %lld, and"
        " rank(output_grad_shape) %lld.",
        ShapeUtil::Rank(operand_shape), ShapeUtil::Rank(output_grad_shape));
  }

  if (ShapeUtil::Rank(mean_shape) != 1) {
    return InvalidArgument(
        "Mean input of batch-norm-grad must have"
        " rank 1, but has rank %lld.",
        ShapeUtil::Rank(mean_shape));
  }

  if (ShapeUtil::Rank(scale_shape) != 1) {
    return InvalidArgument(
        "Scale input of batch-norm-grad must have"
        " rank 1, but has rank %lld.",
        ShapeUtil::Rank(scale_shape));
  }

  if (ShapeUtil::Rank(var_shape) != 1) {
    return InvalidArgument(
        "Var input of batch-norm-grad must have"
        " rank 1, but has rank %lld.",
        ShapeUtil::Rank(var_shape));
  }

  if (!ShapeUtil::ElementIsFloating(operand_shape)) {
    return InvalidArgument(
        "The operand to batch-norm-grad must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::ElementIsFloating(output_grad_shape)) {
    return InvalidArgument(
        "The output_grad to batch-norm-grad must have a floating point "
        "element type, but the shape is %s.",
        PrimitiveType_Name(output_grad_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(output_grad_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of output_grad is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(output_grad_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(scale_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of scale factor is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(scale_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(mean_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of mean is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(var_shape,
                                                     operand_shape)) {
    return InvalidArgument(
        "The inputs should have the same element type for batch-norm-grad, "
        "but the element type of mean is %s "
        "and the element type of operand is %s.",
        PrimitiveType_Name(mean_shape.element_type()).c_str(),
        PrimitiveType_Name(operand_shape.element_type()).c_str());
  }

  const int64 feature_count = operand_shape.dimensions(feature_index);

  Shape feature_shape =
      ShapeUtil::MakeShape(operand_shape.element_type(), {feature_count});

  if (ShapeUtil::GetDimension(mean_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of mean should be the same as feature count,"
        "but the size of offset factor is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(mean_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(scale_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of scale factor should be the same as feature count,"
        "but the size of scale factor is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(scale_shape, 0), feature_count);
  }

  if (ShapeUtil::GetDimension(var_shape, 0) != feature_count) {
    return InvalidArgument(
        "The size of variance should be the same as feature count,"
        "but the size of variance is %lld "
        "and the feature count is %lld.",
        ShapeUtil::GetDimension(var_shape, 0), feature_count);
  }

  // Verify operand_shape and output_grad_shape have same bounds.
  for (int64 i = 0; i < ShapeUtil::Rank(operand_shape); ++i) {
    if (ShapeUtil::GetDimension(operand_shape, i) !=
        ShapeUtil::GetDimension(output_grad_shape, i)) {
      return InvalidArgument(
          "The bounds of operand shape should be the same as output_grad's,"
          "but the bound of operand_shape at dimension %lld is %lld "
          "and the bound of output_grad_shape is %lld.",
          i, ShapeUtil::GetDimension(operand_shape, i),
          ShapeUtil::GetDimension(output_grad_shape, i));
    }
  }

  return ShapeUtil::MakeTupleShape(
      {operand_shape, feature_shape, feature_shape});
}

/* static */ StatusOr<Shape> ShapeInference::InferConvolveShape(
    const Shape& lhs, const Shape& rhs, const Window& window,
    const ConvolutionDimensionNumbers& dnums) {
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(lhs, "lhs of convolution"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(rhs, "rhs of convolution"));

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) {
    return InvalidArgument(
        "Convolution with different element types: %s and %s.",
        ShapeUtil::HumanString(lhs).c_str(),
        ShapeUtil::HumanString(rhs).c_str());
  }
  if (dnums.input_spatial_dimensions_size() !=
      dnums.kernel_spatial_dimensions_size()) {
    return InvalidArgument(
        "Both arguments to convolution must have same number of dimensions.\n"
        "Window: %s",
        window.DebugString().c_str());
  }

  const int num_spatial_dims = dnums.input_spatial_dimensions_size();
  if (window.dimensions_size() != num_spatial_dims) {
    return InvalidArgument(
        "Window must have same number of dimensions as dimension numbers.\n"
        "Window: %s\nDimension numbers: %s.",
        window.DebugString().c_str(), dnums.DebugString().c_str());
  }

  const int num_dims = num_spatial_dims + 2;
  if (ShapeUtil::Rank(lhs) != num_dims) {
    return InvalidArgument(
        "The LHS argument to a convolution should have rank %d; lhs: %s.",
        num_dims, ShapeUtil::HumanString(lhs).c_str());
  }
  if (ShapeUtil::Rank(rhs) != num_dims) {
    return InvalidArgument(
        "The RHS argument to a convolution should have rank %d; lhs: %s.",
        num_dims, ShapeUtil::HumanString(lhs).c_str());
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
        dnums.DebugString().c_str());
  }

  if (input_dnums != expected_dnums) {
    return InvalidArgument(
        "Input dimensions of convolution must contain each dimension exactly "
        "once: %s.",
        dnums.DebugString().c_str());
  }
  if (window_dnums != expected_dnums) {
    return InvalidArgument(
        "Window dimensions of convolution must contain each dimension exactly "
        "once: %s.",
        dnums.DebugString().c_str());
  }
  if (output_dnums != expected_dnums) {
    return InvalidArgument(
        "Output dimensions of convolution must contain each dimension exactly "
        "once: %s.",
        dnums.DebugString().c_str());
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

  if (input_features != kernel_input_features) {
    return InvalidArgument(
        "Expected LHS feature dimension (value %lld) to match RHS "
        "input feature dimension (value %lld); got <conv>(%s, %s)\n"
        "Dimension numbers: {%s}.",
        input_features, kernel_input_features,
        ShapeUtil::HumanString(lhs).c_str(),
        ShapeUtil::HumanString(rhs).c_str(), dnums.DebugString().c_str());
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
        ShapeUtil::HumanString(rhs).c_str(), window.ShortDebugString().c_str(),
        dnums.ShortDebugString().c_str());
  }

  Shape base_shape =
      ShapeUtil::MakeShape(lhs.element_type(), input_spatial_dims);
  TF_ASSIGN_OR_RETURN(
      Shape window_output_shape,
      InferWindowOutputShape(base_shape, window, lhs.element_type(),
                             /*allow_negative_padding=*/true));

  std::vector<int64> dimensions(num_dims);
  dimensions[dnums.output_batch_dimension()] = input_batch;
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
    const tensorflow::gtl::ArraySlice<int64> fft_length) {
  const int64 fft_rank = fft_length.size();
  if (fft_rank < 1 || fft_rank > 3) {
    return InvalidArgument("FFT only supports ranks 1-3; got %lld.", fft_rank);
  }
#define RET_CHECK_RANK(x)                              \
  if (x.dimensions_size() < fft_rank) {                \
    return InvalidArgument(                            \
        "FFT of rank %lld requires input of at least " \
        "same rank; got input of rank %d",             \
        fft_rank, x.dimensions_size());                \
  }
  switch (fft_type) {
    case FFT:
    case IFFT:
      if (in.element_type() != C64) {
        return InvalidArgument("%s requires C64 input type, found %s.",
                               FftType_Name(fft_type).c_str(),
                               PrimitiveType_Name(in.element_type()).c_str());
      }
      RET_CHECK_RANK(in);
      return in;
    case RFFT: {
      if (in.element_type() != F32) {
        return InvalidArgument("RFFT requires F32 input type, found %s.",
                               PrimitiveType_Name(in.element_type()).c_str());
      }
      RET_CHECK_RANK(in);
      for (int i = 0; i < fft_rank; i++) {
        if (in.dimensions(in.dimensions_size() - fft_rank + i) !=
            fft_length[i]) {
          return InvalidArgument(
              "RFFT requires innermost dimensions match fft_length but "
              "dimension %lld is %lld and should be %lld.",
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
                               PrimitiveType_Name(in.element_type()).c_str());
      }
      RET_CHECK_RANK(in);
      Shape result = ShapeUtil::ComplexComponentShape(in);
      for (int i = 0; i < fft_rank - 1; i++) {
        if (in.dimensions(in.dimensions_size() - fft_rank + i) !=
            fft_length[i]) {
          return InvalidArgument(
              "IRFFT requires all but one innermost dimensions match "
              "fft_length, but dimension %lld is %lld and should be %lld.",
              in.dimensions_size() - fft_rank + i,
              in.dimensions(in.dimensions_size() - fft_rank + i),
              fft_length[i]);
        }
      }
      if (in.dimensions(in.dimensions_size() - 1) !=
          fft_length[fft_rank - 1] / 2 + 1) {
        return InvalidArgument(
            "IRFFT requires innermost dimension matches fft_length/2+1, but "
            "dimension %d is %lld and should be %lld.",
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

/* static */ StatusOr<Shape> ShapeInference::InferCrossReplicaSumShape(
    tensorflow::gtl::ArraySlice<const Shape*> operand_shapes) {
  for (const Shape* operand_shape : operand_shapes) {
    TF_RETURN_IF_ERROR(
        ExpectNotTupleOrOpaque(*operand_shape, "operand of cross replica sum"));
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

/* static */ StatusOr<Shape> ShapeInference::InferReduceShape(
    const Shape& arg, const Shape& init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
    const ProgramShape& to_apply) {
  // Check that the dimension to reduce are in-bounds for the given shape.
  for (int64 dimension : dimensions_to_reduce) {
    if (dimension >= ShapeUtil::Rank(arg) || dimension < 0) {
      return InvalidArgument(
          "Reducing out-of-bounds dimension %lld in shape %s.", dimension,
          ShapeUtil::HumanString(arg).c_str());
    }
  }
  TF_RETURN_IF_ERROR(
      VerifyReducerShape(to_apply, init_value, arg.element_type()));

  std::set<int64> dimensions_to_reduce_set(dimensions_to_reduce.begin(),
                                           dimensions_to_reduce.end());
  std::vector<int64> new_dimensions;
  for (int i = 0; i < ShapeUtil::Rank(arg); ++i) {
    if (dimensions_to_reduce_set.find(i) == dimensions_to_reduce_set.end()) {
      new_dimensions.push_back(arg.dimensions(i));
    }
  }

  return ShapeUtil::MakeShape(to_apply.result().element_type(), new_dimensions);
}

/* static */ StatusOr<Shape> ShapeInference::InferReduceWindowShape(
    const Shape& operand_shape, const Shape& init_value_shape,
    const Window& window, const ProgramShape& to_apply_shape) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(operand_shape, "operand of reduce-window"));
  TF_RETURN_IF_ERROR(VerifyReducerShape(to_apply_shape, init_value_shape,
                                        operand_shape.element_type()));
  return InferWindowOutputShape(operand_shape, window,
                                init_value_shape.element_type(),
                                /*allow_negative_padding=*/false);
}

/* static */ StatusOr<Shape> ShapeInference::InferSelectAndScatterShape(
    const Shape& operand_shape, const ProgramShape& select_shape,
    const Window& window, const Shape& source_shape,
    const Shape& init_value_shape, const ProgramShape& scatter_shape) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(operand_shape, "operand of select-and-scatter"));

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
        ShapeUtil::HumanString(select_shape.parameters(0)).c_str(),
        ShapeUtil::HumanString(operand_element_shape).c_str());
  }
  if (!ShapeUtil::CompatibleIgnoringFpPrecision(operand_element_shape,
                                                select_shape.parameters(1))) {
    return InvalidArgument(
        "Select function's second parameter shape currently must "
        "match the operand element shape, but got %s vs %s.",
        ShapeUtil::HumanString(select_shape.parameters(1)).c_str(),
        ShapeUtil::HumanString(operand_element_shape).c_str());
  }

  // Check if the scatter function has a proper shape as a reduction.
  TF_RETURN_IF_ERROR(VerifyReducerShape(scatter_shape, init_value_shape,
                                        source_shape.element_type()));

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
        ShapeUtil::HumanString(source_shape).c_str(),
        ShapeUtil::HumanString(window_result_shape).c_str());
  }
  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferSliceShape(
    const Shape& arg, tensorflow::gtl::ArraySlice<int64> starts,
    tensorflow::gtl::ArraySlice<int64> limits,
    tensorflow::gtl::ArraySlice<int64> strides) {
  auto error = [&](const string& message) {
    return InvalidArgument(
        "%s in slice operation; argument shape: %s; starts: {%s}; limits: "
        "{%s}; strides: {%s}.",
        message.c_str(), ShapeUtil::HumanString(arg).c_str(),
        Join(starts, ",").c_str(), Join(limits, ",").c_str(),
        Join(strides, ",").c_str());
  };
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(arg, "operand of slice"));
  VLOG(2) << tensorflow::strings::Printf(
      "slicing shape %s starts={%s} limits={%s}",
      ShapeUtil::HumanString(arg).c_str(), Join(starts, ", ").c_str(),
      Join(limits, ", ").c_str());

  if (starts.size() != limits.size()) {
    return error(Printf("slice start and limit sizes differ: %zu vs %zu",
                        starts.size(), limits.size()));
  }

  if (starts.size() != strides.size()) {
    return error(Printf("slice start and strides sizes differ: %zu vs %zu",
                        starts.size(), strides.size()));
  }

  if (starts.size() != ShapeUtil::Rank(arg)) {
    return InvalidArgument(
        "Slice index count does not match argument rank: %zu vs %lld.",
        starts.size(), ShapeUtil::Rank(arg));
  }

  std::vector<int64> sizes;
  for (int64 dimension = 0; dimension < starts.size(); ++dimension) {
    int64 start_index = starts[dimension];
    int64 limit_index = limits[dimension];
    int64 stride = strides[dimension];
    if (start_index < 0) {
      return InvalidArgument("Negative start index to slice: %lld.",
                             start_index);
    }
    if (limit_index > arg.dimensions(dimension)) {
      return error(
          Printf("limit index (%lld) must be less than or equal to dimension "
                 "size (%lld)",
                 limit_index, arg.dimensions(dimension)));
    }
    VLOG(2) << tensorflow::strings::Printf("starts[%lld] = %lld", dimension,
                                           start_index);
    VLOG(2) << tensorflow::strings::Printf("limits[%lld] = %lld", dimension,
                                           limit_index);
    if (start_index > limit_index) {
      return error(
          Printf("limit index (%lld) must be greater or equal to "
                 "start index (%lld) in slice with positive stride",
                 limit_index, start_index));
    }
    if (stride <= 0) {
      return InvalidArgument("Stride (%lld) must be positive.", stride);
    }
    sizes.push_back((limit_index - start_index + stride - 1) / stride);
  }

  return ShapeUtil::MakeShape(arg.element_type(), sizes);
}

/* static */ StatusOr<Shape> ShapeInference::InferDynamicSliceShape(
    const Shape& operand_shape, const Shape& start_indices_shape,
    tensorflow::gtl::ArraySlice<int64> slice_sizes) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(operand_shape, "operand of dynamic slice"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(start_indices_shape,
                                            "start indices of dynamic slice"));

  VLOG(2) << tensorflow::strings::Printf(
      "slicing shape %s at dynamic start_indices %s with slice_sizes={%s}",
      ShapeUtil::HumanString(operand_shape).c_str(),
      ShapeUtil::HumanString(start_indices_shape).c_str(),
      Join(slice_sizes, ", ").c_str());

  if (ShapeUtil::Rank(start_indices_shape) != 1) {
    return InvalidArgument(
        "Dynamic slice start indices of rank %lld must be rank1.",
        ShapeUtil::Rank(start_indices_shape));
  }

  if (!ShapeUtil::ElementIsIntegral(start_indices_shape)) {
    return InvalidArgument(
        "Dynamic slice start indices must be of integral type.");
  }

  const int64 start_num_dims = start_indices_shape.dimensions(0);
  if (ShapeUtil::Rank(operand_shape) != start_num_dims) {
    return InvalidArgument(
        "Dynamic slice start number of dimensions %lld (%s) must match rank "
        "%lld of slice input (%s).",
        start_num_dims, ShapeUtil::HumanString(start_indices_shape).c_str(),
        ShapeUtil::Rank(operand_shape),
        ShapeUtil::HumanString(operand_shape).c_str());
  }

  if (slice_sizes.size() != ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Dynamic slice index count does not match argument rank: %zu vs %lld.",
        slice_sizes.size(), ShapeUtil::Rank(operand_shape));
  }

  for (int64 dim = 0; dim < slice_sizes.size(); ++dim) {
    const int64 input_dim_size = operand_shape.dimensions(dim);
    const int64 slice_dim_size = slice_sizes[dim];
    if (slice_dim_size < 0) {
      return InvalidArgument("Negative size index to dynamic slice: %lld.",
                             slice_dim_size);
    }
    if (slice_dim_size > input_dim_size) {
      return InvalidArgument(
          "Slice dim size %lld greater than dynamic slice dimension: %lld.",
          slice_dim_size, input_dim_size);
    }
    VLOG(2) << tensorflow::strings::Printf("slice_sizes[%lld] = %lld", dim,
                                           slice_dim_size);
  }

  return ShapeUtil::MakeShape(operand_shape.element_type(), slice_sizes);
}

/* static */ StatusOr<Shape> ShapeInference::InferDynamicUpdateSliceShape(
    const Shape& operand_shape, const Shape& update_shape,
    const Shape& start_indices_shape) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(operand_shape, "operand of dynamic update slice"));
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(update_shape, "update of dynamic update slice"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      start_indices_shape, "start indices of dynamic update slice"));

  VLOG(2) << tensorflow::strings::Printf(
      "updating slice of shape %s at dynamic start_indices %s with update "
      "shape %s",
      ShapeUtil::HumanString(operand_shape).c_str(),
      ShapeUtil::HumanString(start_indices_shape).c_str(),
      ShapeUtil::HumanString(update_shape).c_str());

  if (ShapeUtil::Rank(start_indices_shape) != 1) {
    return InvalidArgument(
        "Dynamic update slice start indices of rank %lld must be rank1.",
        ShapeUtil::Rank(start_indices_shape));
  }

  if (!ShapeUtil::ElementIsIntegral(start_indices_shape)) {
    return InvalidArgument(
        "Dynamic update slice start indices must be of integral type.");
  }

  const int64 start_num_dims = start_indices_shape.dimensions(0);
  if (ShapeUtil::Rank(operand_shape) != start_num_dims) {
    return InvalidArgument(
        "Dynamic update slice start number of dimensions %lld (%s) must match "
        "rank %lld of slice input (%s).",
        start_num_dims, ShapeUtil::HumanString(start_indices_shape).c_str(),
        ShapeUtil::Rank(operand_shape),
        ShapeUtil::HumanString(operand_shape).c_str());
  }

  if (ShapeUtil::Rank(update_shape) != ShapeUtil::Rank(operand_shape)) {
    return InvalidArgument(
        "Dynamic update slice update rank does not match argument rank: "
        "%lld vs %lld.",
        ShapeUtil::Rank(update_shape), ShapeUtil::Rank(operand_shape));
  }

  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(operand_shape,
                                                     update_shape)) {
    return InvalidArgument(
        "Dynamic update slice update element type does not match argument. "
        "operand.element_type: %s vs update.element_type: %s.",
        PrimitiveType_Name(operand_shape.element_type()).c_str(),
        PrimitiveType_Name(update_shape.element_type()).c_str());
  }

  for (int64 dim = 0; dim < ShapeUtil::Rank(operand_shape); ++dim) {
    const int64 input_dim_size = operand_shape.dimensions(dim);
    const int64 update_dim_size = update_shape.dimensions(dim);
    if (update_dim_size < 0) {
      return InvalidArgument(
          "Size index %lld to dynamic update slice must be >= 0.",
          update_dim_size);
    }
    if (update_dim_size > input_dim_size) {
      return InvalidArgument(
          "Update dim size %lld greater than dynamic slice dimension: %lld.",
          update_dim_size, input_dim_size);
    }
    VLOG(2) << tensorflow::strings::Printf("update_sizes[%lld] = %lld", dim,
                                           update_dim_size);
  }

  return operand_shape;
}

/*static */ StatusOr<Shape> ShapeInference::InferReverseShape(
    const Shape& operand_shape, tensorflow::gtl::ArraySlice<int64> dimensions) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(operand_shape, "operand of reverse"));
  if (!AllUnique(dimensions)) {
    return InvalidArgument("a dimension number is duplicated in reverse");
  }
  for (int64 dimension : dimensions) {
    if (dimension >= ShapeUtil::Rank(operand_shape) || dimension < 0) {
      return InvalidArgument(
          "One of the reverse dimensions (%lld) is out-of-bounds in shape %s.",
          dimension, ShapeUtil::HumanString(operand_shape).c_str());
    }
  }
  return operand_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferGetTupleElementShape(
    const Shape& arg, int64 index) {
  if (!ShapeUtil::IsTuple(arg)) {
    return InvalidArgument(
        "Cannot infer shape: attempting to index into non-tuple: %s.",
        ShapeUtil::HumanString(arg).c_str());
  }

  if (index >= arg.tuple_shapes_size()) {
    return InvalidArgument(
        "Cannot infer shape: attempt to index out of tuple bounds: %lld "
        ">= %d in shape %s.",
        index, arg.tuple_shapes_size(), ShapeUtil::HumanString(arg).c_str());
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
    return tensorflow::strings::Printf(
        "Condition: %s; body: %s; init: %s.",
        ShapeUtil::HumanString(condition).c_str(),
        ShapeUtil::HumanString(body).c_str(),
        ShapeUtil::HumanString(init).c_str());
  };

  // Check the shapes of computation parameters and return types.
  if (!ShapeUtil::ShapeIs(condition.result(), PRED, {})) {
    return InvalidArgument("Condition must return a boolean; got %s.",
                           shape_string().c_str());
  }
  if (!ShapeUtil::Compatible(body.result(), condition.parameters(0)) ||
      !ShapeUtil::Compatible(body.result(), body.parameters(0)) ||
      !ShapeUtil::Compatible(body.result(), init)) {
    return InvalidArgument(
        "The parameter of condition and body, the result of the body, and init "
        "must all have the same shape; got %s.",
        shape_string().c_str());
  }

  return init;
}

/* static */ StatusOr<Shape> ShapeInference::InferConditionalShape(
    const Shape& predicate, const Shape& true_operand,
    const Shape& false_operand, const ProgramShape& true_computation,
    const ProgramShape& false_computation) {
  if (!ShapeUtil::ShapeIs(predicate, PRED, {})) {
    return InvalidArgument("Predicate must be a boolean; got %s.",
                           ShapeUtil::HumanString(predicate).c_str());
  }

  if (true_computation.parameters_size() != 1) {
    return InvalidArgument("true_computation must take 1 argument; got %d.",
                           true_computation.parameters_size());
  }
  if (!ShapeUtil::Compatible(true_computation.parameters(0), true_operand)) {
    auto true_shape_string = [&]() {
      return tensorflow::strings::Printf(
          "true_operand: %s; true_computation: %s",
          ShapeUtil::HumanString(true_operand).c_str(),
          ShapeUtil::HumanString(true_computation).c_str());
    };
    return InvalidArgument(
        "true_operand must match the shape of the only parameter of "
        "true_computation: got %s.",
        true_shape_string().c_str());
  }

  if (false_computation.parameters_size() != 1) {
    return InvalidArgument("false_computation must take 1 argument; got %d.",
                           false_computation.parameters_size());
  }
  if (!ShapeUtil::Compatible(false_computation.parameters(0), false_operand)) {
    auto false_shape_string = [&]() {
      return tensorflow::strings::Printf(
          "false_operand: %s; false_computation: %s",
          ShapeUtil::HumanString(false_operand).c_str(),
          ShapeUtil::HumanString(false_computation).c_str());
    };
    return InvalidArgument(
        "false_operand must match the shape of the only parameter of "
        "false_computation: got %s.",
        false_shape_string().c_str());
  }
  if (!ShapeUtil::Compatible(true_computation.result(),
                             false_computation.result())) {
    auto shape_string = [&]() {
      return tensorflow::strings::Printf(
          "true_computation result: %s; false_computation result: %s.",
          ShapeUtil::HumanString(true_computation.result()).c_str(),
          ShapeUtil::HumanString(false_computation.result()).c_str());
    };
    return InvalidArgument(
        "the result of true_computation and false_computation must have the "
        "same shape: got %s.",
        shape_string().c_str());
  }
  return true_computation.result();
}

/* static */ StatusOr<Shape> ShapeInference::InferBroadcastShape(
    const Shape& operand, tensorflow::gtl::ArraySlice<int64> broadcast_sizes) {
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(operand, "operand of broadcast"));
  for (int64 size : broadcast_sizes) {
    if (size < 0) {
      return InvalidArgument("Broadcast with negative dimension size %lld.",
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

/* static */ StatusOr<Shape> ShapeInference::InferReshapeShape(
    const Shape& operand, tensorflow::gtl::ArraySlice<int64> dimensions,
    tensorflow::gtl::ArraySlice<int64> new_sizes) {
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(operand, "reshape"));

  Shape inferred_shape =
      ShapeUtil::MakeShape(operand.element_type(), new_sizes);
  VLOG(3) << "Reshape inferred shape: "
          << ShapeUtil::HumanString(inferred_shape);

  if (ShapeUtil::ElementsIn(operand) != ShapeUtil::ElementsIn(inferred_shape)) {
    return InvalidArgument(
        "Reshape operation has mismatched element counts: from=%lld (%s) "
        "to=%lld (%s).",
        ShapeUtil::ElementsIn(operand), ShapeUtil::HumanString(operand).c_str(),
        ShapeUtil::ElementsIn(inferred_shape),
        ShapeUtil::HumanString(inferred_shape).c_str());
  }

  std::vector<int64> indices(ShapeUtil::Rank(operand));
  std::iota(indices.begin(), indices.end(), 0);
  if (dimensions.size() != ShapeUtil::Rank(operand) ||
      !std::is_permutation(dimensions.begin(), dimensions.end(),
                           indices.begin())) {
    return InvalidArgument(
        "Reshape dimensions [%s] are not a permutation of the operand "
        "dimensions (operand shape is %s).",
        Join(dimensions, ",").c_str(), ShapeUtil::HumanString(operand).c_str());
  }

  return inferred_shape;
}

/* static */ StatusOr<Shape> ShapeInference::InferTransposeShape(
    const Shape& operand, tensorflow::gtl::ArraySlice<int64> dimensions) {
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(operand, "transpose"));

  std::vector<int64> indices(ShapeUtil::Rank(operand));
  std::iota(indices.begin(), indices.end(), 0);
  if (dimensions.size() != ShapeUtil::Rank(operand) ||
      !std::is_permutation(dimensions.begin(), dimensions.end(),
                           indices.begin())) {
    return InvalidArgument(
        "Transpose dimensions not a permutation of the operand dimensions.");
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
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(min, "clamp min"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(operand, "clamp operand"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(max, "clamp max"));
  if (!ShapeUtil::SameElementTypeIgnoringFpPrecision(min, operand) ||
      !ShapeUtil::SameElementTypeIgnoringFpPrecision(max, operand)) {
    return InvalidArgument("Clamp with different operand types: %s, %s, %s.",
                           ShapeUtil::HumanString(min).c_str(),
                           ShapeUtil::HumanString(operand).c_str(),
                           ShapeUtil::HumanString(max).c_str());
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
  return Unimplemented(
      "%s, %s <clamp> %s is not implemented.", min.ShortDebugString().c_str(),
      max.ShortDebugString().c_str(), operand.ShortDebugString().c_str());
}

// TODO(b/36794510): Make broadcast semantics more consistent, by supporting
// "degenerate" cases, as with binary elementwise ops, as well as scalar
// broadcast from all operands, not just the predicate.
/* static */ StatusOr<Shape> ShapeInference::InferSelectShape(
    const Shape& pred, const Shape& on_true, const Shape& on_false) {
  bool compatible;
  if (ShapeUtil::IsTuple(on_true)) {
    // Select only defines the top-level buffer, so if it's a tuple, the two
    // input must match exactly.
    compatible = ShapeUtil::Compatible(on_true, on_false);
  } else {
    compatible = ShapeUtil::CompatibleIgnoringFpPrecision(on_true, on_false);
  }
  if (!compatible) {
    return InvalidArgument(
        "Operands to select must be the same shape; got %s and %s.",
        ShapeUtil::HumanString(on_true).c_str(),
        ShapeUtil::HumanString(on_false).c_str());
  }
  if (pred.element_type() != PRED) {
    return InvalidArgument(
        "Select's pred operand must have PRED element type; got %s.",
        ShapeUtil::HumanString(pred).c_str());
  }
  if (ShapeUtil::CompatibleIgnoringElementType(pred, on_true) ||
      ShapeUtil::Rank(pred) == 0) {
    // By this stage we know that pred's element type is PRED. Therefore, this
    // check restricts pred to be a PRED scalar, or a PRED array with the same
    // dimensions as on_true and on_false.
    return ShapeUtil::ChangeElementType(
        on_true, ShapeUtil::HigherPrecisionElementType(on_true, on_false));
  } else {
    return InvalidArgument(
        "Select operation with non-scalar predicate with dimensionality "
        " different from the other operands: %s.",
        ShapeUtil::HumanString(pred).c_str());
  }
}

/* static */ StatusOr<Shape> ShapeInference::InferCallShape(
    tensorflow::gtl::ArraySlice<const Shape*> arg_shapes,
    const ProgramShape& to_apply) {
  // The applied function's arity equals the number of arguments.
  if (arg_shapes.size() != to_apply.parameters_size()) {
    string computation_signature = ShapeUtil::HumanString(to_apply);
    string argument_shapes =
        Join(arg_shapes, ", ", [](string* out, const Shape* shape) {
          tensorflow::strings::StrAppend(out, ShapeUtil::HumanString(*shape));
        });
    return InvalidArgument(
        "Call applied function arity must match number of arguments; got: "
        "arity: %d, arguments: %zu; computation signature: %s; argument "
        "shapes: [%s].",
        to_apply.parameters_size(), arg_shapes.size(),
        computation_signature.c_str(), argument_shapes.c_str());
  }

  // All arguments must be compatible with the program shape.
  for (int i = 0; i < arg_shapes.size(); ++i) {
    const Shape& arg_shape = *arg_shapes[i];
    const Shape& param_shape = to_apply.parameters(i);
    if (!ShapeUtil::Compatible(arg_shape, param_shape)) {
      return InvalidArgument(
          "Call parameter must match argument; got parameter %d shape: %s, "
          "argument shape: %s.",
          i, ShapeUtil::HumanString(param_shape).c_str(),
          ShapeUtil::HumanString(arg_shape).c_str());
    }
  }

  return to_apply.result();
}

static Status ValidateGatherDimensionNumbers(
    const Shape& input_shape,
    tensorflow::gtl::ArraySlice<int64> gather_indices_shape,
    const GatherDimensionNumbers& dim_numbers) {
  if (!c_is_sorted(dim_numbers.output_window_dims())) {
    return InvalidArgument(
        "Output window dimensions in gather op must be ascending; got: %s.",
        Join(dim_numbers.output_window_dims(), ", ").c_str());
  }

  if (c_adjacent_find(dim_numbers.output_window_dims()) !=
      dim_numbers.output_window_dims().end()) {
    return InvalidArgument(
        "Output window dimensions in gather op must not repeat; got: %s.",
        Join(dim_numbers.output_window_dims(), ", ").c_str());
  }

  const int64 output_window_dim_count = dim_numbers.output_window_dims_size();
  const int64 output_shape_rank =
      output_window_dim_count + gather_indices_shape.size() - 1;

  for (int i = 0; i < dim_numbers.output_window_dims_size(); ++i) {
    int64 window_index = dim_numbers.output_window_dims(i);
    if (window_index < 0 || window_index >= output_shape_rank) {
      return InvalidArgument(
          "Window index %d in gather op is out of bounds; got %lld, but should "
          "have been in [0,%lld).",
          i, window_index, output_shape_rank);
    }
  }

  if (dim_numbers.gather_dims_to_operand_dims_size() !=
      gather_indices_shape[dim_numbers.index_vector_dim()]) {
    return InvalidArgument(
        "Gather op has %d elements in gather_dims_to_operand_dims and the "
        "bound of dimension index_vector_dim=%lld of gather_indices is "
        "%lld. These two numbers must be equal.",
        dim_numbers.gather_dims_to_operand_dims_size(),
        dim_numbers.index_vector_dim(),
        gather_indices_shape[dim_numbers.index_vector_dim()]);
  }

  for (int i = 0; i < dim_numbers.gather_dims_to_operand_dims_size(); i++) {
    int64 gather_dim_to_input_dim = dim_numbers.gather_dims_to_operand_dims(i);
    if (gather_dim_to_input_dim < 0 ||
        gather_dim_to_input_dim >= input_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid gather_dims_to_operand_dims mapping; domain is [0, %d), "
          "got: %d->%lld.",
          input_shape.dimensions_size(), i, gather_dim_to_input_dim);
    }
  }

  std::vector<int64> sorted_gather_dims_to_operand_dims(
      dim_numbers.gather_dims_to_operand_dims().begin(),
      dim_numbers.gather_dims_to_operand_dims().end());

  c_sort(sorted_gather_dims_to_operand_dims);

  if (c_adjacent_find(sorted_gather_dims_to_operand_dims) !=
      sorted_gather_dims_to_operand_dims.end()) {
    return InvalidArgument(
        "Repeated dimensions are not allowed in gather_dims_to_operand_dims; "
        "got: %s.",
        Join(dim_numbers.gather_dims_to_operand_dims(), ", ").c_str());
  }

  for (int64 elided_dim : dim_numbers.elided_window_dims()) {
    if (elided_dim < 0 || elided_dim >= input_shape.dimensions_size()) {
      return InvalidArgument(
          "Invalid elided_window_dims set in gather op; valid range is [0, "
          "%d), got: %lld.",
          input_shape.dimensions_size(), elided_dim);
    }
  }

  if (!c_is_sorted(dim_numbers.elided_window_dims())) {
    return InvalidArgument(
        "elided_window_dims in gather op must be sorted; got: %s",
        Join(dim_numbers.elided_window_dims(), ", ").c_str());
  }

  if (c_adjacent_find(dim_numbers.elided_window_dims()) !=
      dim_numbers.elided_window_dims().end()) {
    return InvalidArgument(
        "Repeated dimensions not allowed in elided_window_dims in gather op; "
        "got: %s.",
        Join(dim_numbers.elided_window_dims(), ", ").c_str());
  }

  return Status::OK();
}

/*static*/ StatusOr<Shape> ShapeInference::InferGatherShape(
    const Shape& input_shape, const Shape& gather_indices_shape,
    const GatherDimensionNumbers& gather_dim_numbers,
    tensorflow::gtl::ArraySlice<int64> window_bounds) {
  TF_RETURN_IF_ERROR(
      ExpectNotTupleOrOpaque(input_shape, "input tensor operand gather op"));
  TF_RETURN_IF_ERROR(ExpectNotTupleOrOpaque(
      gather_indices_shape, "gather indices operand of gather op"));

  if (!ShapeUtil::ElementIsIntegral(gather_indices_shape)) {
    return InvalidArgument(
        "Gather indices parameter must be an integral tensor; got %s.",
        ShapeUtil::HumanString(gather_indices_shape).c_str());
  }

  // We implicitly reshape gather indices of shape P[A,B,C] to P[A,B,C,1] if
  // index_vector_dim is rank(P).  The bounds of this expanded shape is
  // stored in expanded_gather_indices_shape.

  if (gather_indices_shape.dimensions_size() <
          gather_dim_numbers.index_vector_dim() ||
      gather_dim_numbers.index_vector_dim() < 0) {
    return InvalidArgument(
        "Gather index leaf dimension must be within [0, rank(gather_indices) + "
        "1). rank(gather_indices) is %d and gather index leaf dimension is "
        "%lld.",
        gather_indices_shape.dimensions_size(),
        gather_dim_numbers.index_vector_dim());
  }

  std::vector<int64> expanded_gather_indices_shape;
  expanded_gather_indices_shape.reserve(gather_indices_shape.dimensions_size());
  c_copy(gather_indices_shape.dimensions(),
         std::back_inserter(expanded_gather_indices_shape));
  if (expanded_gather_indices_shape.size() ==
      gather_dim_numbers.index_vector_dim()) {
    expanded_gather_indices_shape.push_back(1);
  }

  TF_RETURN_IF_ERROR(ValidateGatherDimensionNumbers(
      input_shape, expanded_gather_indices_shape, gather_dim_numbers));

  if (window_bounds.size() != input_shape.dimensions_size()) {
    return InvalidArgument(
        "Gather op must have one window bound for every input dimension; got: "
        "len(window_bounds)=%lu, input_shape.rank=%d.",
        window_bounds.size(), input_shape.dimensions_size());
  }

  if (window_bounds.size() !=
      gather_dim_numbers.output_window_dims_size() +
          gather_dim_numbers.elided_window_dims_size()) {
    return InvalidArgument(
        "All components of the window index in a gather op must either be a "
        "output window index or explicitly elided; got len(window_bounds)=%lu, "
        "output_window_bounds=%s, elided_window_bounds=%s.",
        window_bounds.size(),
        Join(gather_dim_numbers.output_window_dims(), ",").c_str(),
        Join(gather_dim_numbers.elided_window_dims(), ",").c_str());
  }

  for (int i = 0; i < window_bounds.size(); i++) {
    int64 window_bound = window_bounds[i];
    int64 corresponding_input_bound = input_shape.dimensions(i);
    if (window_bound < 0 || window_bound > corresponding_input_bound) {
      return InvalidArgument(
          "Window bound at index %d in gather op is out of range, must be "
          "within "
          "[0, %lld), got %lld.",
          i, corresponding_input_bound + 1, window_bound);
    }
  }

  for (int i = 0; i < gather_dim_numbers.elided_window_dims_size(); i++) {
    if (window_bounds[gather_dim_numbers.elided_window_dims(i)] != 1) {
      return InvalidArgument(
          "Gather op can only elide window indices with bound 1, but bound is "
          "%lld for index %lld at position %d.",
          window_bounds[gather_dim_numbers.elided_window_dims(i)],
          gather_dim_numbers.elided_window_dims(i), i);
    }
  }

  int64 result_rank = gather_dim_numbers.output_window_dims_size() +
                      (expanded_gather_indices_shape.size() - 1);
  int64 window_dims_seen = 0;
  int64 gather_dims_seen = 0;
  std::vector<int64> output_dim_bounds;
  output_dim_bounds.reserve(result_rank);
  for (int64 i = 0; i < result_rank; i++) {
    int64 current_bound;
    bool is_window_index =
        c_binary_search(gather_dim_numbers.output_window_dims(), i);
    if (is_window_index) {
      while (c_binary_search(gather_dim_numbers.elided_window_dims(),
                             window_dims_seen)) {
        window_dims_seen++;
      }
      current_bound = window_bounds[window_dims_seen++];
    } else {
      if (gather_dims_seen == gather_dim_numbers.index_vector_dim()) {
        gather_dims_seen++;
      }
      current_bound = expanded_gather_indices_shape[gather_dims_seen++];
    }

    output_dim_bounds.push_back(current_bound);
  }

  return ShapeUtil::MakeShape(input_shape.element_type(), output_dim_bounds);
}

}  // namespace xla
