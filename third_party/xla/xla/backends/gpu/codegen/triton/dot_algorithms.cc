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

#include "xla/backends/gpu/codegen/triton/dot_algorithms.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/tensor_float_32_utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {
namespace triton {

namespace {

namespace arith = ::mlir::arith;
namespace math = ::mlir::math;
namespace ttir = ::mlir::triton;

using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;

Type ElementType(Value v) { return mlir::getElementTypeOrSelf(v); }

// Precision-relevant configuration bits for `dot`s.
struct PrecisionSpec {
  PrecisionConfig::Algorithm algorithm;
  // TODO(bchetioui): we hope to get rid of operand precisions eventually, they
  // are currently a (XLA-wide) bridge to work with ALG_UNSET.
  PrecisionConfig::Precision lhs_operand_precision;
  PrecisionConfig::Precision rhs_operand_precision;
  // Encodes `tt.dot`'s `inputPrecision` attribute.
  ttir::InputPrecision ttir_input_precision;
};

using AlgorithmEmitter = absl::StatusOr<Value> (*)(EmitterLocOpBuilder&,
                                                   const DotOperands&,
                                                   const PrecisionSpec&);

Value RoundToBF16(EmitterLocOpBuilder b, Value input) {
  return Cast(b, input, b.getBF16Type());
}

// Truncates |input| of F32 type to the number representable in Bf16 toward
// zero.
Value MaskToBF16(EmitterLocOpBuilder& b, Value input) {
  ShapedType input_type = mlir::dyn_cast<ShapedType>(input.getType());
  Type input_type_as_i32 = input_type.clone(b.getI32Type());
  Value input_as_i32 = b.create<ttir::BitcastOp>(input_type_as_i32, input);
  Value mask = triton::CreateConst<uint32_t>(b, b.getI32Type(), 0xFFFF0000u,
                                             input_type.getShape())
                   .UnwrapTensor();
  Value high_bits =
      b.create<arith::AndIOp>(input_type_as_i32, input_as_i32, mask);

  return b.create<ttir::BitcastOp>(input_type, high_bits);
}

// If lhs is 1.0, we will have lhs_high = 1.0 and lhs_low = 0.0.
// If rhs is +infinity, we will have:
// +infinity * 1.0 = +infinity
// +infinity * 0.0 = NaN
// We would get the wrong result if we sum these partial products. Instead, we
// must override any accumulated result if the last partial product is
// non-finite. See b/115844437.
Value ZeroNaNs(EmitterLocOpBuilder& b, Value input) {
  Value positive_inf =
      CreateConst<float>(b, b.getF32Type(),
                         std::numeric_limits<float>::infinity(),
                         mlir::cast<ShapedType>(input.getType()).getShape())
          .UnwrapTensor();
  Value abs_input = b.create<math::AbsFOp>(input);
  Value is_finite = b.create<arith::CmpFOp>(arith::CmpFPredicate::OGT,
                                            positive_inf, abs_input);
  return b.create<arith::SelectOp>(is_finite, input, ZerosLike(b, input));
}

absl::Status ExpectType(Value v, Type expected_type) {
  if (ElementType(v) != expected_type) {
    std::string expected_type_str, actual_type_str;
    {
      llvm::raw_string_ostream os_expected(expected_type_str);
      llvm::raw_string_ostream os_actual(actual_type_str);
      expected_type.print(os_expected);
      ElementType(v).print(os_actual);
    }
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected type ", expected_type_str, " but got ", actual_type_str));
  }
  return absl::OkStatus();
}

std::vector<Value> SplitF32(EmitterLocOpBuilder b, Value input,
                            int split_count) {
  std::vector<Value> split_inputs;
  split_inputs.reserve(split_count);
  for (int i = 0; i < split_count; ++i) {
    if (i != split_count - 1) {
      Value masked = MaskToBF16(b, input);
      input = b.create<arith::SubFOp>(input, masked);
      split_inputs.push_back(RoundToBF16(b, masked));
    } else {
      split_inputs.push_back(RoundToBF16(b, input));
    }
  }
  return split_inputs;
}

Value IEEEDot(EmitterLocOpBuilder b, Value lhs, Value rhs, Value acc) {
  return b.create<ttir::DotOp>(lhs, rhs, acc,
                               /*inputPrecision=*/ttir::InputPrecision::IEEE,
                               /*maxNumImpreciseAcc=*/0);
}

// Leverages BF16 datatype for F32 matmul computation. It follows the guidance
// from https://arxiv.org/pdf/1904.06376.pdf.
absl::StatusOr<Value> EmitBF16x9Matmul(EmitterLocOpBuilder& b,
                                       const DotOperands& dot_operands,
                                       const PrecisionSpec& precision_spec) {
  Type f32 = b.getF32Type();
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.lhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.rhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.accumulator, f32));

  std::vector<Value> lhs_parts = SplitF32(b, dot_operands.lhs, 3);
  std::vector<Value> rhs_parts = SplitF32(b, dot_operands.rhs, 3);

  Value local_acc = triton::ZerosLike(b, dot_operands.accumulator);
  Value result;

  // low @ low + low @ mid + mid @ low
  result = IEEEDot(b, lhs_parts[2], rhs_parts[2], local_acc);
  result = IEEEDot(b, lhs_parts[1], rhs_parts[2], result);
  result = IEEEDot(b, lhs_parts[2], rhs_parts[1], result);

  // mid @ mid
  result = IEEEDot(b, lhs_parts[1], rhs_parts[1], result);

  // high @ low + low @ high
  result = IEEEDot(b, lhs_parts[2], rhs_parts[0], result);
  result = IEEEDot(b, lhs_parts[0], rhs_parts[2], result);

  // high @ mid + mid @ high
  result = IEEEDot(b, lhs_parts[1], rhs_parts[0], result);
  result = IEEEDot(b, lhs_parts[0], rhs_parts[1], result);

  result = ZeroNaNs(b, result);
  result = IEEEDot(b, lhs_parts[0], rhs_parts[0], result);
  result = b.create<arith::AddFOp>(dot_operands.accumulator, result);
  return result;
}

// Leverages BF16 datatype for F32 matmul computation. It follows the guidance
// from https://arxiv.org/pdf/1904.06376.pdf.
absl::StatusOr<Value> EmitBF16x6Matmul(EmitterLocOpBuilder& b,
                                       const DotOperands& dot_operands,
                                       const PrecisionSpec& precision_spec) {
  Type f32 = b.getF32Type();
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.lhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.rhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.accumulator, f32));

  std::vector<Value> lhs_parts = SplitF32(b, dot_operands.lhs, 3);
  std::vector<Value> rhs_parts = SplitF32(b, dot_operands.rhs, 3);

  Value local_acc = triton::ZerosLike(b, dot_operands.accumulator);
  Value result = IEEEDot(b, lhs_parts[1], rhs_parts[1], local_acc);
  // high @ low + low @ high
  result = IEEEDot(b, lhs_parts[2], rhs_parts[0], result);
  result = IEEEDot(b, lhs_parts[0], rhs_parts[2], result);

  // high @ mid + mid @ high
  result = IEEEDot(b, lhs_parts[1], rhs_parts[0], result);
  result = IEEEDot(b, lhs_parts[0], rhs_parts[1], result);

  result = ZeroNaNs(b, result);
  result = IEEEDot(b, lhs_parts[0], rhs_parts[0], result);
  result = b.create<arith::AddFOp>(dot_operands.accumulator, result);
  return result;
}

// Compute F32 matmul with 3 BF16 dots. It is less accurate than
// EmitBF16x6Matmul.
absl::StatusOr<Value> EmitBF16x3Matmul(EmitterLocOpBuilder& b,
                                       const DotOperands& dot_operands,
                                       const PrecisionSpec& precision_spec) {
  Type f32 = b.getF32Type();
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.lhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.rhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.accumulator, f32));

  std::vector<Value> lhs_bf16 = SplitF32(b, dot_operands.lhs, 2);
  std::vector<Value> rhs_bf16 = SplitF32(b, dot_operands.rhs, 2);

  Value local_acc = triton::ZerosLike(b, dot_operands.accumulator);
  Value result = IEEEDot(b, lhs_bf16[1], rhs_bf16[0], local_acc);
  result = IEEEDot(b, lhs_bf16[0], rhs_bf16[1], result);
  result = ZeroNaNs(b, result);
  result = IEEEDot(b, lhs_bf16[0], rhs_bf16[0], result);
  result = b.create<arith::AddFOp>(dot_operands.accumulator, result);
  return result;
}

bool IsTf32Allowed(const HloDotInstruction& dot) {
  auto precision_config = dot.precision_config();
  if (precision_config.algorithm() == PrecisionConfig::ALG_UNSET) {
    return tsl::tensor_float_32_execution_enabled() &&
           precision_config.operand_precision(0) == PrecisionConfig::DEFAULT &&
           precision_config.operand_precision(1) == PrecisionConfig::DEFAULT;
  }
  return algorithm_util::HasTf32InputType(precision_config.algorithm());
}

bool DotDependsOnConvertFromByteWideOrSmallerTypeToF32(
    const HloDotInstruction* dot) {
  return HloBfsAnyOf({dot}, [&](const HloInstruction* node) {
    if (node->opcode() != HloOpcode::kConvert) {
      return false;
    }
    int in_width =
        primitive_util::BitWidth(node->operand(0)->shape().element_type());
    return in_width <= 8 && node->shape().element_type() == F32;
  });
}

ttir::InputPrecision InferDotPrecision(const HloDotInstruction& dot) {
  if (dot.precision_config().algorithm() ==
      PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3) {
    return ttir::InputPrecision::TF32x3;
  }

  bool use_tf32 = IsTf32Allowed(dot);
  // TODO(b/320659359) Allow TF32 for 8-bit or less types with F32.
  use_tf32 =
      use_tf32 && !DotDependsOnConvertFromByteWideOrSmallerTypeToF32(&dot);

  return use_tf32 ? ttir::InputPrecision::TF32 : ttir::InputPrecision::IEEE;
}

Type GetAlgUnsetAccumulatorType(EmitterLocOpBuilder& b,
                                const DotOperands& dot_operands) {
  Type lhs_type = ElementType(dot_operands.lhs);
  Type rhs_type = ElementType(dot_operands.rhs);
  Type accumulator_type = ElementType(dot_operands.accumulator);

  // The code below assumes that lhs and rhs have the same type. However
  // this may not always be the case with f8 matmuls, e.g. e4m3×e5m2 is
  // supported at the hardware level. NVIDIA GPUs currently only support f32
  // accumulators for such matmuls.
  if (lhs_type.isFloat(8) && rhs_type.isFloat(8)) {
    return b.getF32Type();
  }

  CHECK(lhs_type == rhs_type);

  // Currently allowing 8x8-bit ints -> i32.
  if (lhs_type == b.getIntegerType(8) && accumulator_type.isInteger(32)) {
    return b.getI32Type();
  }
  return (accumulator_type.isF64() && lhs_type.isF64()) ? b.getF64Type()
                                                        : b.getF32Type();
}

absl::StatusOr<Value> EmitDotAlgUnset(EmitterLocOpBuilder& b,
                                      const DotOperands& dot_operands,
                                      const PrecisionSpec& precision_spec) {
  // Execute matrix multiplication of input tiles and pass the accumulator.
  // TODO(manany): Should be looked into once we enable Hopper workloads.
  // maxNumImpreciseAcc flag was introduced for Hopper to accumulate in a
  // lower precision than the output type. The change was introduced here:
  // https://github.com/openai/triton/commit/31b0c521427109a8eda609b58d756c380b21599a
  Value lhs = dot_operands.lhs;
  Value rhs = dot_operands.rhs;
  Value acc = dot_operands.accumulator;

  Type expected_acc_type = GetAlgUnsetAccumulatorType(b, dot_operands);
  if (ElementType(acc) != expected_acc_type) {
    return absl::FailedPreconditionError(
        "Given accumulator type for unset dot does not match expected type.");
  }

  int max_num_imprecise_acc = 0;
  if (ElementType(lhs).isFloat(8) || ElementType(rhs).isFloat(8)) {
    // For fp8 dots, disable accumulator promotion to mimick cuBLAS. It may make
    // sense to enable frequent accumulator promotion at higher matmul
    // precisions set in the config.
    max_num_imprecise_acc = std::numeric_limits<int>::max();
  }

  return b.create<ttir::DotOp>(
      lhs, rhs, acc,
      /*inputPrecision=*/precision_spec.ttir_input_precision,
      /*maxNumImpreciseAcc=*/max_num_imprecise_acc);
}

absl::StatusOr<Value> EmitRegularDot(EmitterLocOpBuilder& b,
                                     const DotOperands& dot_operands,
                                     const PrecisionSpec& precision_spec) {
  Value lhs = dot_operands.lhs;
  Value rhs = dot_operands.rhs;

  int max_num_imprecise_acc = 0;
  if (ElementType(lhs).isFloat(8) || ElementType(rhs).isFloat(8)) {
    // For fp8 dots, disable accumulator promotion to mimick cuBLAS. It may make
    // sense to enable frequent accumulator promotion at higher matmul
    // precisions set in the config.
    max_num_imprecise_acc = std::numeric_limits<int>::max();
  }

  // Cast F32 inputs to BF16 if the algorithm is BF16_BF16_F32.
  // TODO(bchetioui): abstract this.
  if (precision_spec.algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_F32) {
    if (ElementType(lhs).isF32()) {
      lhs = Cast(b, lhs, b.getBF16Type());
    }

    if (ElementType(rhs).isF32()) {
      rhs = Cast(b, rhs, b.getBF16Type());
    }
  }

  return b.create<ttir::DotOp>(
      dot_operands.lhs, dot_operands.rhs, dot_operands.accumulator,
      /*inputPrecision=*/precision_spec.ttir_input_precision,
      /*maxNumImpreciseAcc=*/max_num_imprecise_acc);
}

}  // namespace

absl::StatusOr<Value> EmitSingleTileDot(EmitterLocOpBuilder& b,
                                        const HloDotInstruction& dot,
                                        DotOperands dot_operands) {
  AlgorithmEmitter algorithm_emitter = nullptr;
  PrecisionSpec precision_spec{dot.precision_config().algorithm(),
                               dot.precision_config().operand_precision(0),
                               dot.precision_config().operand_precision(1),
                               InferDotPrecision(dot)};

  // Algorithms mostly expect that their input and output types correspond to
  // what the algorithm describes. This is not always the case though, e.g.
  // for BF16_BF16_F32_X9, working from inputs casted to BF16 makes no sense;
  // this algorithm instead expects F32 inputs, and performs splits into BF16
  // sub-values under the hood.
  std::optional<Type> force_operands_type;
  std::optional<Type> force_accumulator_type;

  PrecisionConfig::Algorithm algorithm = precision_spec.algorithm;

  Type bf16 = b.getBF16Type();
  Type f16 = b.getF16Type();
  Type f32 = b.getF32Type();
  Type f64 = b.getF64Type();

  switch (algorithm) {
    case PrecisionConfig::ALG_UNSET:
      algorithm_emitter = EmitDotAlgUnset;
      break;
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
      force_operands_type = f16;
      force_accumulator_type = f16;
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      force_operands_type = f32;
      force_accumulator_type = f32;
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      force_operands_type = f64;
      force_accumulator_type = f64;
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
      force_operands_type = f16;
      force_accumulator_type = f32;
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
      force_operands_type = bf16;
      force_accumulator_type = bf16;
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      force_operands_type = bf16;
      force_accumulator_type = f32;
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      force_operands_type = f32;  // This is not a typo.
      force_accumulator_type = f32;
      algorithm_emitter = EmitBF16x3Matmul;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      force_operands_type = f32;  // This is not a typo.
      force_accumulator_type = f32;
      algorithm_emitter = EmitBF16x6Matmul;
      break;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      // TODO(bchetioui): pass around tf32 matmul config.
      force_operands_type = f32;
      force_accumulator_type = f32;
      // TODO(bchetioui): this should be factored out of EmitRegularDot.
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      // TODO(bchetioui): pass around tf32 matmul config.
      force_operands_type = f32;
      force_accumulator_type = f32;
      // TODO(bchetioui): this should be factored out of EmitRegularDot.
      algorithm_emitter = EmitRegularDot;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      force_operands_type = f32;  // This is not a typo.
      force_accumulator_type = f32;
      algorithm_emitter = EmitBF16x9Matmul;
      break;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
      // TODO(bchetioui): How to enforce "any f8"?
      force_accumulator_type = f32;
      break;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      // TODO(bchetioui): How to enforce "any f8"?
      force_accumulator_type = f32;
      break;
    default:
      break;
  }

  // Couldn't find an algorithm emitter for this algorithm. Raise an error.
  if (algorithm_emitter == nullptr) {
    return absl::UnimplementedError(
        absl::StrCat("This algorithm is not supported yet: ",
                     PrecisionConfig::Algorithm_Name(algorithm)));
  }

  if (force_operands_type.has_value()) {
    if (ElementType(dot_operands.lhs) != *force_operands_type) {
      dot_operands.lhs = Cast(b, dot_operands.lhs, *force_operands_type);
    }

    if (ElementType(dot_operands.rhs) != *force_operands_type) {
      dot_operands.rhs = Cast(b, dot_operands.rhs, *force_operands_type);
    }
  }

  if (force_accumulator_type.has_value()) {
    if (ElementType(dot_operands.accumulator) != *force_accumulator_type) {
      dot_operands.accumulator =
          Cast(b, dot_operands.accumulator, *force_accumulator_type);
    }
  }

  TF_ASSIGN_OR_RETURN(Value result,
                      algorithm_emitter(b, dot_operands, precision_spec));

  // TODO(b/393299275): once we've moved on from the legacy emitter, we should
  // make sure that this accumulator type is equal to the one derived here.
  Type outer_accumulator_type = ElementType(dot_operands.accumulator);
  if (ElementType(result) != outer_accumulator_type) {
    result = Cast(b, result, outer_accumulator_type);
  }

  return result;
}

}  // namespace triton
}  // namespace gpu
}  // namespace xla
