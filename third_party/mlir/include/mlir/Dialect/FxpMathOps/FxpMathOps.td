//===- FxpMathOps.td - Fixed point ops  --------------------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This is the operation definition file for fixed point ops (and real
// equivalents).
//
//===----------------------------------------------------------------------===//

#ifdef DIALECT_FXPMATHOPS_FXPMATH_OPS_
#else
#define DIALECT_FXPMATHOPS_FXPMATH_OPS_
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/Dialect/QuantOps/QuantPredicates.td"

def fxpmath_Dialect : Dialect {
  let name = "fxpmath";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// Real value for an (inclusive) min/max clamp limit.
def fxpmath_ClampValueAttr : OptionalAttr<F64Attr>;

// Element-wise activation function to apply.
// Note that RELU activations are not here: they are expressed as clamps.
def fxpmath_EwUnaryFnAttr :
    StringBasedAttr<CPred<"true">, "element-wise unary function"> {
  let returnType = [{ StringRef }];
  let defaultValue = "IDENTITY";
}

class fxpmath_ConstEwUnaryFn<string val> : ConstantAttr<fxpmath_EwUnaryFnAttr, val>;
def fxpmath_EwUnaryFn_Abs     : fxpmath_ConstEwUnaryFn<"ABS">;
def fxpmath_EwUnaryFn_Exp     : fxpmath_ConstEwUnaryFn<"EXP">;
def fxpmath_EwUnaryFn_Identity: fxpmath_ConstEwUnaryFn<"IDENTITY">;
def fxpmath_EwUnaryFn_Log     : fxpmath_ConstEwUnaryFn<"LOG">;
def fxpmath_EwUnaryFn_Neg     : fxpmath_ConstEwUnaryFn<"NEG">;
def fxpmath_EwUnaryFn_Rsqrt   : fxpmath_ConstEwUnaryFn<"RSQRT">;
def fxpmath_EwUnaryFn_Sigmoid : fxpmath_ConstEwUnaryFn<"SIGMOID">;
def fxpmath_EwUnaryFn_Sign    : fxpmath_ConstEwUnaryFn<"SIGN">;
def fxpmath_EwUnaryFn_Sin     : fxpmath_ConstEwUnaryFn<"SIN">;
def fxpmath_EwUnaryFn_Sqrt    : fxpmath_ConstEwUnaryFn<"SQRT">;
def fxpmath_EwUnaryFn_Square  : fxpmath_ConstEwUnaryFn<"SQUARE">;
def fxpmath_EwUnaryFn_Tanh    : fxpmath_ConstEwUnaryFn<"TANH">;

//===----------------------------------------------------------------------===//
// Comparison functions (compares relative to zero on a subtraction result).
//===----------------------------------------------------------------------===//

def fxpmath_CompareZ    : StrEnumAttrCase<"CMPZ">;
def fxpmath_CompareNZ   : StrEnumAttrCase<"CMPNZ">;
def fxpmath_CompareLZ   : StrEnumAttrCase<"CMPLZ">;
def fxpmath_CompareLZE  : StrEnumAttrCase<"CMPLZE">;
def fxpmath_CompareGZ   : StrEnumAttrCase<"CMPGZ">;
def fxpmath_CompareGZE  : StrEnumAttrCase<"CMPGZE">;

def fxpmath_CompareFnAttr : StrEnumAttr<"ComparisonFn",
    "Type of subtraction-result comparison to perform.",
    [
      fxpmath_CompareZ,
      fxpmath_CompareNZ,
      fxpmath_CompareLZ,
      fxpmath_CompareLZE,
      fxpmath_CompareGZ,
      fxpmath_CompareGZE
    ]>;

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class fxpmath_Op<string mnemonic, list<OpTrait> traits> :
    Op<fxpmath_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Fixed-point (fxp) arithmetic ops used by kernels.
// Some of these are temporary pending inclusion into a more core dialect.
//===----------------------------------------------------------------------===//

def fxpmath_ClampISOp : fxpmath_Op<"clampis", [NoSideEffect, SameOperandsAndResultType]> {
  let summary =
      "Clamps a signed-integer like argument to a min/max range.";
  let description = [{
    Element-wise equivalent to:
      r = std::min(clamp_max, std::max(e, clamp_min))
  }];
  let arguments = (ins IntegerLike:$operand,
                       APIntAttr:$clamp_min,
                       APIntAttr:$clamp_max);
  let results = (outs IntegerLike);
}

def fxpmath_ConvertISOp :
    fxpmath_Op<"convertis",
               [NoSideEffect, SameOperandsAndResultShape]> {
  let summary =
      "Does an element-wise conversion from a signed integer to signed integer";
  let description = [{
    Similar to an element-wise static_cast in C++, from a one signed integer
    element type to another.
  }];
  let arguments = (ins IntegerLike:$operand);
  let results = (outs IntegerLike);
}

def fxpmath_ConvertISToFOp :
    fxpmath_Op<"convertistof",
               [NoSideEffect, SameOperandsAndResultShape]> {
  let summary =
      "Does an element-wise conversion from a signed integer to a float";
  let description = [{
    Similar to an element-wise static_cast in C++, from a signed integer
    element type to a floating point element type, rounding to the nearest
    floating point value.
  }];
  let arguments = (ins IntegerLike:$operand);
  let results = (outs FloatLike);
}


def fxpmath_VecScalarSaturatingRoundingDoublingHighMulISOp :
    fxpmath_Op<"vs_saturating_rounding_doubling_high_mulis",
               [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Implements equivalent functionality to ARMv7 NEON VQRDMULH";
  let description = [{
    Equivalent to the ARMv7 NEON VQRDMULH instruction.
    See gemmlowp::SaturatingRoundingDoublingHighMul for a reference
    implementation.
  }];
  let arguments = (ins IntegerLike:$a, APIntAttr:$b);
  let results = (outs IntegerLike);
}

def fxpmath_RoundingDivideByPotISOp :
    fxpmath_Op<"rounding_divide_by_potis", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
    Computes a rounding arithmetic right shift.
  }];
  let description = [{
    Computes integer division by a power-of-two, correctly rounded-to-nearest.
    Also known as a rounding arithmetic right shift. See
    gemmlowp::RoundingDivideByPOT for a reference implementation.
  }];
  let arguments = (ins IntegerLike:$operand, APIntAttr:$exponent);
  let results = (outs IntegerLike:$res);
  let verifier = [{
    auto verifyExponent = exponent().getSExtValue();
    if (verifyExponent < 0 || verifyExponent > 31) {
      return emitOpError("exponent must be in range [0..31]");
    }
    return success();
  }];
}

//===----------------------------------------------------------------------===//
// Real math ops.
//
// Math ops on real numbers which may have a representation in quantized
// arithmetic. It is expected that eligible ops are lowered from a source
// dialect to this set of ops prior to the process of converting a computation
// to a quantized form. It is a non-goal of these ops to preserve enough
// information to convert back to the higher level, source dialect.
//
// These ops support either real/floating point or QuantizedTypes as operands
// and results. Since not all transformations are supported (globally or
// sometimes for specific targets), a computation may end up with
// untransformable RealMathOps, in which case they need to be lowered as is
// (using floating point math).
//
// This op set takes advantage of the fact that it is typically trivial to
// combine a math function with a compatible bias addition and real-valued
// clamp (which can be done at a higher accumulation bit depth).
//
// In addition, all element-wise unary functions are collapsed into a single
// fxpmath_RealUnaryEwOp and selected via an enum-like attribute. Especially at
// low bit depths, this makes matching simpler and allows the construction of
// generic LUT-based implementations. It also allows specific lowering rules
// to consolidate runs of chained unary ops and fuse them to preceding math
// ops, potentially allowing them to operate directly on higher precision
// intermediates without resorting to lots of custom kernels for common
// formulas that can suffer from insufficient precision at low bit depths.
//
// Comparison operators are modeled as element-wise unary functions (i.e.
// CMPZ, CMPNZ, CMPLZ, CMPGZ) intended to follow a sub and output a 1bit
// quantized value. It is expected that lowering rules can fuse them with
// the preceding sub.
//===----------------------------------------------------------------------===//

class fxpmath_RealMathOp<string mnemonic, list<OpTrait> traits = [], dag args> :
    fxpmath_Op<mnemonic, traits>,
    Arguments<!con(args, (ins
        fxpmath_ClampValueAttr:$clamp_min, fxpmath_ClampValueAttr:$clamp_max))>;

//===----------------------------------------------------------------------===//
// Element wise binary real math ops.
//===----------------------------------------------------------------------===//

class fxpmath_RealBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    fxpmath_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$lhs,
                      quant_RealValueType:$rhs)>,
    Results<(outs quant_RealValueType:$res)>;

class fxpmath_RealBinaryBiasOp<string mnemonic, list<OpTrait> traits = []> :
    fxpmath_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$lhs, quant_RealValueType:$rhs,
                          quant_RealValueType:$bias)>,
    Results<(outs quant_RealValueType:$res)>;

def fxpmath_RealAddEwOp :
    fxpmath_RealBinaryOp<"real_add_ew", [NoSideEffect]>;

def fxpmath_RealSubEwOp :
    fxpmath_RealBinaryOp<"real_sub_ew", [NoSideEffect]>;

def fxpmath_RealMulEwOp :
    fxpmath_RealBinaryOp<"real_mul_ew", [NoSideEffect]>;

def fxpmath_RealDivEwOp :
    fxpmath_RealBinaryOp<"real_div_ew", [NoSideEffect]>;

//===----------------------------------------------------------------------===//
// Element wise unary real math op.
//===----------------------------------------------------------------------===//

def fxpmath_RealUnaryEwOp :
    fxpmath_RealMathOp<"real_unary_ew", [NoSideEffect],
        (ins quant_RealValueType:$operand, fxpmath_EwUnaryFnAttr:$fn)>,
    Results<(outs quant_RealValueType:$res)>;

def fxpmath_RealCompareZeroEwOp : fxpmath_Op<"compare", [NoSideEffect]>,
    Arguments<(ins quant_RealValueType:$operand, fxpmath_CompareFnAttr:$fn)>,
    Results<(outs I1Tensor:$res)> {
  let description = [{
    Compares a real value to zero, returning an I1 (boolean) tensor with the
    result of applying the comparison function.
  }];
}

//===----------------------------------------------------------------------===//
// Dot op with fused bias addition.
//===----------------------------------------------------------------------===//

def fxpmath_RealMatMulOp :
    fxpmath_RealBinaryOp<"real_matmul", [NoSideEffect]> {
  let summary = "Matmul";
  let description = [{
    A matrix multiply of [m, k] and [k, n] -> [m, n] where the bias vector is
    of shape [n]. Also accepts rank 3 or more input tensors, in which case
    the leading dimensions are batch dims.

    Many real systems have specific library calls optimized for this precise
    operation, which is why it is handled explicitly versus purely as a
    generalized tensor contraction.
  }];
}

def fxpmath_RealMatMulBiasOp :
    fxpmath_RealBinaryBiasOp<"real_matmul_bias", [NoSideEffect]> {
  let summary = "Matmul with bias";
  let description = [{
    A specialization of a RealMatMulOp that also accepts an [n] dimension
    bias vector.

    In addition, there is often special support for a fused bias and clamp,
    which is why they are included.
  }];
}

#endif  // DIALECT_FXPMATHOPS_FXPMATH_OPS_
