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

#ifdef FXPMATH_OPS
#else

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/Quantization/QuantPredicates.td"

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

def fxpmath_CompareZ    : EnumAttrCase<"CMPZ">;
def fxpmath_CompareNZ   : EnumAttrCase<"CMPNZ">;
def fxpmath_CompareLZ   : EnumAttrCase<"CMPLZ">;
def fxpmath_CompareLZE  : EnumAttrCase<"CMPLZE">;
def fxpmath_CompareGZ   : EnumAttrCase<"CMPGZ">;
def fxpmath_CompareGZE  : EnumAttrCase<"CMPGZE">;

def fxpmath_CompareFnAttr : EnumAttr<"ComparisonFn",
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
    Op<!strconcat("fxpmath.", mnemonic), traits>;

//===----------------------------------------------------------------------===//
// Fixed-point (fxp) arithmetic ops used by kernels.
//===----------------------------------------------------------------------===//

def fxpmath_RoundingDivideByPotFxpOp :
    fxpmath_Op<"rounding_divide_by_poti", [NoSideEffect, SameValueType]>,
    Arguments<(ins quant_StorageValueType:$x, I32Attr:$exponent)>,
    Results<(outs quant_StorageValueType:$y)> {
  let description = [{
    Computes integer division by a power-of-two, correctly rounded-to-nearest.
    Also known as a rounding arithmetic right shift. See
    gemmlowp::RoundingDivideByPOT for a reference implementation.
  }];

  let verifier = [{
    auto verifyExponent = exponent().getSExtValue();
    if (verifyExponent < 0 || verifyExponent > 31) {
      return emitOpError("exponent must be in range [0..31]");
    }
    return success();
  }];
}

def fxpmath_SaturatingAddFxpOp :
    fxpmath_Op<"saturating_addi", [NoSideEffect, SameValueType]>,
    Arguments<(ins quant_StorageValueType:$x,
                   quant_StorageValueType:$y,
                   I32Attr:$clamp_min,
                   I32Attr:$clamp_max)>,
    Results<(outs quant_StorageValueType:$sum)> {
  let description = [{
    Computes saturating addition of two operands, saturating to the given min
    and max value. The implementation is responsible for choosing an
    intermediate register size appropriate to carry out the operation without
    overflow. See gemmlowp::SaturatingAdd for a reference implementation.
  }];
}

//===----------------------------------------------------------------------===//
// Real math ops.
//
// Math ops on real numbers which may have a representation in quantized
// arithmetic. It is expected that eligible ops are lowered from a source
// dialect to this set of ops prior to the process of converting a compuation
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

// The broadcasting dimensions correspond to a tuple that describes how a
// smaller rank shape is broadcast into a larger rank shape. For example,
// given a 2x3x4 cuboid and a 3x4 matrix, a broadcasting tuple (1,2) means
// matching the matrix to dimensions 1 and 2 of the cuboid.
def fxpmath_BroadcastDimAttr : OptionalAttr<ElementsAttr>;

class fxpmath_RealBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    fxpmath_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$x,
                      quant_RealValueType:$y,
                      fxpmath_BroadcastDimAttr:$broadcast_dimensions
                     )>,
    Results<(outs quant_RealValueType:$r)>;

class fxpmath_RealBinaryBiasOp<string mnemonic, list<OpTrait> traits = []> :
    fxpmath_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$x, quant_RealValueType:$y,
                          quant_RealValueType:$bias)>,
    Results<(outs quant_RealValueType:$r)>;

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
        (ins quant_RealValueType:$x, fxpmath_EwUnaryFnAttr:$fn)>,
    Results<(outs quant_RealValueType:$r)>;

def fxpmath_RealCompareZeroEwOp : fxpmath_Op<"compare", [NoSideEffect]>,
    Arguments<(ins quant_RealValueType:$x, fxpmath_CompareFnAttr:$fn)>,
    Results<(outs I1Tensor:$r)> {
  let description = [{
    Compares a real value to zero, returning an I1 (boolean) tensor with the
    result of applying the comparison function.
  }];
}

#endif  // FXPMATH_OPS
