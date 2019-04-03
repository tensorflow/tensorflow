//===- QuantOps.td - Quantization operation definition -----*- tablegen -*-===//
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
// This is the operation definition file for Quantization.
//
//===----------------------------------------------------------------------===//

#ifdef QUANTIZATION_OPS
#else

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Quantization type definitions
//===----------------------------------------------------------------------===//

class quant_TypedPrimitiveOrContainer<Type etype> :
    Type<AnyOf<[etype.predicate,
                TypedTensor<etype>.predicate,
                TypedVector<etype>.predicate]>,
         "primitive/tensor/vector of " # etype.description>;

// An implementation of QuantizedType.
def quant_QuantizedType :
    Type<CPred<"{0}.isa<QuantizedType>()">, "QuantizedType">;

// A primitive type that can represent a real value. This is either a
// floating point value or a quantized type.
def quant_RealPrimitiveType :
    Type<AnyOf<[Float.predicate, quant_QuantizedType.predicate]>,
    "real valued primitive (float or quantized type)">;

// A primitive type that can represent a storage value. This is either an
// integer or quantized type.
def quant_StoragePrimitiveType :
    Type<AnyOf<[Integer.predicate, quant_QuantizedType.predicate]>,
    "quantized storage primitive (integer or quantized type)">;

// A primitive or container of RealPrimitiveType.
def quant_RealValueType :
    quant_TypedPrimitiveOrContainer<quant_RealPrimitiveType>;

// A primitive or container of StoragePrimitiveType.
def quant_StorageValueType :
    quant_TypedPrimitiveOrContainer<quant_StoragePrimitiveType>;

// Either a real valued or storage primitive or container type.
def quant_RealOrStorageValueType :
    Type<AnyOf<[quant_RealValueType.predicate,
                quant_StorageValueType.predicate]>>;

// An implementation of UniformQuantizedType.
def quant_UniformQuantizedType :
    Type<CPred<"{0}.isa<UniformQuantizedType>()">, "UniformQuantizedType">;

// Predicate for detecting a container or primitive of UniformQuantizedType.
def quant_UniformQuantizedValueType :
    quant_TypedPrimitiveOrContainer<quant_UniformQuantizedType>;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// Real value for an (inclusive) min/max clamp limit.
def quant_ClampValueAttr : OptionalAttr<F64Attr>;

// Element-wise activation function to apply.
// Note that RELU activations are not here: they are expressed as clamps.
def quant_EwUnaryFnAttr :
    StringBasedAttr<CPred<"true">, "element-wise unary function"> {
  let returnType = [{ StringRef }];
  let defaultValue = "IDENTITY";
}

class quant_ConstEwUnaryFn<string val> : ConstantAttr<quant_EwUnaryFnAttr, val>;
def quant_EwUnaryFn_Identity: quant_ConstEwUnaryFn<"IDENTITY">;
def quant_EwUnaryFn_Tanh    : quant_ConstEwUnaryFn<"TANH">;
def quant_EwUnaryFn_Sigmoid : quant_ConstEwUnaryFn<"SIGMOID">;
def quant_EwUnaryFn_Exp     : quant_ConstEwUnaryFn<"EXP">;
def quant_EwUnaryFn_Log     : quant_ConstEwUnaryFn<"LOG">;
def quant_EwUnaryFn_Neg     : quant_ConstEwUnaryFn<"NEG">;
def quant_EwUnaryFn_Rsqrt   : quant_ConstEwUnaryFn<"RSQRT">;
def quant_EwUnaryFn_Sin     : quant_ConstEwUnaryFn<"SIN">;
def quant_EwUnaryFn_Square  : quant_ConstEwUnaryFn<"SQUARE">;
def quant_EwUnaryFn_Sqrt    : quant_ConstEwUnaryFn<"SQRT">;
def quant_EwUnaryFn_CmpZ    : quant_ConstEwUnaryFn<"CMPZ">;
def quant_EwUnaryFn_CmpNZ   : quant_ConstEwUnaryFn<"CMPNZ">;
def quant_EwUnaryFn_CmpLZ   : quant_ConstEwUnaryFn<"CMPLZ">;
def quant_EwUnaryFn_CmpGZ   : quant_ConstEwUnaryFn<"CMPGZ">;

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class quant_Op<string mnemonic, list<OpTrait> traits> :
    Op<!strconcat("quant.", mnemonic), traits>;

//===----------------------------------------------------------------------===//
// Quantization barriers
//===----------------------------------------------------------------------===//
class quant_BarrierOp<string mnemonic, list<OpTrait> traits> :
    quant_Op<mnemonic, traits>, Arguments<(ins quant_RealValueType:$arg)>,
    Results<(outs quant_RealValueType)>;

// A QuantizeBarrier (qbarrier) represents a potential type shift from a
// quantizable type to a quantized type.
//
// At runtime, a qbarrier will apply the transformation expressed by its
// operand and result type. For flexibility during transformation, it is also
// possible to have a qbarrier that performs no transformation (both its
// operand and result type are quantizable).
//
// A qbarrier will typically originate from either:
//   a) An expressed or implied constraint in the source dialect which signals
//      that a certain level of quantization is possible or required.
//   b) An inference made by a quantization algorithm indicating that a
//      quantized representation may be acceptable.
//
// Especially early in transformation, it is common to have pairs of
// qbarrier/dbarrier at points where a transition to a quantized type is
// required. In addition, it is also common to have an identity qbarrier
// (where the operand and result type are not quantized) at all points where
// it is legal to use a quantized representation (but is not known to be
// acceptable).
def quant_QuantizeBarrierOp : quant_BarrierOp<"qbarrier", [NoSideEffect]>;

// A DequantizeBarrier (dbarrier) represents the inverse of a qbarrier,
// converting back from a quantized to quantizable (expressed) type.
//
// Like qbarriers, a dbarrier is allowed to have both its operand and result
// as non quantized types. This facilitates transformations and marks edges
// where the computation must be carried out in the expressed type.
//
// Especially early in transformation, it is common to have dbarriers on
// all operands to ops that must operate with the expressed type (typically
// math ops prior to lowering to target-specific, quantized kernels).
def quant_DequantizeBarrierOp : quant_BarrierOp<"dbarrier", [NoSideEffect]>;

// A StorageCast (scast) represents a cast from or to a type based on the
// storage type and a type based on a corresponding quantized type.
//
// This op exists to ensure type coherency for between parts of the computation
// which are operating directly on an underlying storage type and those which
// operate on quantized values.
//
// Examples from storage to quantized type:
//   i8 -> !quant<"uniform[i8:f32]{1.0}">
//   tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
//   vector<4xi8> -> vector<4x!quant<"uniform[i8:f32]{1.0}">>
def quant_StorageCastOp :
    quant_Op<"scast", [NoSideEffect]>,
    Arguments<(ins quant_RealOrStorageValueType:$arg)>,
    Results<(outs quant_RealOrStorageValueType)>;

//===----------------------------------------------------------------------===//
// Integral arithmetic ops used by kernels.
//===----------------------------------------------------------------------===//

def quant_RoundingDivideByPotIOp :
    quant_Op<"rounding_divide_by_poti", [NoSideEffect, SameValueType]>,
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

def quant_SaturatingAddIOp :
    quant_Op<"saturating_addi", [NoSideEffect, SameValueType]>,
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
// quant_RealUnaryEwOp and selected via an enum-like attribute. Especially at
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

class quant_RealMathOp<string mnemonic, list<OpTrait> traits = [], dag args> :
    quant_Op<mnemonic, traits>,
    Arguments<!con(args, (ins
        quant_ClampValueAttr:$clamp_min, quant_ClampValueAttr:$clamp_max))>;

//===----------------------------------------------------------------------===//
// Element wise binary real math ops.
//===----------------------------------------------------------------------===//

class quant_RealBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    quant_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$x, quant_RealValueType:$y)>,
    Results<(outs quant_RealValueType:$r)>;

class quant_RealBinaryBiasOp<string mnemonic, list<OpTrait> traits = []> :
    quant_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$x, quant_RealValueType:$y,
                          quant_RealValueType:$bias)>,
    Results<(outs quant_RealValueType:$r)>;

def quant_RealAddEwOp :
    quant_RealBinaryOp<"real_add_ew", [NoSideEffect]>;

def quant_RealSubEwOp :
    quant_RealBinaryOp<"real_sub_ew", [NoSideEffect]>;

def quant_RealMulEwOp :
    quant_RealBinaryOp<"real_mul_ew", [NoSideEffect]>;

def quant_RealDivEwOp :
    quant_RealBinaryOp<"real_div_ew", [NoSideEffect]>;

//===----------------------------------------------------------------------===//
// Element wise unary real math op.
//===----------------------------------------------------------------------===//

def quant_RealUnaryEwOp :
    quant_RealMathOp<"real_unary_ew", [NoSideEffect],
        (ins quant_RealValueType:$x, quant_EwUnaryFnAttr:$fn)>,
    Results<(outs quant_RealValueType:$r)>;

#endif  // QUANTIZATION_OPS
