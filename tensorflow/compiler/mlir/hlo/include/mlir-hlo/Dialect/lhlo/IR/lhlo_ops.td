/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation definition file for LMHLO, the "late" MHLO variant of
// the dialect, which operates on buffers instead of tensors.
//
// This file largely overlaps with hlo_ops.td at a logical level. It's tempting
// to merge these two files together, but we need to consider the following
// obstacles:
// * We need to have a common representation for arguments. That is to say,
//   HLO_Array<X> translates to HLO_Tensor<X> in HLO dialect, and
//   Arg<LHLO_Buffer<X>, "", [Mem(Read|Write)]> in LHLO. Array types within
//   tuples also need to be transformed.
// * As of now, TableGen's dag functions are not sufficient to accomplish the
//   one above.
// * Traits aren't identical, but need to be copied. For example,
//   SameOperandAndResultType in HLO corresponds to SameTypeOperands in LHLO.
// * Also, currently HLO describes the API in XLA's client side, not service
//   side. LHLO aims for the service side.

#ifndef LHLO_OPS
#define LHLO_OPS

include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir-hlo/Dialect/lhlo/IR/lhlo_dialect.td"
include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops_base.td"
include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.td"
include "mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.td"

//===----------------------------------------------------------------------===//
// LMHLO nullary op definitions.
//===----------------------------------------------------------------------===//

class LHLO_Op<string mnemonic, list<Trait> traits> :
  Op<LHLO_Dialect, mnemonic,
    !listconcat([MemoryEffects<[MemRead, MemWrite]>,
    LmhloStructuredInterface], traits)>;

def LHLO_ConstantOp : LHLO_Op<"constant", []> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];
  let arguments = (ins
    ElementsAttr:$value,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );

  let hasCanonicalizer = 1;
}

def LHLO_IotaOp : LHLO_Op<"iota", []> {
  let summary = "Iota operator";
  let description = [{
    Creates a rank 1 array of values starting at zero and incrementing by one.
  }];
  let arguments = (ins I64Attr:$iota_dimension,
                   Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

//===----------------------------------------------------------------------===//
// LMHLO unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class LHLO_UnaryElementwiseOp<string mnemonic,
                              Type BufferType = LHLO_Buffer,
                              list<Trait> traits = [SameTypeOperands, Elementwise]>
    : LHLO_Op<mnemonic, traits> {
  let arguments = (ins Arg<BufferType, "", [MemRead]>:$input,
                       Arg<BufferType, "", [MemWrite]>:$output);
}

// Abs supports complex to real, so element type is not guaranteed to match.
def LHLO_AbsOp: LHLO_UnaryElementwiseOp<"abs", LHLO_Buffer, [SameOperandsShape]> {
  let summary = "Absolute value operator";
  let description = [{
    Returns `abs(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
  let hasVerifier = 1;
}

// TODO(timshen): add a custom verifier.
def LHLO_BitcastConvertOp:
    LHLO_UnaryElementwiseOp<"bitcast_convert", LHLO_Buffer, [SameOperandsShape]> {
  let summary = "BitcastConvert operator";
  let description = [{
    Similar to a 'tf.bitcast' in TensorFlow, performs an element-wise bitcast
    operation from a data shape to a target shape. The dimensions must match,
    and the conversion is an element-wise one. Bitcast is implemented as a
    low-level cast, so machines with different floating-point representations
    will give different results.

    See https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype.
  }];
}
def LHLO_CbrtOp: LHLO_UnaryElementwiseOp<"cbrt", LHLO_FpBuffer> {
  let summary = "Cubic root operator";
  let description = [{
    Returns element-wise cubic root of the operand.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_CeilOp: LHLO_UnaryElementwiseOp<"ceil", LHLO_FpBuffer> {
  let summary = "Ceil operator";
  let description = [{
    Returns `Ceil(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_ClzOp: LHLO_UnaryElementwiseOp<"count_leading_zeros", LHLO_IntBuffer> {
  let summary = "Count-leading-zeros (Clz) operator";
  let description = [{
    Returns the number of leading zeros in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
// TODO(timshen): add a custom verifier.
def LHLO_ConvertOp : LHLO_UnaryElementwiseOp<"convert", LHLO_Buffer, [SameOperandsShape]> {
  let summary = "Convert operator";
  let description = [{
    Performs element-wise conversion of values from one type to another, e.g.
    float to int.

    See https://www.tensorflow.org/xla/operation_semantics#convertelementtype.
  }];
}
def LHLO_CosOp: LHLO_UnaryElementwiseOp<"cosine", LHLO_FpOrComplexBuffer> {
  let summary = "Cos operator";
  let description = [{
    Returns `Cos(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_ExpOp: LHLO_UnaryElementwiseOp<"exponential", LHLO_FpOrComplexBuffer> {
  let summary = "Exponential operator";
  let description = [{
    Returns `e^(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_Expm1Op: LHLO_UnaryElementwiseOp<"exponential_minus_one", LHLO_FpOrComplexBuffer> {
  let summary = "Exponential minus one operator";
  let description = [{
    Returns `e^(operand) - 1` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_FloorOp: LHLO_UnaryElementwiseOp<"floor", LHLO_FpBuffer> {
  let summary = "Floor operator";
  let description = [{
    Returns `Floor(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_ImagOp: LHLO_Op<"imag", [SameOperandsShape]> {
  let summary = "Imag operator";
  let description = [{
    Returns `Imag(operand)` element-wise.
  }];
  let arguments = (ins Arg<LHLO_ComplexBuffer, "", [MemRead]>:$input,
                       Arg<LHLO_FpBuffer, "", [MemWrite]>:$output);
}

def LHLO_IsFiniteOp: LHLO_Op<"is_finite", [SameOperandsShape]> {
  let summary = "IsFinite operator";
  let description = [{
    Tests whether each element of operand is finite, i.e., is not positive or
    negative infinity, and is not NaN. Returns a tensor of 1-bit integers with
    the same shape as the input, where each element is nonzero (i.e. true) if
    and only if the corresponding input element is finite.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
  let arguments = (ins Arg<LHLO_FpBuffer, "", [MemRead]>:$input,
                       Arg<LHLO_PredBuffer, "", [MemWrite]>:$output);
}

def LHLO_LogOp: LHLO_UnaryElementwiseOp<"log", LHLO_FpOrComplexBuffer> {
  let summary = "Logarithm operator";
  let description = [{
    Returns `log(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_LogisticOp : LHLO_UnaryElementwiseOp<"logistic", LHLO_FpOrComplexBuffer> {
  let summary = "Logistic operator";
  let description = [{
    Returns `logistic(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_Log1pOp: LHLO_UnaryElementwiseOp<"log_plus_one", LHLO_FpOrComplexBuffer> {
  let summary = "Log1p operator";
  let description = [{
    Returns `log(operand+1)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_NegOp: LHLO_UnaryElementwiseOp<"negate"> {
  let summary = "Negation operator";
  let description = [{
    Returns `-operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_NotOp: LHLO_UnaryElementwiseOp<"not", LHLO_PredOrIntBuffer> {
  let summary = "Not operator";
  let description = [{
    Returns `!operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_PopulationCountOp: LHLO_UnaryElementwiseOp<"popcnt", LHLO_IntBuffer> {
  let summary = "PopulationCount operator";
  let description = [{
    Returns the number of bits set in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_RealOp: LHLO_Op<"real", [SameOperandsShape]> {
  let summary = "Real operator";
  let description = [{
    Returns `Real(operand)` element-wise.
  }];
  let arguments = (ins Arg<LHLO_ComplexBuffer, "", [MemRead]>:$input,
                       Arg<LHLO_FpBuffer, "", [MemWrite]>:$output);
}

def LHLO_RoundOp: LHLO_UnaryElementwiseOp<"round_nearest_afz", LHLO_FpBuffer> {
  let summary = "Round operator";
  let description = [{
    Returns `Round(operand)` element-wise, rounding to nearest integer with
    half-way cases rounding away from zero.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_RsqrtOp: LHLO_UnaryElementwiseOp<"rsqrt", LHLO_FpOrComplexBuffer> {
  let summary = "Reciprocal Square-root operator";
  let description = [{
    Returns `1.0 / sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_SqrtOp: LHLO_UnaryElementwiseOp<"sqrt", LHLO_FpOrComplexBuffer> {
  let summary = "Square-root operator";
  let description = [{
    Returns `sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_SignOp: LHLO_UnaryElementwiseOp<"sign"> {
  let summary = "Sign operator";
  let description = [{
    Returns `sign(operand)` element-wise, where

    ```
    sign(x) = -1  : x < 0
            = -0  : x = -0
            = NaN : x = NaN
            = +0  : x = +0
            = 1   : x > 0
    ```

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_SineOp: LHLO_UnaryElementwiseOp<"sine", LHLO_FpOrComplexBuffer> {
  let summary = "Sin operator";
  let description = [{
    Returns `Sin(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def LHLO_TanhOp: LHLO_UnaryElementwiseOp<"tanh", LHLO_FpOrComplexBuffer> {
  let summary = "Tanh operator";
  let description = [{
    Returns `tanh(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
//===----------------------------------------------------------------------===//
// LMHLO binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class LHLO_BinaryElementwiseOp<string mnemonic, Type BufferType = LHLO_Buffer,
                               list<Trait> traits = [SameTypeOperands, Elementwise]> :
        LHLO_Op<mnemonic, traits> {
  let arguments = (ins
      Arg<BufferType, "", [MemRead]>:$lhs,
      Arg<BufferType, "", [MemRead]>:$rhs,
      Arg<BufferType, "", [MemWrite]>:$out,
      OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
}

def LHLO_AddOp : LHLO_BinaryElementwiseOp<"add"> {
  let summary = "Addition operator";
  let description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_AndOp: LHLO_BinaryElementwiseOp<"and", LHLO_PredOrIntBuffer> {
  let summary = "Logical and";
  let description = [{
    Returns `logical_and(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_Atan2Op : LHLO_BinaryElementwiseOp<"atan2", LHLO_FpOrComplexBuffer> {
  let summary = "Atan2 operator";
  let description = [{
    Returns `atan2(lhs/rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_ComplexOp: LHLO_Op<"complex", [SameOperandsShape]> {
  let summary = "Complex operator";
  let description = [{
    Performs element-wise conversion of a pair of real and imaginary values to
    a complex value.
  }];
  let arguments = (ins
      Arg<LHLO_FpBuffer, "", [MemRead]>:$lhs,
      Arg<LHLO_FpBuffer, "", [MemRead]>:$rhs,
      Arg<LHLO_ComplexBuffer, "", [MemWrite]>:$output,
      OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
}

def LHLO_DivOp : LHLO_BinaryElementwiseOp<"divide"> {
  let summary = "Division operator";
  let description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_MaxOp : LHLO_BinaryElementwiseOp<"maximum"> {
  let summary = "Maximum operator";
  let description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_MinOp : LHLO_BinaryElementwiseOp<"minimum"> {
  let summary = "Minimum operator";
  let description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_MulOp : LHLO_BinaryElementwiseOp<"multiply"> {
  let summary = "Multiplication operator";
  let description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_OrOp : LHLO_BinaryElementwiseOp<"or", LHLO_PredOrIntBuffer> {
  let summary = "Logical or";
  let description = [{
    Returns `logical_or(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_PowOp : LHLO_BinaryElementwiseOp<"power"> {
  let summary = "Power operator";
  let description = [{
    Returns `lhs ^ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_RemOp : LHLO_BinaryElementwiseOp<"remainder", LHLO_IntOrFpBuffer> {
  let summary = "Remainder operator";
  let description = [{
    Returns `lhs % rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_ShiftLeftOp : LHLO_BinaryElementwiseOp<"shift_left", LHLO_IntBuffer> {
  let summary = "Shift Left operator";
  let description = [{
    Returns `lhs << rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_ShiftRightArithmeticOp : LHLO_BinaryElementwiseOp<"shift_right_arithmetic", LHLO_IntBuffer> {
  let summary = "Shift right arithmetic operator";
  let description = [{
    Returns arithmetic `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_ShiftRightLogicalOp : LHLO_BinaryElementwiseOp<"shift_right_logical", LHLO_IntBuffer> {
  let summary = "Shift right logical operator";
  let description = [{
    Returns logical `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_SubOp : LHLO_BinaryElementwiseOp<"subtract"> {
  let summary = "Subtraction operator";
  let description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def LHLO_XorOp : LHLO_BinaryElementwiseOp<"xor", LHLO_PredOrIntBuffer> {
  let summary = "Logical xor";
  let description = [{
    Returns `logical_xor(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
//===----------------------------------------------------------------------===//
// LMHLO control flow op definitions.
//===----------------------------------------------------------------------===//

// TODO(b/139813999): specify required function signature in a type-safe way.
//
// The region `body` may return lmhlo.TerminatorOp or mhlo.ReturnOp. We are
// moving towards mhlo.ReturnOp, but some code that needs cleanup still assumes lmhlo.TerminatorOp.
// TODO(timshen): cleanup lmhlo.TerminatorOp.
def LHLO_ReduceOp: LHLO_Op<"reduce", [SameVariadicOperandSize]> {
  let summary = "Reduce operator";
  let description = [{
    Returns the result of executing a reduction function on one or more arrays
    in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reduce.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$init_values,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$out,
    I64ElementsAttr:$dimensions
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCanonicalizer = 1;
}

def LHLO_ReduceWindowOp: LHLO_Op<"reduce_window", [SameVariadicOperandSize]> {
  let summary = "ReduceWindow operator";
  let description = [{
    Returns the result of executing a reduction function over all elements in
    each window of one or more arrays in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reducewindow.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$init_values,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$out,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the input dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

// TODO(timshen): Add a custom syntax for this.
def LHLO_CaseOp: LHLO_Op<"case", [
      SingleBlockImplicitTerminator<"TerminatorOp">,
      DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Switch-Case operator";
  let description = [{
    Returns the result of executing `branches[index]`. If
    `index` is < 0 or >= N, then `branches[N-1] is executed as
    the default branch.

    Each branch `branches[b]` must take in a single argument of same type as
    `branch_operands[b]` and will be invoked with `branch_operands[b]`. The type
    of the returned value of each branch must be the same.

    Note that only one of the branches will be executed depending on the value
    of index.
    See https://www.tensorflow.org/xla/operation_semantics#conditional.
  }];

  let arguments = (ins Arg<LHLO_PredOrIntBuffer, "", [MemRead]>:$index);

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
}

// TODO(timshen): Add a custom syntax for this.
def LHLO_WhileOp: LHLO_Op<"while", [
      DeclareOpInterfaceMethods<RegionBranchOpInterface>,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "While operator";
  let description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_PredBuffer>, "", [MemWrite]>:$cond_val,
    OptionalAttr<I64Attr>:$trip_count);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);
}

def LHLO_CustomCallOp : LHLO_Op<"custom_call", [AttrSizedOperandSegments]> {
  let summary = "CustomCall operator";
  let description = [{
    A custom call invokes code external to XLA. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.

    `call_target_name` and `backend_config` can be arbitrary strings, but
    `call_target_name` should be short as it may be used in labels.
    `backend_config` can encode arbitrarily large amounts of information.

    See https://www.tensorflow.org/xla/operation_semantics#customcall.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$args,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$output,
    StrAttr:$call_target_name,
    DefaultValuedAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    // TODO(b/189822916): Remove this field when all clients are migrated to
    // the status-returning API.
    DefaultValuedAttr<HLO_CustomCallApiVersionAttr,
                      "mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
                      $api_version,
    OptionalAttr<CustomCallTargetArgMappingAttr>:$target_arg_mapping
  );
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LMHLO tuple op definitions.
//===----------------------------------------------------------------------===//

def LHLO_CompareOp: LHLO_Op<"compare", []> {
  let summary = "Comparison operator";
  let description = [{
    Compares `lhs` and `rhs` elementwise according to `comparison_direction`
    and `compare_type`. If unspecified, `compare_type` is FLOAT for float element
    types, SIGNED for signed element types and UNSIGNED for unsigned element
    types.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    Arg<LHLO_PredBuffer, "", [MemWrite]>:$out,
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions,
    HLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<HLO_ComparisonTypeAttr>:$compare_type
  );
}

//===----------------------------------------------------------------------===//
// LMHLO Slice definitions.
//===----------------------------------------------------------------------===//

def LHLO_SliceOp: LHLO_Op<
      "slice",
      [AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );
}

def LHLO_DynamicSliceOp: LHLO_Op<"dynamic_slice",
      [AllElementTypesMatch<["operand", "output"]>]> {
  let summary = "Dynamic Slice operator";
  let description = [{
    Extracts a sub-array from the input array at dynamic start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicslice.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$start_indices,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$slice_sizes
  );
}

def LHLO_DynamicUpdateSliceOp: LHLO_Op<"dynamic-update-slice", []> {
  let summary = "Dynamic Update Slice operator";
  let description = [{
    DynamicUpdateSlice generates a result which is the value of the input array
    operand, with a slice update overwritten at start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$update,
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$start_indices,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

//===----------------------------------------------------------------------===//
// LMHLO Other op definitions.
//===----------------------------------------------------------------------===//

def LHLO_BatchNormGradOp : LHLO_Op<"batch_norm_grad", []> {
  let summary = "Batch Normalization Gradient";
  let description = [{
    Calculates gradients of batch norm.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormgrad
  }];

  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$scale,
    Arg<LHLO_Buffer, "", [MemRead]>:$mean,
    Arg<LHLO_Buffer, "", [MemRead]>:$variance,
    Arg<LHLO_Buffer, "", [MemRead]>:$grad_output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$grad_operand,  // gradient of $operand.
    Arg<LHLO_Buffer, "", [MemWrite]>:$grad_scale,
    Arg<LHLO_Buffer, "", [MemWrite]>:$grad_offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

}

def LHLO_BatchNormInferenceOp : LHLO_Op<"batch_norm_inference", []> {
  let summary = "Batch Normalization for Inference";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnorminference
  }];

  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$scale,
    Arg<LHLO_Buffer, "", [MemRead]>:$offset,
    Arg<LHLO_Buffer, "", [MemRead]>:$mean,
    Arg<LHLO_Buffer, "", [MemRead]>:$variance,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
}

def LHLO_BatchNormTrainingOp : LHLO_Op<"batch_norm_training", []> {
  let summary = "Batch Normalization for Training";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormtraining
  }];

  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$scale,
    Arg<LHLO_Buffer, "", [MemRead]>:$offset,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    Arg<LHLO_Buffer, "", [MemWrite]>:$batch_mean,
    Arg<LHLO_Buffer, "", [MemWrite]>:$batch_var,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
}

def LHLO_BroadcastOp : LHLO_Op<"broadcast",
      []> {
  let summary = "Broadcast a tensor to a higher rank by prepending dimensions";
  let description = [{
    Broadcasts the operand tensor to a higher rank by prepending
    `broadcast_sizes` to the dimensions. The current values of the operand are
    copied into the other dimensions.

    This is a more limited form of broadcasting, that corresponds to the XLA
    client Broadcast method. For a more general form of broadcasting, see the
    BroadcastInDimOp.

    See https://www.tensorflow.org/xla/operation_semantics#broadcast.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$broadcast_sizes
  );
}

def LHLO_BroadcastInDimOp : LHLO_Op<"broadcast_in_dim",
      []> {
  let summary = "Broadcast a tensor into the given shape by adding dimensions.";
  let description = [{
    Broadcasts the `operand` tensor to a higher rank. This is not the limited
    form of broadcasting exposed as the XLA client broadcast op, but rather the
    more powerful "InDim" broadcasting, which is closer to the HLO broadcast op
    and exposed in the XLA client BroadcastInDim method.

    `broadcast_dimensions` maps the operand dimension number to the target shape
    dimension number. It must have the same size as the rank of the operand. The
    mapped dimensions must either be the same size or the dimension being
    broadcast from must be size 1 (degenerate broadcasting).

    For a scalar (0D tensor) operand, `broadcast_dimensions` must be empty. The
    The scalar value will be broadcast to every element in the target shape.

    See https://www.tensorflow.org/xla/broadcasting.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    BroadcastDimAttr:$broadcast_dimensions
  );
}

def LHLO_ClampOp : LHLO_Op<"clamp", []> {
  let summary = "Clamp operator";
  let description = [{
    Clamps an operand to within the range between a minimum and maximum value.

    Note: All three arrays must be the same shape. Alternatively, as a
          restricted form of broadcasting, min and/or max can be a scalar (0D
          tensor) of the element type of the tensor operand.

    See https://www.tensorflow.org/xla/operation_semantics#clamp.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$min,
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$max,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_ConcatenateOp : LHLO_Op<"concatenate", []> {
  let summary = "XLA's concatenate op";
  let description = [{
     Concatenates a set of tensors along the specified dimension.

     See https://www.tensorflow.org/xla/operation_semantics#concatenate.
   }];
   let arguments = (ins
     Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$val,
     Arg<LHLO_Buffer, "", [MemWrite]>:$output,
     I64Attr:$dimension
   );
}

def LHLO_ConvolutionOp : LHLO_Op<"convolution", []> {
  let summary = "Convolution operator";
  let description = [{
    Computes a convolution of the kind used in neural networks.

    See https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
  }];
  let arguments = !con(
    (ins
       Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
       Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
       Arg<LHLO_Buffer, "", [MemWrite]>:$output),
    ConvolutionAttributes.attributes);

  code extraClassDeclaration = [{
    bool hasWindowReversal() {
      auto reversal = getWindowReversalAttr();
      return reversal && llvm::any_of(reversal.getValues<bool>(),
                                      [](bool v) { return v; });
    }
  }];

 let assemblyFormat = [{
    `(`operands`)`
       `dim_numbers` `=` custom<ConvolutionDimensions>($dimension_numbers) `,`
       `window` `=` `{` custom<WindowAttributes>($window_strides, $padding,
                                                 $lhs_dilation, $rhs_dilation,
                                                 $window_reversal) `}`
       attr-dict `:` functional-type(operands, results)
  }];
}

def LHLO_CopyOp: LHLO_Op<"copy", [CopyOpInterface]> {
  let summary = "Copy operator";
  let description = [{
    Returns a copy of `operand`.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );

  let extraClassDeclaration = [{
    Value getSource() { return getOperand();}
    Value getTarget() { return getOutput(); }
  }];
}

def LHLO_DotOp: LHLO_Op<"dot", []> {
  let summary = "Dot operator";
  let description = [{
    Performs dot products between vectors, vector/matrix and matrix/matrix
    multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dot.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    DotDimensionNumbers:$dot_dimension_numbers,
    HLO_PrecisionConfigAttr:$precision_config,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_GatherOp: LHLO_Op<"gather", []> {
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$start_indices,
    GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_ReshapeOp: LHLO_Op<"reshape", []> {
  let summary = "Reshape operator";
  let description = [{
    Reshapes the dimensions of `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#reshape.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_ScatterOp: LHLO_Op<"scatter", []> {
  let summary = "Scatter operator";
  let description = [{
    Generates a result which is the value of the input array `operand`,
    with several slices (at indices specified by `scatter_indices`)
    updated with the values in `updates` using `update_computation`.

    See https://www.tensorflow.org/xla/operation_semantics#scatter.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$scatter_indices,
    Arg<LHLO_Buffer, "", [MemRead]>:$updates,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedAttr<BoolAttr, "false">:$unique_indices
  );

  let regions = (region SizedRegion<1>:$update_computation);
}

def LHLO_SelectOp: LHLO_Op<"select", [Elementwise]> {
  let summary = "Select operator";
  let description = [{
    Constructs an output tensor from the elements of `on_true` and `on_false`
    based on the values of `pred`.

    `pred`, `on_true` and `on_false` must be broadcast compatible.
  }];
  let arguments = (ins
    Arg<LHLO_PredBuffer, "", [MemRead]>:$pred,
    Arg<LHLO_Buffer, "", [MemRead]>:$on_true,
    Arg<LHLO_Buffer, "", [MemRead]>:$on_false,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_SelectAndScatterOp: LHLO_Op<"select_and_scatter", []> {
  let summary = "SelectAndScatter operator";
  let description = [{
    Runs a windowed selection `select` function over `operand` with shape
    `window_dimensions` and stride `window_strides`. This will produce an amount
    of selected locations whose shape matches `source`. These are then scattered
    to the output which is initialized with `init_value`.
    Multiple scattered elements which land in the same output location are
    combined using the `scatter` function.

    See https://www.tensorflow.org/xla/operation_semantics#selectandscatter.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$source,
    Arg<LHLO_Buffer, "", [MemRead]>:$init_value,
    Arg<LHLO_Buffer, "", [MemWrite]>:$out,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);
}

def LHLO_ReverseOp: LHLO_Op<"reverse", []> {
  let summary = "Reverse operator";
  let description = [{
    Reverses the specified dimensions of `operand` according to the given
    `dimensions`.

    See https://www.tensorflow.org/xla/operation_semantics#rev_reverse.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    I64ElementsAttr:$dimensions,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_PadOp: LHLO_Op<"pad", []> {
  let summary = "Pad operator";
  let description = [{
    Pads the edges of `operand` with the `padding_value` and according to
    the passed configuration.

    See https://www.tensorflow.org/xla/operation_semantics#pad.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high,
    I64ElementsAttr:$interior_padding,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
  let hasVerifier = 1;
}

def LHLO_TransposeOp: LHLO_Op<"transpose", []> {
  let summary = "Transpose operator";
  let description = [{
    Permutes the dimensions of `operand` according to the given `permutation`.

    `res_dimensions[i] = operand_dimensions[permutation[i]]`

    See https://www.tensorflow.org/xla/operation_semantics#transpose.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    I64ElementsAttr:$permutation,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_ReducePrecisionOp: LHLO_Op<"reduce_precision", [SameTypeOperands]> {
  let summary = "Reduce precision operator";
  let description = [{
    Models the effect of converting floating - point values to a lower -
    precision format(such as IEEE - FP16) and back to the original
    format. The number of exponent and mantissa bits in the lower -
    precision format can be specified arbitrarily,
    although all bit sizes may not be supported on all hardware
    implementations.

    See https://www.tensorflow.org/xla/operation_semantics#reduceprecision.
  }];
  let arguments = (ins
    Arg<LHLO_FpBuffer, "", [MemRead]>:$operand,
    Arg<LHLO_FpBuffer, "", [MemWrite]>:$output,
    I32Attr:$exponent_bits,
    I32Attr:$mantissa_bits
  );
}

// Common base class for AllReduce, AllGather, and AllToAll.
class LHLO_CollectiveCommunicationOp<string name, list<Trait> traits = []> :
  LHLO_Op<name, !listconcat(traits, [SameVariadicOperandSize])> {
  dag arguments_base = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$outputs,
    I64ElementsAttr:$replica_groups,
    DefaultValuedAttr<BoolAttr, "false">:$constrain_layout,
    OptionalAttr<ChannelHandle>:$channel_id,
    DefaultValuedAttr<BoolAttr, "false">:$use_global_device_ids
  );
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // AllGather is cross replica if channel_id is not set.
    bool IsCrossReplica() { return !getChannelId().hasValue(); }
  }];
}

def LHLO_AllGatherOp : LHLO_CollectiveCommunicationOp<"all_gather"> {
  let summary = "AllGather operator";
  let description = [{
    Performs concatenation across replicas.

    See https://www.tensorflow.org/xla/operation_semantics#allgather
  }];
  let arguments = !con(
    arguments_base,
    (ins I64Attr:$all_gather_dimension));
}

def LHLO_AllReduceOp : LHLO_CollectiveCommunicationOp<"all_reduce", [SameOperandsElementType]> {
  let summary = "AllReduce operator";
  let description = [{
    Performs a custom reduction across replicas.

    See https://www.tensorflow.org/xla/operation_semantics#allreduce.
  }];
  let arguments = arguments_base;
  let regions = (region SizedRegion<1>:$computation);
}

def LHLO_ReduceScatterOp : LHLO_CollectiveCommunicationOp<"reduce_scatter", [SameOperandsElementType]> {
  let summary = "ReduceScatter operator";
  let description = [{
     Performs all_reduce followed by a scatter.

     See https://www.tensorflow.org/xla/operation_semantics#reducescatter
  }];
  let arguments = !con(
    arguments_base,
    (ins I64Attr:$scatter_dimension));
  let regions = (region SizedRegion<1>:$computation);
}

def LHLO_AllToAllOp : LHLO_CollectiveCommunicationOp<"all_to_all", [SameOperandsElementType]> {
  let arguments = !con(
    arguments_base,
    (ins OptionalAttr<I64Attr>:$split_dimension));
}

def LHLO_CollectivePermuteOp: LHLO_Op<"collective_permute", [SameTypeOperands]> {
  let summary = "CollectivePermute operator";
  let description = [{
    CollectivePermute is a collective operation that sends and receives data
    cross replicas.
    Note that there are the following restrictions on the source_target_pair:
    - Any two pairs should not have the same target replica id, and they should
    not have the same source replica id.
    - If a replica id is not a target in any pair, then the output on that
    replica is a tensor consists of 0(s) with the same shape as the input.

    See https://www.tensorflow.org/xla/operation_semantics#collectivepermute.

  }];

  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$source_target_pairs,
    OptionalAttr<ChannelHandle>:$channel_id
  );
  let hasVerifier = 1;
}

def LHLO_FftOp: LHLO_Op<"fft", []> {
  let summary = "Fast fourier transform operator";
  let description = [{
    Returns the fast-fourier-transform of the input array.

    See
    https://www.tensorflow.org/xla/operation_semantics#fft.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    HLO_FftTypeAttr:$fft_type,
    I64ElementsAttr:$fft_length
  );
}

def LHLO_CholeskyOp: LHLO_Op<"cholesky", [SameOperandsElementType]> {
  let summary = "Cholesky operator";
  let description = [{
  Computes the Cholesky decomposition of a batch of symmetric (Hermitian)
  positive definite matrices.

  If lower is true, computes lower-triangular matrices l such that
  `a=l.Transpose(l)`. If lower is false, computes upper-triangular matrices u such
  that `a=Transpose(u).u`.

  Input data is read only from the lower/upper triangle of a, depending on the
  value of lower. Values from the other triangle are ignored. Output data is
  returned in the same triangle; the values in the other triangle are
  implementation-defined and may be anything.

  If the rank of a is greater than 2, a is treated as a batch of matrices, where
  all except the minor 2 dimensions are batch dimensions.

  If a is not symmetric (Hermitian) positive definite, the result is
  implementation-defined.

    See https://www.tensorflow.org/xla/operation_semantics#cholesky.
  }];
  let arguments = (ins
    Arg<LHLO_FpOrComplexBuffer, "", [MemRead]>:$a,
    Arg<LHLO_FpOrComplexBuffer, "", [MemWrite]>:$output,
    DefaultValuedAttr<BoolAttr, "false">:$lower
  );
}

def LHLO_InfeedOp: LHLO_Op<"infeed", []> {
  let summary = "Infeed operator";
  let description = [{
    Reads a single data item from the implicit Infeed streaming interface of
    the device, interpreting the data as the given shape and its layout, and
    returns an LHLO op of the data. Multiple Infeed operations are allowed in a
    computation, but there must be a total order among the Infeed operations.
    For example, two Infeeds in the code below have a total order since there
    is a dependency between the while loops.

    See https://www.tensorflow.org/xla/operation_semantics#infeed
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$outputs,
    DefaultValuedStrAttr<StrAttr, "">:$config
  );
}

def LHLO_OutfeedOp: LHLO_Op<"outfeed", []> {
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$config
  );
}

def LHLO_ReplicaIdOp : LHLO_Op<"replica_id", []> {
  let summary = "ReplicaId operator";
  let description = [{
    Returns the unique ID (int32 scalar) of the replica.

    The unique ID of each replica is an unsigned integer in the interval [0, N),
    where N is the number of replicas. Since all the replicas are running the
    same program, a ReplicaId() call in the program will return a different
    value on each replica.

    See https://www.tensorflow.org/xla/operation_semantics#replicaid.
  }];
  let arguments = (ins Arg<MemRefOf<[UI32]>, "", [MemWrite]>);
}

def LHLO_PartitionIdOp : LHLO_Op<"partition_id", []> {
  let summary = "PartitionId operator";
  let description = [{
    Returns the unique ID (int32 scalar) of the partition.
  }];
  let arguments = (ins Arg<MemRefOf<[UI32]>, "", [MemWrite]>);
}

def LHLO_TriangularSolveOp: LHLO_Op<"triangular_solve", [SameOperandsElementType]> {
  let summary = "TriangularSolve operator";
  let description = [{
    Solves systems of linear equations with lower or upper triangular
    coefficient matrices by forward- or back-substitution. Broadcasting along
    leading dimensions, this routine solves one of the matrix systems
    op(a) * x = b, or x * op(a) = b, for the variable x, given a and b, where
    op(a) is either op(a) = a, or op(a) = Transpose(a), or
    op(a) = Conj(Transpose(a)).

    Input data is read only from the lower/upper triangle of a, depending on the
    value of lower. Values from the other triangle are ignored. Output data is
    returned in the same triangle; the values in the other triangle are
    implementation-defined and may be anything.

    If the rank of a and b are greater than 2, they are treated as batches of
    matrices, where all except the minor 2 dimensions are batch dimensions. a
    and b must have equal batch dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#triangularsolve.
  }];
  let arguments = (ins
    Arg<LHLO_FpOrComplexBuffer, "", [MemRead]>:$a,
    Arg<LHLO_FpOrComplexBuffer, "", [MemRead]>:$b,
    Arg<LHLO_FpOrComplexBuffer, "", [MemWrite]>:$output,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    HLO_TransposeAttr:$transpose_a,
    HLO_LayoutAttr:$layout_a,
    HLO_LayoutAttr:$layout_b,
    HLO_LayoutAttr:$layout_output
  );
}

// TODO(timshen): add a custom verifier.
def LHLO_MapOp: LHLO_Op<"map", [SameOperandsShape]> {
  let summary = "Map operator";
  let description = [{
  Applies a scalar function over the given operands arrays, producing an array
  of the same dimensions where each element is the result of the mapped function
  applied to the corresponding elements in the input arrays.

  The mapped function is an arbitrary computation with the restriction that it
  has N inputs of scalar type T and a single output with type S. The output has
  the same dimensions as the operands except that the element type T is replaced
  with S.

  See https://www.tensorflow.org/xla/operation_semantics#map.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
}

def LHLO_RngGetAndUpdateStateOp: LHLO_Op<"rng_get_and_update_state", []> {
  let arguments = (ins
    Arg<MemRefOf<[UI64]>, "", [MemRead, MemWrite]>:$state,
    I64Attr:$delta
  );
}

// TODO(timshen): add a custom verifier.
def LHLO_SortOp: LHLO_Op<"sort", [SameVariadicOperandSize, SameOperandsShape]> {
  let summary = "Sort operator";
  let description = [{
    Sorts the given `operands` at the given `dimension` with the given
    `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];
  let arguments = (ins
    Arg<Variadic<LHLO_Buffer>, "", [MemRead]>:$inputs,
    Arg<Variadic<LHLO_Buffer>, "", [MemWrite]>:$output,
    DefaultValuedAttr<I64Attr, "-1">:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$is_stable
  );

  let regions = (region SizedRegion<1>:$comparator);
}

//===----------------------------------------------------------------------===//
// Late operations
//===----------------------------------------------------------------------===//

def FusionOp : LHLO_Op<"fusion", [
      SingleBlockImplicitTerminator<"TerminatorOp">,
      DeclareOpInterfaceMethods<RegionBranchOpInterface>
      ]> {
  let summary = "Fusion operator";
  let description = [{
    Models the fusion instruction generated by the XLA compiler's fusion pass.

    Fusion instructions are generated by the fusion pass of the XLA compiler.
    They serve as a hint to the backend that it is beneficial to emit the
    contained instructions into a single loop nest or kernel. The XLA fusion
    pass is designed such that it only generates fusion nodes that can be
    handled by the XLA compilers backends.
    The XLA runtime expects this hint to be followed, as it expects a single
    kernel per HLO instruction. This restriction might be lifted in the future.
  }];
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
     OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    SmallVector<Value> getInputBuffers() {
      SmallVector<Value> buffers;
      for (auto load : getRegion().front().getOps<bufferization::ToTensorOp>()) {
        buffers.push_back(load.getMemref());
      }
      return buffers;
    }

    SmallVector<Value> getOutputBuffers() {
      SmallVector<Value> buffers;
      for (auto store : getRegion().front().getOps<memref::TensorStoreOp>()) {
        buffers.push_back(store.memref());
      }
      return buffers;
    }

    SmallVector<Value> getFusionParameters() {
      SmallVector<Value> buffers;
      for (auto load : getRegion().front().getOps<bufferization::ToTensorOp>()) {
        buffers.push_back(load);
      }
      return buffers;
    }

    SmallVector<Value> getFusionResults() {
      SmallVector<Value> buffers;
      for (auto store : getRegion().front().getOps<memref::TensorStoreOp>()) {
        buffers.push_back(store.tensor());
      }
      return buffers;
    }

    SmallVector<Operation*> getFusionRoots() {
      SmallVector<Operation*> roots;
      for (auto value : getFusionResults()) {
        Operation* op = value.getDefiningOp();
        if (roots.empty() || roots.back() != op) {
          roots.push_back(op);
        }
      }
      return roots;
    }
  }];
}

def TerminatorOp :
    LHLO_Op<"terminator", [ReturnLike, Terminator]> {
  let summary = "LHLO termination operation";
  let description = [{
    Terminator operation for the LHLO dialect.
  }];
  let builders = [
    OpBuilder<(ins "ValueRange":$operands),
    [{ build($_builder, $_state, llvm::None, operands, llvm::None); }]>];
}

def LHLO_RealDynamicSliceOp: LHLO_Op<
      "real_dynamic_slice",
      [AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let summary = "LHLO Real Dynamic Slice operator";
  let description = [{
    The dynamic shape version of DynamicSliceOp. Extracts a sub-array from the
    input array according to dynamic start_indices, limit_indices and strides.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$start_indices,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$limit_indices,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$strides,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_DynamicBroadcastInDimOp : LHLO_Op<"dynamic_broadcast_in_dim",
      []> {
  let summary = "Broadcast a tensor into the given dynamic shape by adding dimensions.";
  let description = [{
    The dynamic shape version of BroadcastInDimOp. This is a generalization of the
    BroadcastInDimOp which accepts its output dimensions as an argument. It should
    eventually supercede the statically shaped original, but is being phased as a
    separate op in order to support compatibility with lowerings and translations that
    precede dynamic shapes.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$output_dimensions,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output,
    BroadcastDimAttr:$broadcast_dimensions
  );
}

def LHLO_DotGeneralOp: LHLO_Op<"dot_general", []> {
  let summary = "LHLO General Dot operator";
  let description = [{
    Performs general dot products between vectors, vector/matrix and
    matrix/matrix multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dotgeneral.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    DotDimensionNumbers:$dot_dimension_numbers,
    HLO_PrecisionConfigAttr:$precision_config,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_DynamicGatherOp: LHLO_Op<"dynamic_gather", []> {
  string summary = "LHLO Dynamic Gather operator";
  string description = [{
    The dynamic shape version of GatherOp. Stitches together several slices of an input
    array. slice_sizes is not a const.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$start_indices,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$slice_sizes,
    GatherDimensionNumbers:$dimension_numbers,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_DynamicPadOp: LHLO_Op<
      "dynamic_pad",
      [AllTypesMatch<["edge_padding_low", "edge_padding_high", "interior_padding"]>]> {
  let summary = "LHLO Dynamic Pad operator";
  let description = [{
    The dynamic shape version of PadOp. Pads the edges of `operand` with the `padding_value` and according to
    the passed configuration. Passed configuration are dynamic shape.
    See
    https://www.tensorflow.org/xla/operation_semantics#pad
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemRead]>:$padding_value,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$edge_padding_low,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$edge_padding_high,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$interior_padding,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_BitcastOp: LHLO_Op<"bitcast", []> {
  let summary = "LHLO Bitcast operator";
  let description = [{
    This op changes the shape of the input in the way that the physical
    arrangement of elements are unchanged.

    However, the op needs layout information to make sense of "physical
    arrangement of elements". Layout support in MHLO is currently under
    exploration.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_DynamicBitcastOp: LHLO_Op<"dynamic_bitcast", []> {
  let summary = "LHLO Dynamic Bitcast operator";
  let description = [{
    The dynamic shape version of BitcastOp. This op changes the shape of the
    input in the way that the physical arrangement of elements are unchanged.

    However, the op needs layout information to make sense of "physical
    arrangement of elements". Layout support in MHLO is currently under
    exploration.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_IntBuffer, "", [MemRead]>:$shape,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

def LHLO_DynamicIotaOp : LHLO_Op<"dynamic_iota", []> {
  let summary = "Create linear increasing values from 0 to length -1.";
  let description = [{
    The dynamic shape version of IotaOp. Produces an output of the specified shape,
    with an incremental set of values along the specified dimension starting at 0.
    See
    https://www.tensorflow.org/xla/operation_semantics#iota
  }];
  let arguments = (ins Arg<LHLO_DimensionBuffer, "", [MemRead]>:$shape,
                   I64Attr:$iota_dimension,
                   Arg<LHLO_Buffer, "", [MemWrite]>:$output);
}

def LHLO_DynamicConvOp : LHLO_Op<"dynamic_conv", []> {
  let arguments = !con(
    (ins Arg<LHLO_Buffer, "", [MemRead]>:$lhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$rhs,
    Arg<LHLO_Buffer, "", [MemRead]>:$d_padding,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output),
    ConvolutionAttributes.attributes);
}

def LHLO_DynamicReshapeOp: LHLO_Op<"dynamic_reshape", []> {
  let summary = "Reshape a tensor to a given, possibly dynamic, shape.";
  let description = [{
    The dynamic shape version of ReshapeOp. Reshapes `operand` to `output`.
  }];
  let arguments = (ins
    Arg<LHLO_Buffer, "", [MemRead]>:$operand,
    Arg<LHLO_DimensionBuffer, "", [MemRead]>:$shape,
    Arg<LHLO_Buffer, "", [MemWrite]>:$output
  );
}

#endif // LHLO_OPS
