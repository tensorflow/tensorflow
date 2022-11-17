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

#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_BASE
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_BASE

include "stablehlo/dialect/Base.td"

//===----------------------------------------------------------------------===//
// MHLO type definitions.
//===----------------------------------------------------------------------===//

defvar MHLO_Pred = HLO_Pred;

defvar MHLO_SInt = HLO_SInt;
defvar MHLO_UInt = HLO_UInt;
defvar MHLO_Int = HLO_Int;

def MHLO_F8E4M3FN : Type<CPred<"$_self.isFloat8E4M3FN()">, "f8E4M3FN type">,
                    BuildableType<"$_builder.getFloat8E4M3FNType()">;
def MHLO_F8E5M2 : Type<CPred<"$_self.isFloat8E5M2()">, "f8E5M2 type">,
                  BuildableType<"$_builder.getFloat8E5M2Type()">;

def MHLO_Float : AnyTypeOf<[MHLO_F8E4M3FN, MHLO_F8E5M2, F16, F32, F64, BF16]>;
defvar MHLO_Float32Or64 = HLO_Float32Or64;

defvar MHLO_Complex = HLO_Complex;

//===----------------------------------------------------------------------===//
// Quantized element type definitions.
//===----------------------------------------------------------------------===//

// Integer-based uniform quantized types. The definitions can be used to specify
// operand's tensor types.
defvar MHLO_QuantizedSignedInt = HLO_QuantizedSignedInt;
defvar MHLO_QuantizedUnsignedInt = HLO_QuantizedUnsignedInt;
defvar MHLO_QuantizedInt = HLO_QuantizedInt;

// The broadcasting dimensions correspond to a tuple that describes how a
// smaller rank shape is broadcast into a larger rank shape. For example,
// given a 2x3x4 cuboid and a 3x4 matrix, a broadcasting tuple (1,2) means
// matching the matrix to dimensions 1 and 2 of the cuboid.
defvar MHLO_BroadcastDimAttr = I64ElementsAttr;

// Token type.
defvar MHLO_Token = HLO_Token;

// Any integer tensor types
defvar MHLO_IntTensor = HLO_IntTensor;

// Any integer tensor type with rank 0 (i.e. representing a single integer).
defvar MHLO_ScalarIntTensor = HLO_ScalarIntTensor;

// Any floating-point tensor types
def MHLO_FpTensor : TensorOf<[MHLO_Float]>;

// 32 or 64 bits floating-point tensor types
defvar MHLO_Fp32Or64Tensor = HLO_Fp32Or64Tensor;

// Any quantized integer tensor types
defvar MHLO_QuantizedIntTensor = HLO_QuantizedIntTensor;

defvar MHLO_PredTensor = HLO_PredTensor;

def MHLO_Tensor : TensorOf<[MHLO_Float, MHLO_Pred, MHLO_Int, MHLO_Complex, MHLO_QuantizedInt]>;

defvar MHLO_ComplexTensor = HLO_ComplexTensor;

defvar MHLO_Tuple = HLO_Tuple;

defvar MHLO_TensorOrToken = HLO_TensorOrToken;

defvar MHLO_TensorOrTokenOrTuple = HLO_TensorOrTokenOrTuple;

defvar MHLO_DimensionValue = HLO_DimensionValue;

// Dynamic representation of a shape vector as a tensor.
defvar MHLO_DimensionTensor = HLO_DimensionTensor;

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
def MHLO_StaticShapeTensor : StaticShapeTensorOf<[
      MHLO_Float, MHLO_Pred, MHLO_Int, MHLO_Complex, MHLO_QuantizedInt]>;

//===----------------------------------------------------------------------===//
// MHLO combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or floating-point tensor types

// Any integer or floating-point tensor types
def MHLO_IntOrFpTensor : TensorOf<[MHLO_Int, MHLO_Float]>;

// Any integer or predicate tensor types
defvar MHLO_PredOrIntTensor = HLO_PredOrIntTensor;

// Any floating-point or complex tensor types
def MHLO_FpOrComplexTensor : TensorOf<[MHLO_Float, MHLO_Complex]>;

// Any int, floating-point or complex tensor types
def MHLO_IntFpOrComplexTensor : TensorOf<[MHLO_Int, MHLO_Float, MHLO_Complex]>;

// Any pred, int or floating-point tensor types
def MHLO_PredIntOrFpTensor : TensorOf<[MHLO_Pred, MHLO_Int, MHLO_Float]>;

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_BASE
