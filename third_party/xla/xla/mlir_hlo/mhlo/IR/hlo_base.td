/* Copyright 2019 The OpenXLA Authors.

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

defvar MHLO_Float = HLO_Float;
defvar MHLO_Float32Or64 = HLO_Float32Or64;

defvar MHLO_Complex = HLO_Complex;

//===----------------------------------------------------------------------===//
// Quantized element type definitions.
//===----------------------------------------------------------------------===//

// Integer-based uniform quantized types. The definitions can be used to specify
// operand's tensor types.
defvar MHLO_QuantizedSignedInt = AnyTypeOf<[HLO_QuantizedSignedInt, HLO_PerAxisQuantizedSignedInt]>;
defvar MHLO_QuantizedUnsignedInt = AnyTypeOf<[HLO_QuantizedUnsignedInt, HLO_PerAxisQuantizedUnsignedInt]>;
defvar MHLO_QuantizedInt = AnyTypeOf<[HLO_QuantizedSignedInt, HLO_PerAxisQuantizedSignedInt, HLO_QuantizedUnsignedInt, HLO_PerAxisQuantizedUnsignedInt]>;

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
defvar MHLO_FpTensor = HLO_FpTensor;

// 32 or 64 bits floating-point tensor types
defvar MHLO_Fp32Or64Tensor = HLO_Fp32Or64Tensor;

// Any quantized integer tensor types
defvar MHLO_QuantizedIntTensor = HLO_QuantizedIntOrPerAxisQuantizedIntTensor;

defvar MHLO_PredTensor = HLO_PredTensor;

// TODO: b/327490705 - Change this alias back to HLO_Tensor once MHLO_Tensor no
// longer needs per axis quantization.
defvar MHLO_Tensor = HLO_TensorOrPerAxisQuantizedTensor;

defvar MHLO_AnyTensor = HLO_AnyTensor;

defvar MHLO_ComplexTensor = HLO_ComplexTensor;

defvar MHLO_Tuple = HLO_Tuple;

defvar MHLO_TensorOrToken = HLO_TensorOrPerAxisQuantizedTensorOrToken;

defvar MHLO_TensorOrTokenOrTuple = AnyTypeOf<[MHLO_Tensor, MHLO_Token, MHLO_Tuple]>;

defvar MHLO_DimensionValue = HLO_DimensionValue;

// Dynamic representation of a shape vector as a tensor.
defvar MHLO_DimensionTensor = HLO_DimensionTensor;

defvar MHLO_CustomCallValue = HLO_CustomCallValue;

//===----------------------------------------------------------------------===//
// MHLO combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or floating-point tensor types

// Any integer or floating-point tensor types
defvar MHLO_IntOrFpTensor = HLO_IntOrFpTensor;

// Any integer or predicate tensor types
defvar MHLO_PredOrIntTensor = HLO_PredOrIntTensor;

// Any floating-point or complex tensor types
defvar MHLO_FpOrComplexTensor = HLO_FpOrComplexTensor;

// Any floating-point, complex or quantized tensor types
defvar MHLO_FpComplexOrQuantizedIntTensor = HLO_FpComplexOrQuantizedIntTensor;

// Any int, floating-point, complex or quantized tensor types
defvar MHLO_IntFpOrComplexOrQuantizedIntTensor = RankedTensorOf<[HLO_Int, HLO_Float, HLO_Complex, HLO_QuantizedInt, HLO_PerAxisQuantizedInt]>;

// Any pred, int or floating-point tensor types
defvar MHLO_PredIntOrFpTensor = HLO_PredIntOrFpTensor;

// Any pred, int, floating-point or quantized tensor types
defvar MHLO_PredIntFpOrQuantizedTensor = HLO_PredIntFpOrQuantizedTensor;

//===----------------------------------------------------------------------===//
// MHLO static shape type definitions.
//===----------------------------------------------------------------------===//

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
defvar MHLO_StaticShapeTensor = HLO_StaticShapeTensorOrPerAxisQuantizedTensor;

defvar MHLO_StaticShapeTensorOrToken = HLO_StaticShapeTensorOrPerAxisQuantizedTensorOrToken;

defvar MHLO_StaticShapeIntOrFpTensor = HLO_StaticShapeIntOrFpTensor;

defvar MHLO_StaticShapeIntFpOrComplexTensor = HLO_StaticShapeIntFpOrComplexTensor;

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_BASE
