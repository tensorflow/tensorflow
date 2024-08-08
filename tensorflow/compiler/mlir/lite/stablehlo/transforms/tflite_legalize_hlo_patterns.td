/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mhlo/IR/hlo_ops.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

def CreateTFLCastToInt32Op : NativeCodeCall<
  "CreateCastToInt32($0, $_loc, $_builder)">;

def LegalizeTranspose : Pat<(MHLO_TransposeOp $arg, $perm),
                            (TFL_TransposeOp $arg,
                            (CreateTFLCastToInt32Op (TFL_ConstOp $perm)))>;

def LowerCbrt : Pat<(MHLO_CbrtOp $opr),
  (TFL_PowOp $opr,
    (TFL_DivOp
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_None)),
      [(F32Tensor $opr)]>;


//===----------------------------------------------------------------------===//
// logical and bitwise ops
//===----------------------------------------------------------------------===//

class GetRankedScalarAttr<string prefix, int width, string signed, string value> :
  NativeCodeCall<"DenseElementsAttr::get<" # prefix # "int" # width # "_t>("
    "RankedTensorType::get({}, $_builder.getIntegerType("
      # width # signed # "))," # value # ")">;

def : Pat<(MHLO_NotOp I1Tensor:$input), (TFL_LogicalNotOp $input)>;

// TFL does not support bitwise negation. not(x) is equivalant to xor(x, y) if
// y has a 1 in every bit position (xor(1, 1) = 0 and xor(0, 1) = 1).

// Signed: The 2s complement of -1 has a 1 in every bit position.
def : Pat<(MHLO_NotOp I8Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 8, "", "-1">)))>;

def : Pat<(MHLO_NotOp I16Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 16, "", "-1">)))>;

def : Pat<(MHLO_NotOp I32Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 32, "", "-1">)))>;


// Unsigned: 0xFFF... has a 1 in every bit position.
def : Pat<(MHLO_NotOp TensorOf<[UI8]>:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"u", 8, ", false", "0xFFU">)))>;

def : Pat<(MHLO_NotOp TensorOf<[UI16]>:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"u", 16, ", false", "0xFFFFU">)))>;

def : Pat<(MHLO_NotOp TensorOf<[UI32]>:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"u", 32, ", false", "0xFFFFFFFFUL">)))>;
