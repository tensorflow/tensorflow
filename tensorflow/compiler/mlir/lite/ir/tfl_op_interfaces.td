/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation interface definition file for TensorFlow Lite.

#ifndef TFL_OP_INTERFACES
#define TFL_OP_INTERFACES

include "mlir/IR/OpBase.td"

def TFL_Dialect : Dialect {
  let name = "tfl";

  let description = [{
    The TensorFlow Lite dialect.

    This dialect maps to TensorFlow Lite operations.

    Invariants:

    * All values are of Tensor type (in particular, scalars are
      represented using zero-dimensional tensors);
  }];

  let cppNamespace = "::mlir::TFL";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

// Attributes used for encoding sparse tensors.
// Please find detailed explanation of these parameters in the TFLite schema.
def TFL_DT_Dense : StrEnumAttrCase<"DENSE", 0>;
def TFL_DT_SparseCSR : StrEnumAttrCase<"SPARSE_CSR", 1>;

def TFL_DimensionTypeAttr : StrEnumAttr<
    "DimensionType", "dimension type", [TFL_DT_Dense, TFL_DT_SparseCSR]>;

//===----------------------------------------------------------------------===//
// TFL op interface for stateful operands.

def TFL_StatefulOp : OpInterface<"StatefulOpInterface"> {
  let description = [{
    Interface for ops that are stateful and need to identify stateful operands.

    Stateful operands correspond to TF's variables semantics. An op that has 1
    or more stateful operands is a stateful op.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the indices of stateful operands.}],
      "std::vector<int>", "GetStatefulOperands", (ins)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// TFL op interface for sparse operands.

def TFL_SparseOp : OpInterface<"SparseOpInterface"> {
  let description = [{
    Interface for ops that support sparse computation.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the indices of sparse operands.}],
      "std::vector<int>", "GetSparseOperands", (ins)
    >,
    InterfaceMethod<
      [{Returns the supported block size of float sparse operands.}],
      "std::vector<std::vector<int>>", "GetFloatBlockSize", (ins)
    >,
    InterfaceMethod<
      [{Returns the supported block size of quantized sparse operands.}],
      "std::vector<std::vector<int>>", "GetQuantizedBlockSize", (ins)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// TFL runtime type verification of operand/result types.

def TFL_RuntimeVerification : OpInterface<"TflRuntimeVerifyOpInterface"> {
  let description = [{
    Interface to verify TFLite runtime op verification.

    This verifies that the converted TFLite ops has operand/result type
    supported by the TFLite runtime.
  }];

  let methods = [
    StaticInterfaceMethod<
      [{Returns whether the op's operands/results are supported by runtime.}],
      "LogicalResult", "VerifyTflRuntimeConstraints",
      (ins "Operation*":$op, "bool":$emit_error_on_verify_fail)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// TFL arithmetic count interface.

def TFL_ArithmeticCount : OpInterface<"TflArithmeticCountOpInterface"> {
  let description = [{
    Interface for TFLite ops to calculate arithmetic count (Multiply-Add Count).
  }];

  let methods = [
    StaticInterfaceMethod<
      [{Returns an integer representing the op's arithmetic count (Multiply-Add
      Count), return -1 if the arithmetic count cannot be determined.}],
       "int64_t", "GetArithmeticCount", (ins "Operation*":$op)
    >,
  ];
}
#endif // TFL_OP_INTERFACES
