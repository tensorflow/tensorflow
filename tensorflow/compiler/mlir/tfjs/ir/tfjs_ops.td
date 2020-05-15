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

//===----------------------------------------------------------------------===//
//
// This is the operation definition file for TensorFlow.js dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef TFJS_DIALECT
#define TFJS_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// TensorFlow.js dialect definitions
//===----------------------------------------------------------------------===//

def TFJSDialect : Dialect {
  let name = "tfjs";

  let summary = "Types and operations for TensorFlow.js dialect";
  let description = [{
    This dialect contains operations for TensorFlow.js. This dialect will be
    used in conjunction with the TensorFlow dialects for converting & optimizing
    TF graphs to be deployed on TFJS.
  }];

  let cppNamespace = "tfjs";
}

//===----------------------------------------------------------------------===//
// TensorFlow.js op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class TFJS_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFJSDialect, mnemonic, traits>;

def TFJS_PReluOp : TFJS_Op<"Prelu", [NoSideEffect, ResultsBroadcastableShape,
                                     SameOperandsAndResultElementType]> {
  let summary = "Parametric Rectified Linear Unit operator";
  let description = [{
    Element-wise PReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$alpha);
  let results = (outs AnyTensor:$output);
  let assemblyFormat =
    " operands attr-dict `:` `(` type(operands) `)` `->` type($output)";
}
#endif // TFJS_DIALECT
