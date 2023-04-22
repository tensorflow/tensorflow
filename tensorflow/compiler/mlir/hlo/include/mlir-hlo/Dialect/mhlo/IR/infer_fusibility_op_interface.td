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

// This file contains inferFusiblityOpInterface, which is used to guide
// fusion decision.

#ifndef MLIR_INFER_FUSIBILITY_OP_INTERFACE
#define MLIR_INFER_FUSIBILITY_OP_INTERFACE

include "mlir/IR/OpBase.td"

// OpInterface to query if an op is fusible and to query the shape equality
// constraint among the inputs and outputs of an op.
def InferFusibilityOpInterface : OpInterface<"InferFusibilityOpInterface"> {
  let description = [{
    Interface to query if an op is fusible and to query the shape equality
    constraint among the inputs and outputs of an op.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{If true, this op can be fused with its operands
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isFusibleWithOperand",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        /// Returns whether this op can be fused with its operands
        return true;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{If true, this op can be fused with its consumers
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isFusibleWithConsumer",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        /// Return whether this op can be fused with its consumers
        return true;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return whether two inputs have the same shape (assuming no"
               "implicit broadcasting).",
      /*retTy=*/"bool",
      /*methodName=*/"inferInputsShapeEquality",
      /*args=*/(ins "int":$lhs, "int":$rhs),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        /// Return whether two inputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return InferFusibilityOpInterface::inferShapeEquality(op->getOperand(lhs), op->getOperand(rhs));
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return whether two outputs have the same shape (assuming no"
               " implicit broadcasting).",
      /*retTy=*/"bool",
      /*methodName=*/"inferOutputsShapeEquality",
      /*args=*/(ins "int":$lhs, "int":$rhs),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        /// Return whether two outputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs >= 0 && rhs >= 0);
        if (lhs == rhs) return true;
        return InferFusibilityOpInterface::inferShapeEquality(op->getResult(lhs), op->getResult(rhs));
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return whether the input and the output have the same"
               " shape (assuming no implicit broadcasting).",
      /*retTy=*/"bool",
      /*methodName=*/"inferInputOutputShapeEquality",
      /*args=*/(ins "int":$input, "int":$output),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        /// Return whether the input and the output have the same shape.
        Operation *op = this->getOperation();
        assert(input >= 0 && output >= 0);
        return InferFusibilityOpInterface::inferShapeEquality(op->getOperand(input), op->getResult(output));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{Return the effective workload shape for the operation.

      Here the effective workload shape roughly represents the maximum
      parallelism can be used during the codegen stage. It's used to check
      the shape-compatibility of the operation. During fusion, we only
      try to fuse shape-compatible ops for performance.
      For example, the effective workload shape of an elementwise op is its
      output shape, while the effective workload shape of a reduction op may
      be its operand shape.
      Return None if such an inference is not possible.
      }],
      /*retTy=*/"llvm::Optional<Value>",
      /*methodName=*/"inferEffectiveWorkloadShape",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        /// Return effective workload size if possible, otherwise None.
        return {};
      }]
    >,
  ];

  let extraClassDeclaration = [{
    // Returns whether the given values have the same static shape.
    static bool inferShapeEquality(Value first, Value second) {
      // If both lhs and rhs have static shapes, check them directly.
      auto first_ty = first.getType().dyn_cast<RankedTensorType>();
      auto second_ty = second.getType().dyn_cast<RankedTensorType>();
      if (!first_ty || !first_ty.hasStaticShape() ||
          !second_ty || !second_ty.hasStaticShape() ||
          first_ty.getRank() != second_ty.getRank()) {
        return false;
      }
      return first_ty.getShape() == second_ty.getShape();
    }
  }];
}

#endif
