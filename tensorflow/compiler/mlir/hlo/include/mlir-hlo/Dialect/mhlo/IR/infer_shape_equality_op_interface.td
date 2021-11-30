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

#ifndef MLIR_INFER_SHAPE_EQUALITY_OP_INTERFACE
#define MLIR_INFER_SHAPE_EQUALITY_OP_INTERFACE

include "mlir/IR/OpBase.td"

// OpInterface to query the shape equality constraint among the inputs and
// outputs of an op.
def InferShapeEqualityOpInterface : OpInterface<"InferShapeEqualityOpInterface"> {
  let description = [{
    Interface to query if an op is fusible and to query the shape equality
    constraint among the inputs and outputs of an op.
  }];

  let methods = [
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
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getOperand(lhs), op->getOperand(rhs));
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
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getResult(lhs), op->getResult(rhs));
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
        return InferShapeEqualityOpInterface::inferShapeEquality(op->getOperand(input), op->getResult(output));
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
