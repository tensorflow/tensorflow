/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
include "stablehlo/dialect/StablehloOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"

def CloneF32ElementsAttrWithOnes
  : NativeCodeCall<"DenseElementsAttr::get(llvm::cast<ShapedType>($0.getType()), (float)1.0)">;

def NotConstant : Constraint<
    CPred<"llvm::isa<BlockArgument>($0) || !llvm::isa<stablehlo::ConstantOp>($0.getDefiningOp())">,
    "Is not a constant.">;

def : Pat<(StableHLO_DivOp $l,
            (StableHLO_ConstantOp:$divisor FloatElementsAttr<32>:$cst)),
          (StableHLO_MulOp $l,
            (StableHLO_DivOp
              (StableHLO_ConstantOp (CloneF32ElementsAttrWithOnes $cst)),
              $divisor)),
          [(NotConstant $l)]>;




