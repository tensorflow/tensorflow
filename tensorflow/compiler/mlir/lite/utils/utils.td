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

// Utility predicates that are shared by multiple passes.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

// Checks value is not produced by a TFL_Quant or
// from TFL_Quant Op with same quant type.
def NotFromQuantOpOrSameQuantType : Constraint<
  CPred<"tflite::NotFromQuantOpOrSameQuantType($0,$1)">>;

// Accepts two inputs and check if both have the same element type.
def SameElementType : Constraint<
  CPred<"getElementTypeOrSelf($0) == getElementTypeOrSelf($1)">>;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;
