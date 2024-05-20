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

include "mlir/Pass/PassBase.td"

def OutlineCompositesPass: Pass<"outline-composites", "func::FuncOp"> {
  let summary = "Outlines specific patterns into composites.";
  let description = [{
    Outline specific patterns into composites. Specific patterns can be any
    sub-DAG within a single `Block*`. The signature of the new composite
    matches the inupt and output edges from a node in the sub-DAG to a node out
    of it. The associated decomposition has the same semantic as the matched
    ops, but may not have identical structure.
  }];

  let options = [];
  let constructor = "CreateOutlineCompositesPass()";
  let dependentDialects = ["mlir::chlo::ChloDialect", "mlir::stablehlo::StablehloDialect", "mlir::func::FuncDialect"];
}

def SHLOSimplifyPass: Pass<"shlo-simplify", "ModuleOp"> {
  let summary = "Apply internal canonicalizations and foldings.";
  let description = [{
    Applies various internally defined patterns.
  }];

  let options = [

  ];
  let constructor = "CreateSHLOSimplifyPass()";
}