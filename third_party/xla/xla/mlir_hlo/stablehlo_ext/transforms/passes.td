/* Copyright 2023 The StableHLO Authors.

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

def StablehloCanonicalizeDynamismPass : Pass<"stablehlo-ext-canonicalize-dynamism", "func::FuncOp"> {
  let summary = "Canonicalizes dynamic StableHLO ops into static ops, with some XLA specialization.";
  let description = [{
    Experimental version of the --stablehlo-canonicalize-dynamism pass.
  }];
  let dependentDialects = ["mlir::chlo::ChloDialect"];
}

def StablehloRefineShapesPass : Pass<"stablehlo-ext-refine-shapes", "ModuleOp"> {
  let summary = "Refines shapes across a StableHLO program, with some XLA specialization.";
  let description = [{
    Experimental version of the --stablehlo-refine-shapes pass.
  }];
}

def ChloRecomposeOpsPass : Pass<"stablehlo-ext-chlo-recompose-ops", "ModuleOp"> {
  let summary = "(Experimental) Recompose CHLO ops serialized as custom calls.";
  let description = [{
    Experimental version of CHLO serialization support.
  }];
  let dependentDialects = ["chlo::ChloDialect"];
}
