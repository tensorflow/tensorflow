/* Copyright 2026 The OpenXLA Authors.

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

def StablehloCanonicalizeFromHloImportPass : Pass<"stablehlo-ext-canonicalize-from-hlo-import", "mlir::func::FuncOp"> {
  let summary = "Simplify StableHLO imported from HLO";

  let dependentDialects = ["stablehlo::StablehloDialect"];

  let description = [{
    This pass simplifies StableHLO imported from HLO. This pass is a subset of
    the graph simplification passes and is intended to bring the imported HLO
    back to a more canonical form without applying a full set of graph
    simplifications.

    Namely, this pass:
    * Simplifies tuples, undoing `tuple(get_tuple_element)` and
      `get_tuple_element(tuple)`.
    * Converts WhileOp explicit captured constants to implicit captures.
    * Flattens tuples in operands and results of operators that support both
      tuple and variadic type.
    * Flattens tuples in entry function of the module.
  }];

  let options = [
    Option<"entryFunctionNameOption", "entry-function", "std::string",
           /*default=*/[{"main"}], "the name of entry function of the module">,
  ];
}
