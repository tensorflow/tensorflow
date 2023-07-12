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


#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SpmdExpandableInterfaceVerificationPass
    : Pass<"spmd-expandable-interface-verification", "mlir::ModuleOp"> {
  let summary = "Verify all ops have IFRT SpmdExpandableInterface implemented";

  let description = [{
    This pass verifies all ops appearing in atom programs have implemented the
    IFRT SpmdExpandableInterface. This makes sure the interface methods have
    implementation before they are being called in later passes.

    This pass will fail if it encounters any op without the implementation.

    When the atom program is invoked on 1 device, the verification is skipped as
    no expansion is needed.
  }];

  let constructor = "CreateSpmdExpandableInterfaceVerificationPass()";

  let options = [
    ListOption<"excluded_dialects_", "excluded-dialects", "std::string",
       "Comma-separated list of MILR dialect namespaces that are excluded from checking">
  ];
}

def IfrtDuplicatedCalleeEliminationPass
    : Pass<"ifrt-duplicated-callee-elimination", "mlir::ModuleOp"> {
  let summary = "Deduplicate callees of CallOp";
  let description = [{
    Deduplicate `ifrt.CallOp` callee `FuncOp`s that are the same except their
    symbol names. The callee attrs of `ifrt.CallOp`s will be merged to one of
    them. The duplicated callee `FuncOp` will not be removed.
  }];

  let constructor = "CreateIfrtDuplicatedCalleeEliminationPass()";
}

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_
