/* Copyright 2023 The OpenXLA Authors.

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


#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_

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

def SpmdExpansionPass : Pass<"spmd-expansion", "mlir::ModuleOp"> {
  let summary = "Rewrite each op from global view to per-shard view";

  let description = [{
This pass rewrites all ops in the module from their global view to per-shard
view so that the program can be executed with SPMD.

Prerequisites for this pass are:
1. The module has `ifrt.devices` attribute.
2. The module has an entry function at the top level whose name is either
"main" or otherwise annotated by a string attribute "ifrt.entry_function".
3. The entry function's arguments have `ifrt.devices` and `ifrt.sharding`
attributes.
4. All ops in the entry function and other functions called by the entry
function has `IfrtSpmdExpandable` interface implemented.

For example, the following:

```mlir
#device = #ifrt<devices[0,1]>
#sharding = #ifrt.sharding_param<2x1 to [0] on 2>
module attributes {ifrt.devices = #device} {
  func.func @main(
      %arg0: tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device})
      -> (tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device}) {
    return %arg0 : tensor<2x2xi32>
  }
}
```

will be transformed into:

```mlir
// The function's input and output shapes are now local.
#device = #ifrt<devices[0,1]>
#sharding = #ifrt.sharding_param<2x1 to [0] on 2>
module attributes {ifrt.devices = #device} {
  func.func @main(
      %arg0: tensor<1x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device})
      -> (tensor<1x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device}) {
    return %arg0 : tensor<1x2xi32>
  }
}
```
  }];

  let constructor = "CreateSpmdExpansionPass()";
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

def IfrtVerifyShardingSpecifiedPass :
    Pass<"ifrt-verify-sharding-specified", "mlir::ModuleOp"> {
  let summary = "Verify that all `!ifrt.array` have sharding specified.";
  let description = [{
Verify that each `!ifrt.array` has sharding attribute that is not of type
`!ifrt.sharding_unspecified`.
  }];

  let constructor = "CreateIfrtVerifyShardingSpecifiedPass()";
}


#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_
