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

def IfrtOutlineAtomProgramToModulePass :
    Pass<"ifrt-outline-atom-program-to-module", "mlir::ModuleOp"> {
  let summary = "Wraps every atom function with a ModuleOp with a @main FuncOp";
  let description = [{
For every unique atom program this passes produces a ModuleOp with the same name
as the callee, clones the callee into the ModuleOp, and redirects all the
CallOps calling it to the new callee.

This pass must be run if the compiler (e.g., the XLA compiler) expects each atom
program to be outlined in a ModuleOp with a @main FuncOp.

For example, the following code

```mlir
%0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0, 1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

func.func private @callee(%arg0: tensor<2x2xi32>) -> (tensor<4x4xi32>) {}
```

will be replaced by

```mlir
%0, %ctrl_0 = ifrt.Call @callee::@main(%arg0) on devices [0, 1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

module @callee attributes {sym_visibility = "private"} {
  func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<4x4xi32>) {}
}
```
  }];

  let constructor = "CreateIfrtOutlineAtomProgramToModulePass()";
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

def IfrtMergeReshardsPass :
    Pass<"ifrt-merge-reshards", "mlir::func::FuncOp"> {
  let summary = "Merge reshards for each (src, dst) pair into a single reshard.";
  let description = [{
Merges reshards from a source op which flow into the same destination op,
ignoring control dependencies.

E.g.

```mlir
%c:4, %ctrl_0  = ifrt.Call ...
%r0, %ctrl_1 = ifrt.Reshard %c#0 ...
%r1, %ctrl_2 = ifrt.Reshard %c#1 ...
%d:4, %ctrl_3  = ifrt.Call (%r0, %r1)
```

will be replaced by:

```mlir
%c:4, %ctrl_0 = ifrt.Call ...
%r:2, %ctrl_1 = ifrt.Reshard (%c#0, %c#1)
%d:4, %ctrl_2 = ifrt.Call (%r#0, %r#1)
```

Currently this handles the case where the source is the Func BlockArgs or
the outputs of a ifrt.Call, ifrt.CallLoadedExecutable, ifrt.Reshard,
ifrt.CopyArrays, and ifrt.RemapArrays.
  }];

  let constructor = "CreateIfrtMergeReshardsPass()";
}

def IfrtVerifyDonationPass :
    Pass<"ifrt-verify-donation", "mlir::func::FuncOp"> {
  let summary = "Verify that `!ifrt.array` are not donated more than once.";
  let description = [{
Verifiy that no `!ifrt.array` is donated more than once, and that all
`ifrt.array`s that are donated and arguments of the main func have
`ifrt.donated` attribute.
  }];

  let constructor = "CreateIfrtVerifyDonationPass()";
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

def IfrtPopulateAtomProgramMetadataPass :
    Pass<"ifrt-populate-atom-program-metadata", "mlir::ModuleOp"> {
  let summary = "Populate metadata from call site to atom functions";
  let description = [{
For every CallOp, this pass
  1. clones the callee's parent ModuleOp
  2. adds `ifrt.num_devices` attribute to the callee's parent ModuleOp
  2. attaches shardings and devices to the inputs and outputs of the callee's
     main FuncOp
  3. attaches `tf.aliasing_output` attr to the callee main FuncOp's inputs
     according to `io_aliases`
  4. attaches `jax.buffer_donor` attr to the callee main FuncOp's inputs
     according to `donated_input_indices`

For CallOps with the same callee, a different clone will be created for each
CallOp, even if the populated metadata are the same. User may want to run
`ifrt-duplicated-callee-elimination` pass to dedup the clones.

For example, the following code

```mlir
%0, %ctrl_0 = ifrt.Call @callee::@main(%arg0) on devices [0, 1]
    {io_aliases=[array<i32: 1, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                   [0,1]>,
       !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

module @callee {
  func.func private @main(%arg0: tensor<2x2xi32>, %arg1: tensor<4x4xi32>)
      -> tensor<4x4xi32> {}
}
```

will be replaced by

```mlir
%0, %ctrl_0 = ifrt.Call @new_callee::@main(%arg0) on devices [0, 1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

module @new_callee attributes {ifrt.num_devices = 2} {
  func.func private @new_callee(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.devices = #ifrt<devices[0,1]>},
      %arg1: tensor<4x4xi32> {
        ifrt.sharding = #ifrt.sharding_param<1x2 to [0] on 2>,
        ifrt.devices = #ifrt<devices[0,1]>
        tf.aliasing_output = 0 : i32})
      -> (tensor<4x4xi32> {
        ifrt.sharding = #ifrt.sharding_param<1x2 to [0] on 2>,
        ifrt.devices = #ifrt<devices[0,1]>})
      {}
}
```
  }];

  let constructor = "CreateIfrtPopulateAtomProgramMetadataPass()";
}

def IfrtReshardToCopyArraysPass :
    Pass<"ifrt-reshard-to-copy-arrays", "mlir::ModuleOp"> {
  let summary = "Replaces `ifrt.Reshard` with `ifrt.CopyArrays`";
  let description = [{
Replaces each `ifrt.Reshard` op with an `ifrt.Reshard` op with inputs only
the arrays that are being resharded, and several `ifrt.CopyArrays` ops to copy
the arrays that are not being resharded. An `ifrt.CopyArrays` op is added for
unique output `ifrt.DevicesAttr`.

For example, the following code
```mlir
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [2,3]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [2,3]>
func.func @main(%arg0: !array0, %arg1: !array1) -> (!array1, !array2)
    attributes {ifrt.function} {
  %0, %1, %ctrl_0 = ifrt.Reshard(%arg0, %arg1)
      : (!array0, !array1) -> (!array1, !array2)
  return %0, %1: !array1, !array2
}
```

will be replaced by:

```mlir
func.func @main(%arg0: !array0, %arg1: !array1) -> (!array1, !array2)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Reshard(%arg1) : (!array1) -> !array2
  %1, %ctrl_1 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  return %0, %1: !array1, !array2
}
```
  }];

  let constructor = "CreateIfrtReshardToCopyArraysPass()";
}

def IfrtLowerAtomProgramMetadataToXlaPass :
    Pass<"ifrt-lower-atom-program-metadata-to-xla", "mlir::ModuleOp"> {
  let summary = "Converts IFRT metadata to XLA-compatible attributes";
  let description = [{
Replaces `ifrt.sharding` attr by `mhlo.sharding` attr, and `ifrt.memory_kind`
by `mhlo.memory_kind`.

For example, the following code

```mlir
module attributes {
  func.func @main(
    %arg0: tensor<2x2xi32> {
      ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>})
    -> (tensor<2x2xi32> {
      ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>}) {}
}
```

will be replaced by

```mlir
module attributes {
  func.func @main(
    %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
    -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {}
}
```
  }];

  let constructor = "CreateIfrtLowerAtomProgramMetadataToXlaPass()";
}

def IfrtRemoveIfrtAttrsPass :
    Pass<"ifrt-remove-ifrt-attrs", "mlir::ModuleOp"> {
  let summary = "Remove IFRT attrs from modules and functions of atom programs";
  let description = [{
Remove IFRT-related attributes from module and functions' arguments and
results.

This is necessary for the compilation of TfSpmd program.

For example, the following code

```mlir
module {
  func.func @main(
      %arg0: tensor<2x2xi32> {
          ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 1>})
      -> (tensor<2x2xi32> {
          ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 1>}) {...}
}
```

will be replaced by

```mlir
module {
  func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {...}
}
```
  }];

  let constructor = "CreateIfrtRemoveIfrtAttrsPass()";
}

def IfrtRemoveAttrsFromOtherDialectsPass :
    Pass<"ifrt-remove-attrs-from-other-dialects", "mlir::ModuleOp"> {
  let summary = "Remove attrs from other dialects in IFRT functions";
  let description = [{
Remove attributes that are not from IFRT or Builtin dialect from IFRT functions.
Does not modify atom programs.
  }];
  let constructor = "CreateIfrtRemoveAttrsFromOtherDialectsPass()";
}

def IfrtLowerMpmdReshardToCallPass :
    Pass<"ifrt-lower-mpmd-reshard-to-call", "mlir::ModuleOp"> {
  let summary = "Lowers MPMD `ifrt.Reshard` to `ifrt.Call`";
  let description = [{
Replaces each MPMD `ifrt.Reshard` with an `ifrt.Call` to a newly created
module. The module has a main function with a `ifrt.reshard_function` attribute,
and that comprises of the `ifrt.Reshard` ops part of the MPMD reshard.

For example, the following code
```mlir
!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
func.func @main(%arg0: !array0) -> (!array1) {
  %0, %ctrl = ifrt.Reshard(%arg0) : (!array0) -> !array1
  return %0: !array1
}
```

will be replaced by

```mlir
!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
func.func @main(%arg0: !array0) -> (!array1) {
  %0, %ctrl = ifrt.Call(%arg) {ifrt.module_type = "mpmd_reshard"}
      : (!array0) -> !array1
  return %0: !array1
}

module @reshard attributes {
    ifrt.num_devices = 3,
    sym_visibility = "private"} {
  func.func @main(%arg0: !array0) -> !array1
      attributes {ifrt.reshard_function} {
    %0, %ctrl = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }
}
```
  }];

  let constructor = "CreateIfrtLowerMpmdReshardToCallPass()";
}

def IfrtPrecompileAtomProgramPreprocessingPass :
    Pass<"ifrt-precompile-atom-program-preprocessing", "mlir::ModuleOp"> {
  let summary = "Prepares atom modules for compilation depending on their type";
  let description = [{
This passes visits each CallOp and does the following:
1. it sets `ifrt.module_type` attribute based on the platform name of the
 devices used by the atom program at compilation time.
2. depending on the module type it invokes a different preprocessing pipeline on
 the atom program module to prepare it for compilation (e.g., set MHLO shardings
 for XLA compilation).
```
  }];

  let constructor = "CreateIfrtPrecompileAtomProgramPreprocessingPass()";
  let options = [
    ListOption<"platform_names", "platform_names", "std::string",
               "The mapping from logical device ID to platform name (e.g., tpu, cuda) corresponding to the logical device ID.">
  ];
}

def IfrtVerifyDeviceTypeConsistencyPass :
    Pass<"ifrt-verify-device-type-consistency", "mlir::ModuleOp"> {
  let summary = "Verify the device types in `ifrt.CallOp` to ensure consistency.";
  let description = [{
Verify that each `ifrt.CallOp` uses devices of a single platform type and the
platform type is consistent with the dialects the callee is using.


This pass fails if
1. any `ifrt.CallOp` uses more than one platform type.
2. any callee of `ifrt.CallOp` has a mixture of MHLO and TF dialects.
3. the platform type and dialect in `ifrt.CallOp` callee is inconsistent with
 each other.
  }];

  let constructor = "CreateIfrtVerifyDeviceTypeConsistencyPass()";
  let options = [
    ListOption<"platform_names", "platform_names", "std::string",
               "The mapping from logical device ID to platform name (e.g., tpu, cuda) corresponding to the logical device ID.">
  ];
}

def IfrtLegalizeToVifrtPass : Pass<"ifrt-legalize-to-vifrt", "mlir::ModuleOp"> {
  let summary = "Legalize IFRT to VIFRT.";
  let dependentDialects = ["xla::ifrt::VifrtDialect"];
}

def VifrtLegalizeToIfrtPass : Pass<"vifrt-legalize-to-ifrt", "mlir::ModuleOp"> {
  let summary = "Legalize VIFRT to IFRT.";
  let dependentDialects = ["xla::ifrt::IfrtDialect"];
}

def VifrtToVersionPass : Pass<"vifrt-to-version", "mlir::ModuleOp"> {
  let summary = "Convert between versions of VIFRT.";
  let options = [
    Option<"target_version", "target_version", "std::string", "",
           "The VIFRT target version. Must be a version of the form #.#.#.">,
  ];
  let constructor = "CreateVifrtToVersionPass()";
  let dependentDialects = ["xla::ifrt::VifrtDialect"];
}

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_
