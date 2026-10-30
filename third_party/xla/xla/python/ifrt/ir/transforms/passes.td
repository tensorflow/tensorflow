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

def IfrtAddCtrlDependenciesPass :
    Pass<"ifrt-add-ctrl-dependencies", "mlir::func::FuncOp"> {
  let summary = "Adds control dependencies to ifrt.CallOps.";
  let description = [{
Adds control dependencies to `ifrt.CallOp` based on program order.
  }];
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
}

def IfrtDuplicatedCalleeEliminationPass
    : Pass<"ifrt-duplicated-callee-elimination", "mlir::ModuleOp"> {
  let summary = "Deduplicate callees of CallOp";
  let description = [{
Deduplicate `ifrt.CallOp` callee `FuncOp`s that are the same except their
symbol names. The callee attrs of `ifrt.CallOp`s will be merged to one of
them. The duplicated callee `FuncOp` will not be removed.
  }];
}

def IfrtInsertCopyArraysForReturnedManyTimesPass :
    Pass<"ifrt-insert-copy-arrays-for-returned-many-times", "mlir::func::FuncOp"> {
  let summary = "Add CopyArrays for program arguments that are returned.";
  let description = [{
Insert CopyArrays ops for program arguments that are returned. These CopyArrays
op always set reuse=false, and set donated=true only on the last return of a
donated input argument. Thus, IFRT programs provide similar semantics as XLA.

```mlir
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
func.func @main(%arg0: !array, %arg1: !array {ifrt.donated}) -> (!array, !array) {
  return %arg0, %arg1: !array, !array
}
```

will be replaced by:

```mlir
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
func.func @main(%arg0: !array, %arg1: !array {ifrt.donated}) -> (!array, !array) {
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array) -> !array
  %1, %ctrl_1 = ifrt.CopyArrays(%arg1) {donated=true} : (!array) -> (!array)
  return %0, %1: !array, !array
}
```
  }];
}


def IfrtMergeCopiesAndReshardsPass :
    Pass<"ifrt-merge-copies-and-reshards", "mlir::func::FuncOp"> {
  let summary = "Merge copies and reshards into a single op.";
  let description = [{
Merges copies and reshards from a source op which flow into the same destination
op, ignoring control dependencies.

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
the outputs of another ifrt op.
  }];
}

def IfrtVerifyDonationPass :
    Pass<"ifrt-verify-donation", "mlir::func::FuncOp"> {
  let summary = "Verify that `!ifrt.array` are not donated more than once.";
  let description = [{
Verifiy that no `!ifrt.array` is donated more than once, and that all
`ifrt.array`s that are donated and arguments of the main func have
`ifrt.donated` attribute.
  }];
}

def IfrtVerifyShardingSpecifiedPass :
    Pass<"ifrt-verify-sharding-specified", "mlir::func::FuncOp"> {
  let summary = "Verify that all `!ifrt.array` have sharding specified.";
  let description = [{
Verify that each `!ifrt.array` has sharding attribute that is not of type
`!ifrt.sharding_unspecified`.
  }];
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
}

def IfrtDumpAtomProgramsPass
    : Pass<"ifrt-dump-atom-programs", "mlir::ModuleOp"> {
  let summary = "Extracts atom programs from module and dumps them to files.";
  let description = [{
The pass dumps:
1. Unmodified atom program ModuleOps (one per file).
2. A modified version of the program with all the non-IFRT function bodies
  removed. This offers a more compact representation of programs that is useful
  for debugging.
  }];
  let options = [
    Option<"dump_dir", "dump_dir", "std::string", "",
           "The directory to dump the atom programs and the main function to.">,
  ];
}

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_TD_
