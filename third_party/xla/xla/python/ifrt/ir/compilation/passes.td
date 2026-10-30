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

#ifndef XLA_PYTHON_IFRT_IR_COMPILATION_PASSES_TD_
#define XLA_PYTHON_IFRT_IR_COMPILATION_PASSES_TD_

include "mlir/Pass/PassBase.td"

def IfrtIRCompileOptions :
    Option<"compile_options", "compile_options",
           "std::shared_ptr<xla::ifrt::IfrtIRCompileOptions>",
           "std::make_shared<xla::ifrt::IfrtIRCompileOptions>()",
           "The IFRT IR compile options used for compilation.">;

def IfrtLowerAtomProgramMetadataToXlaPass :
    Pass<"ifrt-lower-atom-program-metadata-to-xla", "mlir::ModuleOp"> {
  let summary = "Converts IFRT metadata to XLA-compatible attributes";
  let description = [{
Replaces `ifrt.sharding` attr by `mhlo.sharding` attr (if Shardy is disabled)
and `ifrt.memory_kind` by `mhlo.memory_kind`.

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

  let options = [
    IfrtIRCompileOptions
  ];
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

  let options = [
    ListOption<"platform_names", "platform_names", "std::string",
               "The mapping from logical device ID to platform name (e.g., tpu, cuda) corresponding to the logical device ID.">,
    IfrtIRCompileOptions,
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

  let options = [
    ListOption<"platform_names", "platform_names", "std::string",
               "The mapping from logical device ID to platform name (e.g., tpu, cuda) corresponding to the logical device ID.">
  ];
}

def IfrtToDotPass :
   Pass<"ifrt-to-dot", "mlir::ModuleOp"> {
  let summary = "Extracts a dot graph representation of an IFRT IR program.";
  let description = [{
    The pass dumps a dot graph graph of a module. `ifrt.LoadedExecutableOp`s are
    represented as nodes. An edge is added between a source and destination node
    if the destination node has a control dependency to execute after the source
    node or uses an output of the source node directly, or indirectly via a
    `ifrt.CopyArraysOp`.

    Note: Running this pass breaks asynchronous compilation since the pass needs
    to query executable information, making IFRT IR compilation much more
    expensive. Therefore, This pass must be enabled only for debugging purposes.
  }];
  let options = [
    Option<"dot_graph_dump_to", "dot_graph_dump_to", "std::string", "",
           "The directory to dump the dot graph to.">,
    Option<"dot_graph_min_executable_peak_memory_bytes",
           "dot_graph_min_executable_peak_memory_bytes", "int64_t", "0",
           "The minimum peak memory (in bytes) an executable node must have to "
           "be included in the dot graph.">,
    Option<"dot_graph_min_executable_flops", "dot_graph_min_executable_flops",
           "float", "0.0",
           "The minimum flops an executable node must have to be included in the "
           "dot graph.">,
    Option<"dot_graph_min_per_device_transfer_size_bytes",
           "dot_graph_min_per_device_transfer_size_bytes", "int64_t", "0",
           "The minimum per device transfer size (in bytes) an edge between "
           "executables must have to be included in the dot graph.">,
  ];
}

#endif  // XLA_PYTHON_IFRT_IR_COMPILATION_PASSES_TD_
