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

def RewriteClusterToIfrtCallPass: Pass<"rewrite-cluster-to-ifrt-call", "mlir::ModuleOp"> {
  let summary = "Lowers tf_device.cluster_func to tf.ifrt_call";

  let description = [{
    This pass converts the clustered tpu program into an IFRT program and rewrites 
    tf_device.cluster_fun to tf.ifrt_call.
  }];

  let dependentDialects = ["mlir::tf_device::TensorFlowDeviceDialect"];

  let constructor = "CreateRewriteClusterToIfrtCallPass()";

  let options = [
    Option<"tpu_compile_metadata_debug_", "tpu-compile-metadata-debug", "bool", "false",
          "if enabled, output compile metadata as readable string in "
         "an extra __tpu_compile_metadata_debug attribute for debug">,
  ];

}

def SinkVariableAsNamedArrayPass: Pass<"sink-variable-as-named-array", "mlir::ModuleOp"> {
  let summary = "Sink variable tensor for tpu device as named IFRT array";

  let description = [{
    This pass converts applicable `tf.AssignVariableOp` to `tf.IfrtLoadVariableOp`. 

    A `tf.AssignVariableOp` can be lowered to `tf.IfrtLoadVariableOp` if its variable tensor
    is used as an input to `tf.IfrtCall` TPU program. Additionally, the corresponding tensor
    will be sunk as an attribute from `tf.IfrtCall`'s argument list because the runtime
    ensures that `tf.IfrtCall` kernel can bind the IFRT array by its name as input to the TPU
    program.

    If a variable tensor is exclusively used for TPU devices, then its defining
    `tf.VarHandleOp`, `tf.ReadVariableOp` will be pruned.
  }];

  let constructor = "CreateSinkVariableAsNamedArrayPass()";
}
