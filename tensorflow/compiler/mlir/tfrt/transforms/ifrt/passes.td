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
 }

def SinkVariableAsNamedArrayPass: Pass<"sink-variable-as-named-array", "mlir::ModuleOp"> {
  let summary = "Sink variable tensor for tpu device as named IFRT array for tf.IfrtCall";

  let description = [{
    This pass sinks variable tensor argument to `tf.IfrtCall` as variable_arg_indices
    and variable_names attributes and also lowers `tf.ReadVariableOp` to
    `tf.IfrtLoadVariableOp`.

    The runtime ensures that `tf.IfrtCall` kernel can bind the IFRT array by
    its name as input to the TPU program.

 }];

  let constructor = "CreateSinkVariableAsNamedArrayPass()";
}

def LowerToIfrtRestoreVariablePass: Pass<"lower-to-ifrt-restore-variable", "mlir::ModuleOp"> {
  let summary = "Lowers tf.RestoreV2 to tf.IfrtRestoreVariable";

  let description = [{
    This pass rewrites `RestoreV2Op` and `AssignVariableOp` to `IfrtRestoreVariableOp`.

    This pass is ran so that the runtime may run restore operation asynchronously
    such that the expensive resotre operation can be parallerized with other 
    model compilation operation.
  }];

  let constructor = "CreateLowerToIfrtRestoreVariablePass()";
 }

def TfRestorePruningPass
    : Pass<"tf-restore-pruning", "mlir::func::FuncOp"> {
  let summary = "Prune unused`tf.RestoreV2` ops";

  let description = [{
    This pass prune unused `tf.RestoreV2` op. A typical use case is to combine
    `TfRestoreSplittingPass`, this pass and `TfRestoreMergingPass` in sequence
    so that the un-used restored tensors are not read into host memory.
  }];

  let constructor = "CreateTfRestorePruningPass()";
}


def TfRestoreSplittingPass
    : Pass<"tf-restore-splitting", "mlir::func::FuncOp"> {
  let summary = "Splits `tf.RestoreV2` ops";

  let description = [{
    This pass splits each `tf.RestoreV2` op so that one restore op handles one
    variable only. This pass can split restore ops only if the tensor names and
    the shape/slices arguments are constants, which is usually the case.

    Splitting monolithic restore ops into per-tensor restore ops makes it easier
    to shard SavedModel initialization across multiple clusters.
  }];

  let constructor = "CreateTfRestoreSplittingPass()";
}

def TfRestoreMergingPass : Pass<"tf-restore-merging", "mlir::func::FuncOp"> {
  let summary = "Merges `tf.RestoreV2` ops";

  let description = [{
    This pass merges multiple `tf.RestoreV2` ops into one `tf.RestoreV2` op
    using variadic results. The current implementation merges restore ops only
    if they have the same `prefix` and have constant tensor names and
    shape/slice arguments.

    This pass is run in order to undo `tf-restore-splitting` after cluster
    formation and reduce the op dispatch overhead.
  }];

  let constructor = "CreateTfRestoreMergingPass()";
}

def TfIdentityPropagationPass
    : Pass<"tf-identity-propagation", "mlir::func::FuncOp"> {
  let summary = "Propagates inputs of no-op identity ops to their outputs";

  let description = [{
    This pass finds identity ops that are no-op and propagates their inputs
    directly to outputs so that identity ops can be skipped.

    One example of identity ops that are not no-op is identity ops with XLA
    sharding annotation. Since some models use identity ops with `_XlaSharding`
    attributes to change output sharding, this pass doesn't propagate the inputs
    of such identity ops in order to preserve the sharding changes.

    This pass is useful to make sure that ineffective identity ops don't affect
    the graph partitioning. For example, in a pipelined model, if there is a CPU
    identity op between two TPU computation stages (which sometimes happens
    because TensorFlow inserts it), this will unnecessarily route the
    intermediate tensors through the CPU device. By forwarding the inputs of the
    identity op directly to its outputs, we can avoid such inefficiency.
  }];

  let constructor = "CreateTfIdentityPropagationPass()";
}

def TfDeviceCleanupPass : Pass<"tf-device-cleanup", "mlir::func::FuncOp"> {
  let summary = "Cleans up device attributes from all ops";

  let description = [{
    This pass removes `device` attributes from all TF ops. Some Serving
    doesn't rely on `device` attributes from SavedModel.
  }];

  let constructor = "CreateTfDeviceCleanupPass()";
}

