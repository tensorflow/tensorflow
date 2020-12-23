/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// TF dialect passes.

def TensorFlowShapeInferencePass : Pass<"tf-shape-inference", "ModuleOp"> {
  let summary = "Simple Shape Inference on TensorFlow Dialect";
  // TODO(jpienaar): Write `description`.

  let constructor = "TF::CreateTFShapeInferencePass()";

  let options = [
    Option<"max_iterations_", "max-iterations", "int64_t", /*default=*/"10",
           "Maximum shape inference iterations">
  ];
}

def ExecutorGraphPruningPass : FunctionPass<"tf-executor-graph-pruning"> {
  let summary = "Prunes unreachable ops in a tf_executor.graph";

  let description = [{
This pass removes ops from a `tf_executor.graph` that are not transitively, via
data or control dependencies, connected to the associated `tf_executor.fetch`
op. The order of ops will be preserved. Functions named `main` with no
`tf.entry_function` attribute will not be pruned, as such graphs/functions may
have been imported from a V1 TensorFlow graph, where feeds/fetches/targets are
not provided at certain stages of IR transformation (e.g. pre-placement).

For example, the following:

```mlir
func @graph(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %graph = tf_executor.graph {
    %transitive_reachable_data:2 = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
    %reachable_data:2 = tf_executor.island wraps "tf.Identity"(%transitive_reachable_data#0) : (tensor<i32>) -> tensor<i32>
    %unreachable_data:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %transitive_reachable_control = tf_executor.island wraps "tf.NoOp"() : () -> ()
    %reachable_control = tf_executor.island(%transitive_reachable_control) wraps "tf.NoOp"() : () -> ()
    %unreachable_control = tf_executor.island wraps "tf.NoOp"() : () -> tensor<i32>
    tf_executor.fetch %reachable_data#0, %reachable_control : tensor<i32>, !tf_executor.control
  }
  return %graph : tensor<i32>
}
```

will be transformed into:

```mlir
func @graph(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %graph = tf_executor.graph {
    %transitive_reachable_data:2 = tf_executor.island wraps "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
    %reachable_data:2 = tf_executor.island wraps "tf.Identity"(%transitive_reachable_data#0) : (tensor<i32>) -> tensor<i32>
    %transitive_reachable_control = tf_executor.island wraps "tf.NoOp"() : () -> ()
    %reachable_control = tf_executor.island(%transitive_reachable_control) wraps "tf.NoOp"() : () -> ()
    tf_executor.fetch %reachable_data#0, %reachable_control : tensor<i32>, !tf_executor.control
  }
  return %graph : tensor<i32>
}
```
  }];

  let constructor = "tf_executor::CreateTFExecutorGraphPruningPass()";
}

def ExecutorDialectToFunctionalPass : FunctionPass<"tf-executor-to-functional-conversion"> {
  let summary = "Lifts tf_executor.island inner ops from a tf_executor.graph";

  let description = [{
This pass converts tf_executor.graphs consisting of only tf_executor.islands and
a tf_executor.fetch into a sea of nodes consisting of TensorFlow Dialect ops by
lifting such ops out of a tf_executor.graph's tf_executor.islands. If V1 control
flow ops are present in a tf_executor.graph, an error will be returned.

For example, the following:

```mlir
func @my_fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %graph_results:2 = tf_executor.graph {
    %island_0_result, %island_0_control = tf_executor.island {
      %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %identity : tensor<i32>
    }
    %island_1_result, %island_1_control = tf_executor.island {
      %identity_n:2 = "tf.IdentityN"(%arg1, %island_0_result) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
      tf_executor.yield %identity_n#0
    }
    tf_executor.fetch %island_0_result, %island_1_result : tensor<i32>, tensor<i32>
  }
  return %graph_results#0, %graph_results#1 : tensor<i32>, tensor<i32>
}
```

will be transformed into:

```mlir
func @my_fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  %identity_n:2 = "tf.IdentityN"(%arg1, %identity) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %identity, %identity_n#0 : tensor<i32>, tensor<i32>
}
```
  }];

  let constructor = "CreateExecutorDialectToFunctionalConversionPass()";
}

def TPUClusterFormationPass : Pass<"tf-tpu-cluster-formation", "ModuleOp"> {
  let summary = "Forms clusters from operations assigned to the same TPU computation";

  let description = [{
TPU computations from the frontend are composed of a `tf.TPUReplicateMetadata`
op, a subgraph of ops (TensorFlow Dialect) each with a matching `_tpu_replicate`
attribute relative to the associated `tf.TPUReplicateMetadata` op, and
optionally `tf.TPUReplicatedInput` and `tf.TPUReplicatedOutput` ops feeding in
inputs and outputs to and from a replicated TPU computation. The number of times
a TPU computation is replicated is defined in the `tf.TPUReplicateMetadata` op
(`num_replicas` attribute) and operand and result sizes of
`tf.TPUReplicatedInput` and `tf.TPUReplicatedOutput` respectively must match,
excluding packed tensors. It is also assumed ops of the same TPU computation do
not have ops outside of the TPU computation that are both inputs and outputs to
the same TPU computation.

This pass takes the TPU computation subgraph, moves them into a
`tf_device.cluster`, and copies over attributes from the associated
`tf.TPUReplicateMetadata` op to the newly created `tf_device.cluster`. If the
computation is replicated (`num_replicas` > 1), the `num_replicas` attribute is
not copied over but instead the `tf_device.cluster` is further wrapped with a
`tf_device.replicate`, and associated `tf.TPUReplicatedInput` and
`tf.TPUReplicatedOutput` ops are replaced as the `tf_device.replicate` operands
and results. Otherwise, the single operands and results of the associated
`tf.TPUReplicatedInput` and `tf.TPUReplicatedOutput` ops are simply forwarded to
the `tf_device.cluster`.

For example, the following non replicated computation:

```mlir
func @tpu_computation(%arg0: tensor<i32>) -> tensor<i32> {
  // Metadata op for cluster `cluster` with 1 replica, 1 core per replica and
  // with topology `<topology>`.
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", num_relicas = 1, num_cores_per_replica = 1, topology = "<topology>", device_assignment = [], padding_map = []} : () -> ()
  %replicated_input = "tf.TPUReplicatedInput"(%arg0) : (tensor<i32>) -> tensor<i32>
  %identity = "tf.Identity"(%replicated_input) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
  %replicated_output = "tf.TPUReplicatedOutput(%identity) : (tensor<i32>) -> tensor<i32>
  return %replicated_output : tensor<i32>
}
```

will be transformed into:

```mlir
func @tpu_computation(%arg0: tensor<i32>) -> tensor<i32> {
  %cluster = "tf_device.cluster"() ( {
    %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
    tf_device.return %identity : tensor<i32>
  }) {_tpu_replicate = "cluster", num_cores_per_replica = 1, topology = "topology", device_assignment = [], padding_map = []} : () -> (tensor<i32>)
  return %cluster : tensor<i32>
}
```

The following replicated computation:

```mlir
func @tpu_computation(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", num_relicas = 2, num_cores_per_replica = 1, topology = "topology", device_assignment = [], padding_map = []} : () -> ()
  %replicated_input = "tf.TPUReplicatedInput"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %identity = "tf.Identity"(%replicated_input) {_tpu_replicate = "cluster"} : (tensor<i32>) -> tensor<i32>
  %replicated_output:2 = "tf.TPUReplicatedOutput(%identity) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %replicated_output#0, %replicated_output#1 : tensor<i32>, tensor<i32>
}
```

will be transformed into:

```mlir
func @tpu_computation(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %replicate:2 = tf_device.replicate([%arg0, %arg1] as %replicated_input) {n = 2 : i32} {
    %cluster = "tf_device.cluster"() ( {
      %identity = "tf.Identity"(%replicated_input) : (tensor<i32>) -> tensor<i32>
      tf_device.return %identity : tensor<i32>
    }) {_tpu_replicate = "cluster", num_cores_per_replica = 1, topology = "topology", device_assignment = [], padding_map = []} : () -> (tensor<i32>)
    tf_device.return %cluster : tensor<i32>
  }
  return %replicate#0, %replicate#1 : tensor<i32>, tensor<i32>
}
```
  }];

  let constructor = "TFTPU::CreateTPUClusterFormationPass()";
}

def ClusterConstantSinkingPass : FunctionPass<"tf-device-constant-sinking"> {
  let summary = "Sinks constants implicitly captured in a tf_device.cluster region.";

  let description = [{
This pass sinks implicitly captured constants (`tf.Const` ops) used by and into
a `tf_device.cluster` region. Performing this prior to outlining will reduce the
number of arguments of the outlined function.

For example, the following:

```mlir
func @cluster() -> tensor<i32> {
  %const = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cluster = "tf_device.cluster"() ( {
    %identity = "tf.Identity"(%const) : (tensor<i32>) -> tensor<i32>
    tf_device.return %identity : tensor<i32>
  }) : () -> (tensor<i32>)
  return %cluster : tensor<i32>
}
```

will be transformed into:

```mlir
func @cluster() -> tensor<i32> {
  %cluster = "tf_device.cluster"() ( {
    %const = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %identity = "tf.Identity"(%const) : (tensor<i32>) -> tensor<i32>
    tf_device.return %identity : tensor<i32>
  }) : () -> (tensor<i32>)
  return %cluster : tensor<i32>
}
```
  }];

  let constructor = "TFDevice::CreateClusterConstantSinkingPass()";
}

def ClusterOutliningPass : Pass<"tf-device-cluster-outlining", "ModuleOp"> {
  let summary = "Outlines regions of tf_device.cluster operations";

  let description = [{
This pass outlines the body of a `tf_device.cluster` into a function and
replaces the `tf_device.cluster` op with an equivalent `tf_device.cluster_func`
op. Implicit operands will be captured and materialized as explicit arguments to
the newly created functions and associated `tf_device.cluster_func` ops.

For example, the following:

```mlir
func @computation(%arg0: tensor<i32>) -> tensor<i32> {
  %cluster = "tf_device.cluster"() ( {
    %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
    tf_device.return %identity : tensor<i32>
  }) : () -> (tensor<i32>)
  return %cluster : tensor<i32>
}
```

will be transformed into:

```mlir
func @computation(%arg0: tensor<i32>) -> tensor<i32> {
  %cluster = "tf_device.cluster_func"(%arg0) {func = @_func} : (tensor<i32>) -> tensor<i32>
  return %cluster : tensor<i32>
}

func @_func(%arg0: tensor<i32>) -> tensor<i32> {
  %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  return %identity : tensor<i32>
}
```
  }];

  let constructor = "TFDevice::CreateClusterOutliningPass()";
}

def TPUExtractOutsideCompilationPass : Pass<"tf-tpu-extract-outside-compilation", "ModuleOp"> {
  let summary = "Extracts TPU outside compilation computation to a separate tf_device.parallel_execute region.";

  let description = [{
This pass extracts a CPU computation cluster with `_xla_outside_compilation`
annotation, which denotes ops that should be run on CPU/host, from a TPU cluster.
Each outside compilation cluster is moved to
a tf_device.parallel_execute region. The TPU cluster is also moved to a
tf_device.parallel_execute region. Communication ops between device and host are
added to pass inputs/outputs to/from the outside compiled region.

For example, the following tf_device.cluster with an op marked for `xla_outside_compilation`:

```mlir
func @outside_compilation() -> tensor<f32> {
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.Const"() {_xla_outside_compilation = "0", value = dense<1.0> : tensor<f32>} : () -> (tensor<f32>)
    %2 = "tf.Identity"(%1) {_xla_outside_compilation = "0"} : (tensor<f32>) -> (tensor<f32>)
    %3 = "tf.AddV2"(%1, %2) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
    tf_device.return %3 : tensor<f32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<f32>
  return %0 : tensor<f32>
}
```

will become a tf_device.parallel_execute op with a CPU/host region and
a tf_device.cluster with communication ops to send data to/from device/host:

```mlir
func @outside_compilation() -> tensor<f32> {
  %0 = "tf_device.parallel_execute"() ( {
    "tf_device.launch"() ( {
      %1 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf.string>
      %2 = "tf._XlaRecvAtHost"(%1) {device_ordinal = 0 : i64, key = "host_compute_channel_0_0_args"} : (tensor<3x!tf.string>) -> tensor<f32>
      %3 = "tf.Identity"(%2) : (tensor<f32>) -> tensor<f32>
      "tf._XlaSendFromHost"(%3, %1) {device_ordinal = 0 : i64, key = "host_compute_channel_0_0_retvals"} : (tensor<f32>, tensor<3x!tf.string>) -> ()
      tf_device.return
    }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
    tf_device.return
  },  {
    %1 = "tf_device.cluster"() ( {
      %2 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
      %3 = "tf._XlaHostComputeMlir"(%2) {recv_key = "host_compute_channel_0_0_retvals", send_key = "host_compute_channel_0_0_args", tpu_core = 0 : i64} : (tensor<f32>) -> tensor<f32>
      %4 = "tf.AddV2"(%2, %3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      tf_device.return %4 : tensor<f32>
    }) {device_assignment = [], num_cores_per_replica = 1 : i64, topology = ""} : () -> tensor<f32>
    tf_device.return %1 : tensor<f32>
  }) : () -> tensor<f32>
  return %0 : tensor<f32>
}
```
  }];

  let constructor = "TFTPU::CreateTPUExtractOutsideCompilationPass()";
}

def MarkOpsForOutsideCompilationPass : Pass<"tf-mark-ops-for-outside-compilation", "ModuleOp"> {
  let summary = "Marks ops in device cluster for outside compilation if they are unsupported on device.";

  let description = [{
This pass marks unsupported ops in a device cluster with
`_xla_outside_compilation` attribute so the operations will run on the host
instead of the device. Unsupported ops are ops that can not be code
generated to run on the device for the cluster including:

1. String operations on TPUs.
2. Operations that don't have a kernel defined for the device.

This pass is conservative in that it will mark all ops for outside compilation
that can not be compiled for the device.  Exceptions for this are added for ops
that will be rewritten or decomposed before compiling on device.


For example, tf_device.cluster op with an unsupported op, tf.UnsupportedOp:

```mlir
func @unsupported_op() -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.UnsupportedOp"() : () -> tensor<i32>
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %2 : tensor<i32>
  }) {allow_soft_placement = true, num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  return %0 : tensor<i32>
}
```

will mark tf.UnsupportedOp with `_xla_outside_compilation` attribute:

```mlir
func @unsupported_op() -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.UnsupportedOp"() {_xla_outside_compilation = "auto0"} : () -> tensor<i32>
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    tf_device.return %2 : tensor<i32>
  }) {allow_soft_placement = true, device_assignment = [], num_cores_per_replica = 1 : i64, topology = ""} : () -> tensor<i32>
  return %0 : tensor<i32>
}
```
  }];

  let constructor = "TFDevice::CreateMarkOpsForOutsideCompilationPass()";
}
