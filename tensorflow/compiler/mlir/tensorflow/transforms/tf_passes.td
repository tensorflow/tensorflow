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

Option `ops-to-preserve` allows to specify ops that should not be pruned,
regardless of their reachability.

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

  let options = [
   ListOption<"ops_to_preserve_", "ops-to-preserve", "std::string",
               "Comma separated list of ops that should not be pruned "
               "regardless of reachability",
               "llvm::cl::MiscFlags::CommaSeparated">
  ];
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

def TPUResourceReadForWritePass : Pass<"tf-tpu-resource-read-for-write", "ModuleOp"> {
  let summary = "Inserts tf.ReadVariableOp inputs to a TPU cluster for resource writes with no reads";

  let description = [{
This pass materializes `tf.ReadVariableOp` inputs to an outlined TPU computation
for resource variables where only writes are present so later in the pipeline
such resource variables can be fused with generated `tf.TPUExecute` ops, which
only supports resource variable read or read + write. For all TPU computations,
resource variables are required to be initialized prior to execution. Write only
resource variable uses can be generated currently via packed tensor uses.

For example, the following:

```mlir
func @write_only_resource(%value: tensor<i32>, %resource: tensor<*x!tf.resource<tensor<i32>>>) {
  %0 = "tf_device.cluster_func"(%value) {func = @cluster} : (tensor<i32>) -> tensor<i32>
  "tf.AssignVariableOp"(%resource, %0) : (tensor<*x!tf.resource<tensor<i32>>>, tensor<i32>) -> ()
  return
}

func @cluster(%arg0: tensor<i32>) -> tensor<i32> {
  %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  return %identity : tensor<i32>
}
```

will be transformed into:

```mlir
func @write_only_resource(%value: tensor<i32>, %resource: tensor<*x!tf.resource<tensor<i32>>>) {
  %resource_read = "tf.ReadVariableOp"(%resource) : (tensor<*x!tf.resource<tensor<i32>>>) -> tensor<i32>
  %0 = "tf_device.cluster_func"(%value, %resource_read) {func = @cluster} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  "tf.AssignVariableOp"(%resource, %0) : (tensor<*x!tf.resource<tensor<i32>>>, tensor<i32>) -> ()
  return
}

func @cluster(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  return %identity : tensor<i32>
}
```
  }];

  let constructor = "TFTPU::CreateTPUResourceReadForWritePass()";
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

def FunctionalControlFlowToRegionsPass : Pass<"tf-functional-control-flow-to-regions", "ModuleOp"> {
  let summary = "Transforms functional control flow operations to their region-based counterparts";

  let description = [{
This pass transforms functional control flow operations in the TensorFlow
dialect to their region-based counterparts, i.e., `tf.If` is transformed to
`tf.IfRegion` and `tf.While` is transformed to `tf.WhileRegion`.

For example, this functional operation

```mlir
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @then_branch_func, else_branch = @else_branch_func, is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> tensor<*xf32>
```

will be transformed into this region-based operation

```mlir
    %0 = "tf.IfRegion"(%arg0) ( {
      %1 = call @then_branch_func(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
      "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    },  {
      %1 = call @else_branch_func(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
      "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }) {is_stateless = false} : (tensor<i1>) -> tensor<*xf32>
```
  }];

  let constructor = "TF::CreateTFFunctionalControlFlowToRegions()";
}

def RegionControlFlowToFunctionalPass : Pass<"tf-region-control-flow-to-functional", "ModuleOp"> {
  let summary = "Transforms region-based control flow operations to their functional counterparts";

  let description = [{
This pass transforms region-based control flow operations in the TensorFlow
dialect to their functional counterparts, i.e., `tf.IfRegion` is transformed to
`tf.If` and `tf.WhileRegion` is transformed to `tf.While`.

For example, this region-based operation

```mlir
    %0 = "tf.IfRegion"(%arg0) ( {
      %1 = call @then_branch_func(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
      "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    },  {
      %1 = call @else_branch_func(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
      "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }) {is_stateless = false} : (tensor<i1>) -> tensor<*xf32>
```

will be transformed into this functional operation

```mlir
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @then_branch_func, else_branch = @else_branch_func, is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> tensor<*xf32>
```
  }];

  let constructor = "TF::CreateTFRegionControlFlowToFunctional()";
}

def TPUReorderReplicateAndPartitionedInputsPass : FunctionPass<"tf-tpu-reorder-replicate-partitioned-inputs"> {
  let summary = "Reorder replicated and partitioned input ops.";

  let description = [{
This pass rewrites how data parallelism and model parallelism is expressed for
inputs. It reorders `tf.TPUPartitionedInput` (model parallelism) and
`tf.TPUReplicatedInput` (data parallelism) ops. It transforms a DAG where
multiple `tf.TPUPartitionedInput` ops are feeding into a single
`tf.TPUReplicatedInput` into a DAG where multiple `tf.TPUReplicatedInput` ops
are feeding into a single `tf.TPUPartitionedInput`. Transforming the IR in such
a manner will allow subsequent cluster formation pass to handle IR with both
data and model parallelism in an easier manner.

For example, the following:

```mlir
!rtype = type tensor<!tf.resource<tensor<10x3xf32>>>
func @data_and_model_parallelism(%arg0: !rtype, %arg1: !rtype, %arg2: !rtype, %arg3: !rtype) -> !rtype {
  %pi_0 = "tf.TPUPartitionedInput"(%arg0, %arg1) {_XlaSharding = "", device = "", partition_dim = -1 : i64} : (!rtype, !rtype) -> !rtype
  %pi_1 = "tf.TPUPartitionedInput"(%arg2, %arg3) {_XlaSharding = "", device = "", partition_dim = -1 : i64} : (!rtype, !rtype) -> !rtype
  %ri = "tf.TPUReplicatedInput"(%pi_0, %pi_1) : (!rtype, !rtype) -> !rtype
  return %ri : !rtype
}
```

will be transformed into:

```mlir
!rtype = type tensor<!tf.resource<tensor<10x3xf32>>>
func @data_and_model_parallelism(%arg0: !rtype, %arg1: !rtype, %arg2: !rtype, %arg3: !rtype) -> !rtype {
  %ri_0 = "tf.TPUReplicatedInput"(%arg0, %arg2) : (!rtype, !rtype) -> !rtype
  %ri_1 = "tf.TPUReplicatedInput"(%arg1, %arg3) : (!rtype, !rtype) -> !rtype
  %pi = "tf.TPUPartitionedInput"(%ri_0, %ri_1) {_XlaSharding = "", device = "", partition_dim = -1 : i64} : (!rtype, !rtype) -> !rtype
  return %pi : !rtype
}
```
  }];

  let constructor = "TFTPU::CreateTPUReorderReplicateAndPartitionedInputsPass()";
}

def TPUResourceReadsWritesPartitioningPass : FunctionPass<"tf-tpu-resource-partition"> {
  let summary = "Partitions unpartitioned resource read/write to partitioned resource variables.";

  let description = [{
This pass creates individual resource reads/writes from the unpartitioned
resource variable (from `tf.TPUPartitionedInput`) to individual partitioned
resource variables (`tf.TPUPartitionedInput` operands). As resource op
decomposition/lifting occurs with the unpartitioned resource variables,
transforming the IR in such a manner will allow for subsequent passes to operate
on individual resource variable handles per core/device.

For example, the following:

```mlir
func @cluster(%arg0: tensor<!tf.resource<tensor<i32>>>, %arg1: tensor<!tf.resource<tensor<i32>>>) {
  %partitioned_variable = "tf.TPUPartitionedInput"(%arg0, %arg1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<!tf.resource<tensor<i32>>>, tensor<!tf.resource<tensor<i32>>>) -> tensor<!tf.resource<tensor<i32>>>
  %read = "tf.ReadVariableOp"(%partitioned_variable) : (tensor<!tf.resource<tensor<i32>>>) -> tensor<i32>
  %computation = "tf_device.cluster_func"(%read) {func = @computation, use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
  "tf.AssignVariableOp"(%partitioned_variable, %computation) : (tensor<!tf.resource<tensor<i32>>>, tensor<i32>) -> ()
  return
}

func @computation(%arg0: tensor<i32>) -> tensor<i32> {
  return %arg0: tensor<i32>
}
```

will be transformed into:

```mlir
func @cluster(%arg0: tensor<!tf.resource<tensor<i32>>>, %arg1: tensor<!tf.resource<tensor<i32>>>) {
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<i32>>>) -> tensor<i32>
  %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<i32>>>) -> tensor<i32>
  %partitioned_input = "tf.TPUPartitionedInput"(%read0, %read1) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %computation = "tf_device.cluster_func"(%partitioned_input) {func = @computation, use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
  %partitioned_output:2 = "tf.TPUPartitionedOutput"(%computation) {N = 2 : i64, _XlaSharding = "", partition_dim = -1 : i64} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf.resource<tensor<i32>>>, tensor<i32>) -> ()
  "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf.resource<tensor<i32>>>, tensor<i32>) -> ()
  return
}

func @computation(%arg0: tensor<i32>) -> tensor<i32> {
  return %arg0: tensor<i32>
}
```
  }];

  let constructor = "TFTPU::CreateTPUResourceReadsWritesPartitioningPass()";
}

def TPURewritePass : Pass<"tf-tpu-rewrite", "ModuleOp"> {
  let summary = "Rewrites a `tf_device.cluster_func` on TPUs into TPU runtime operations.";

  let description = [{
This pass rewrites a `tf_device.cluster_func` operation into a sequence of `tf._TPUCompileMlir`
and `tf.TPUExecute` operations. `tf._TPUCompileMlir` contains a MLIR module that is
functionally equivalent to the function referenced by `tf_device.cluster_func`.
This makes the module to be jit-compiled and executed on TPU.
If it is not possible to rewrite the operation or device assignment fails,
a failure will be returned.

Note, many parameters to the `tf_device.cluster_func` are ommited in this
and following examples.
For example, a non replicated `tf_device.cluster_func`:

```mlir
func @tf_tpu_rewrite(%arg0: tensor<i8>) {
  %0 = "tf_device.cluster_func"(%arg0) {_tpu_replicate = "cluster0", func = @func} : (tensor<i8>) -> tensor<i8>
  return
}
```

will be rewritten as:

```mlir
func @tf_tpu_rewrite(%arg0: tensor<i8>) {
  %0:2 = "tf_device.launch"() ( {
    %compilation_status, %program = "tf._TPUCompileMlir"() {mlir_module = "<serialized func>"} : () -> (tensor<!tf.string>, tensor<3x!tf.string>)
    tf_device.return %compilation_status, %program : tensor<!tf.string>, tensor<3x!tf.string>
  }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<3x!tf.string>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
  %1 = "tf_device.launch"() ( {
    %2 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<i8>, tensor<3x!tf.string>) -> tensor<i8>
    tf_device.return %2 : tensor<i8>
  }) {device = "/job:worker/replica:0/task:0/device:TPU:0"} : () -> tensor<i8>
  return
}
```

A replicated `tf_device.cluster_func`:

```mlir
func @tf_tpu_rewrite(%arg0: tensor<i8>, %arg1: tensor<i8>) {
  %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<i8>) {n = 2 : i32} {
    %1 = "tf_device.cluster_func"(%ri) {_tpu_replicate = "cluster0", func = @func} : (tensor<i8>) -> tensor<i8>
    tf_device.return %1 : tensor<i8>
  }
  return
}
```

will be rewritten as:

```mlir
func @tf_tpu_rewrite(%arg0: tensor<i8>, %arg1: tensor<i8>) {
  %0:2 = tf_device.replicate([%arg0, %arg1] as %arg2: tensor<i8>) {devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"], TPU_REPLICATED_HOST = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:CPU:0"]}, n = 2 : i32} {
    %1:2 = "tf_device.launch"() ( {
      %compilation_status, %program = "tf._TPUCompileMlir"() {mlir_module = "<serialized func>"} : () -> (tensor<!tf.string>, tensor<3x!tf.string>)
      tf_device.return %compilation_status, %program : tensor<!tf.string>, tensor<3x!tf.string>
    }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<3x!tf.string>)
    "tf_device.launch"() ( {
      "tf.TPUCompileSucceededAssert"(%1#0) : (tensor<!tf.string>) -> ()
      tf_device.return
    }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
    %2 = "tf_device.launch"() ( {
      %3 = "tf.TPUExecute"(%arg2, %1#1) : (tensor<i8>, tensor<3x!tf.string>) -> tensor<i8>
      tf_device.return %3 : tensor<i8>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<i8>
    tf_device.return %2 : tensor<i8>
  }
  return
}

A non replicated `tf_device.cluster_func` with the model parallelism:

```mlir
func @tf_tpu_rewrite(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = "tf_device.cluster_func"(%arg0) {_tpu_replicate = "cluster0", func = @func, num_cores_per_replica = 2, input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"]} : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
```

will be rewritten as:

```mlir
func @tf_tpu_rewrite(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0:3 = "tf_device.launch"() ( {
    %compilation_status, %program:2 = "tf._TPUCompileMlir"() {mlir_module = "<serialized func>"} : () -> (tensor<!tf.string>, tensor<3x!tf.string>, tensor<3x!tf.string>)
    tf_device.return %compilation_status, %program#0, %program#1 : tensor<!tf.string>, tensor<3x!tf.string>, tensor<3x!tf.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<3x!tf.string>, tensor<3x!tf.string>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  %1 = "tf_device.parallel_execute"() ( {
    %2 = "tf_device.launch"() ( {
      %3 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<8xi32>, tensor<3x!tf.string>) -> tensor<8xi32>
      tf_device.return %3 : tensor<8xi32>
    }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> tensor<8xi32>
    tf_device.return %2 : tensor<8xi32>
  },  {
    "tf_device.launch"() ( {
      "tf.TPUExecute"(%0#2) : (tensor<3x!tf.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:TPU:1"} : () -> ()
    tf_device.return
  }) : () -> tensor<8xi32>
  return %1 : tensor<8xi32>
}
```
  }];

  let constructor = "TFTPU::CreateTPURewritePass()";
}

def ClusterOpsByPolicyPass : FunctionPass<"cluster-ops-by-policy"> {
  let summary = "Clusters ops according to specified policy.";

  let description = [{
This pass clusters ops according to the policy specified by the pass options.
Clustered ops are moved to a tf_device::clusterOp region.

First you need to specify the 'oplist=<list of ops>' option. This option
specifies the names of the ops that should be clustered together. Then you need
to specify the algorithm for forming a cluster with a `mode=<algorithm>` option:

1. `use-def` (default): cluster ops together if they form a single use def-use
   chain, that is, the next op in the list uses the result of the previous op
   and is the only user of that result.
2. `union-find`: cluster ops together that are connected to each other with
   potentially different use def chains using union-find algorithm.

For both algorithms the ops should be located in the same block, be assigned to
the same device and have no side effects.

For example, running this pass with options:
  "oplist=tf.Cast,tf.Add algorithm=use-def"

```mlir
func @cluster_oplist(%arg0 : tensor<f32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Cast"(%arg0) : (tensor<f32>) -> tensor<i32>
  %1 = "SomeOp" (%arg1) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %2 : tensor<i32>
}
```

will produce tf_device::opCluster enclosing tf.Add and tf.Neg:

```mlir
func @cluster_oplist(%arg0: tensor<f32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = "SomeOp"(%arg1) : (tensor<i32>) -> tensor<i32>
  %1 = "tf_device.cluster"() ( {
    %2 = "tf.Cast"(%arg0) : (tensor<f32>) -> tensor<i32>
    %3 = "tf.Add"(%2, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) : () -> tensor<i32>
  return %1 : tensor<i32>
}
```

Running with `union-find` algorithm allows to cluster together operations that
do not form a single use-def chain:
  "oplist=tf.Add,tf.Sub algorithm=union-find"

```mlir
func @cluster_oplist(%arg0 : tensor<f32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<i32>
  %1 = "tf.Sub"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<i32>
  return %2 : tensor<i32>
}
```

will produce tf_device::opCluster enclosing tf.Add and tf.Sub:

```mlir
func @cluster_oplist(%arg0: tensor<f32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<i32>
    %2 = "tf.Sub"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<i32>
    %3 = "tf.Add"(%1, %2) : (tensor<f32>, tensor<f32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) : () -> tensor<i32>
  return %0 : tensor<i32>
}
  }];

  let constructor = "TFDevice::CreateClusterOpsByPolicyPass()";

  let options = [
   Option<"policy_name", "policy-name", "std::string", /*default=*/"",
          "Adds a policy string attribute to all extracted clusters. This "
          "attribute allows to distinguish clusters formed by different "
          "policies or maybe other clustering algorithms.">,
   Option<"min_cluster_size", "min-cluster-size", "int" , /*default=*/"1",
          "Do not form clusters smaller of the given size.">,
   Option<"algorithm", "algorithm", "std::string", /*default=*/"\"use-def\"",
          "Clustering algorithm type: `use-def` or `union-find`">,
   ListOption<"oplist", "oplist", "std::string",
               "Cluster listed ops when they form a single use def-use chain, "
               "such that each op's single user is the next op in the list.",
               "llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def VerifySuitableForExportPass : Pass<"tf-verify-for-export", "ModuleOp"> {
  let summary = "Verify module is suitable for export back to TF Graph";
  let description = [{
    Verifies whether all functions in module are of single tf_executor.graph and
    each tf_executor.island in tf_executor.graph only has a single op.
  }];

  let constructor = "TF::CreateVerifySuitableForExportPass()";
}

def PrepareTpuComputationForTfExportPass : Pass<"prepare-tpu-computation-for-tf-export", "FuncOp"> {
  let summary = "Prepare TPU computation to be legal for export to TensorFlow";
  let description = [{
    Prepares TPU computation module attached to _TPUCompileMlir op for
    TensorFlow graph export by making transformation such as replacing or
    removing MLIR or XLA specific attributes that are not legal in TensorFlow
    graph.
  }];

  let constructor = "TF::CreatePrepareTpuComputationForTfExportPass()";
}

def DeviceAttributeToLaunchPass : FunctionPass<"tf-device-attribute-to-launch"> {
  let summary = "Wraps each TF op which has a non-empty device attribute in a tf_device.launch.";

  let description = [{
This pass wraps TF ops which have a non-empty device attribute in a tf_device.lauch with
the same device attribute.

For example, the following:

```mlir
func @single_op_launch() {
  %a = "tf.opA"() {device = "CPU:0"} : () -> tensor<i1>
  return %a
}
```

will be transformed into:

```mlir
func @single_op_launch() {
  %1 = tf_device.launch() ( {
    %a = "tf.opA"() : () -> tensor<i1>
    tf_device.return %a
  }) {device = "CPU:0"} : () -> tensor<i1>
  return %1
}
```
  }];

  let constructor = "TFDevice::CreateDeviceAttributeToLaunchPass()";
}
