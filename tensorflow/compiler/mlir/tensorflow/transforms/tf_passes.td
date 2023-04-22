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

def TensorflowGPUFusion : FunctionPass<"tf-gpu-op-fusion"> {
  let summary = "Fusion optimization for GPU targets";
  let description = [{
    This pass is performing fusion specific to GPU targets. This is an ad-hoc
    pass for now, but should be integrated with some notion of "target" in the
    MLIR pipeline in the future.
  }];
  let constructor = "TF::CreateGpuOpFusionPass()";
}

def MaterializePassthroughOp : FunctionPass<"tf-materialize-passthrough-op"> {
  let summary = "Materialize the MlirPassthroughOp by replacing it with the "
    "MLIR module attached as an attribute";
  let constructor = "TF::CreateMaterializePassthroughOpPass()";
}

def TensorFlowShapeInferencePass : Pass<"tf-shape-inference", "ModuleOp"> {
  let summary = "Simple Shape Inference on TensorFlow Dialect";
  // TODO(jpienaar): Write `description`.

  let constructor = "TF::CreateTFShapeInferencePass()";

  let options = [
    Option<"max_iterations_", "max-iterations", "int64_t", /*default=*/"10",
           "Maximum shape inference iterations">
  ];
}

def TensorFlowEnsureStaticShapesPass : FunctionPass<"tf-ensure-static-shapes"> {
  let summary = "Performs checks that the whole module does not contain dynamic shapes.";

  let description = [{
    This pass performs check that none of the ops in the MLIR module
    have dynamic shapes.
    Note, the pass is created temporary to stage the rollout of the second
    phase of the MLIR bridge and will be deleted after the rollout stage is completed.
  }];

  let constructor = "TF::CreateTFEnsureStaticShapesPass()";
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

def ExecutorIslandCoarseningPass : FunctionPass<"tf-executor-island-coarsening"> {
  let summary = "Walks tf_executor::GraphOp and merges individual tf_executor::IslandOps.";
  let description = [{
    This pass performs whole graph analysis for a graph encapsulated into tf_executor::GraphOp.
    The analysis identifies all IslandOps within the graph which could be merged together.
    The goal is to merge as many islands as possible.
    Once analysis is completed, the pass merges all IslandOps in a single scan.
  }];
  let constructor = "tf_executor::CreateTFExecutorIslandCoarseningPass()";
}

def ExecutorTPUV1IslandInliningPass : Pass<"tf-executor-tpu-v1-island-inlining", "ModuleOp"> {
  let summary = "Inline calls to the nested TPU module.";

  let description = [{
    This pass inlines the islands calling into the nested module that was
    outlined, thus reversing the effect of the
    `-tf-executor-tpu-v1-island-outlining` pass.

    For example, the following:
    ```mlir
    module {
      func @foo(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = tf_executor.graph {
          %outputs, %control = tf_executor.island wraps "tf.PartitionedCall"(%arg0) {f = @_tpu_v1_compat_outlined::@bar} : (tensor<f32>) -> tensor<f32>
          tf_executor.fetch %outputs : tensor<f32>
        }
        return %0 : tensor<f32>
      }
      module @_tpu_v1_compat_outlined {
        func nested @bar(%arg0: tensor<f32>) -> tensor<f32> {
          %0 = "tf.opA"(%arg0) : (tensor<f32>) -> tensor<f32>
          return %0 : tensor<f32>
        }
      }
    }
    ```

    will be transformed into:

    ```mlir
    module  {
      func @foo(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = tf_executor.graph {
          %outputs, %control = tf_executor.island {
            %1 = "tf.opA"(%arg0) : (tensor<f32>) -> tensor<f32>
            tf_executor.yield %1 : tensor<f32>
          }
          tf_executor.fetch %outputs : tensor<f32>
        }
        return %0 : tensor<f32>
      }
    }
    ```
  }];

  let constructor = "tf_executor::CreateTFExecutorTPUV1IslandInliningPass()";
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

def LaunchOutliningPass : Pass<"tf-device-launch-outlining", "ModuleOp"> {
  let summary = "Outlines regions of tf_device.launch operations";

  let description = [{
    This pass outlines the body of a `tf_device.launch` into a function and
    replaces the `tf_device.launch` op with an equivalent `tf_device.launch_func`
    op. Implicit operands will be captured and materialized as explicit arguments to
    the newly created functions and associated `tf_device.launch_func` ops. The
    `device` attribute from the `launch` op is transferred to `launch_func`.

    For example, the following:

    ```mlir
    func @computation(%arg0: tensor<i32>) -> tensor<i32> {
      %launch = "tf_device.launch"() ( {
        %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
        tf_device.return %identity : tensor<i32>
      }) {device = "some_device"} : () -> (tensor<i32>)
      return %launch : tensor<i32>
    }
    ```

    will be transformed into:

    ```mlir
    func @computation(%arg0: tensor<i32>) -> tensor<i32> {
      %launch = "tf_device.launch_func"(%arg0) {device = "some_device", func = @_func} : (tensor<i32>) -> tensor<i32>
      return %launch : tensor<i32>
    }

    func @_func(%arg0: tensor<i32>) -> tensor<i32> {
      %identity = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
      return %identity : tensor<i32>
    }
    ```
  }];

  let constructor = "TFDevice::CreateLaunchOutliningPass()";
}

def ConvertLaunchFuncToTFCallPass : Pass<"tf-device-convert-launch-func-to-tf-call", "ModuleOp"> {
  let summary = "Rewrites tf_device::LaunchFuncOp to TF::PartitionedCallOp";

  let description = [{
    This pass converts tf_device::LaunchFuncOp into an equivalent
    TF::PartitionedCallOp so that it can be exported to TensorFlow GraphDef.
  }];

  let constructor = "TFDevice::CreateConvertLaunchFuncToTFCallPass()";
}

def MarkInputOutputAliasesPass : Pass<"tf-device-mark-input-output-aliases", "ModuleOp"> {
  let summary = "Marks device cluster inputs-output pairs that read/write to the same variable as aliases";

  let description = [{
    This pass analyzes the inputs and outputs to device cluster and marks those
    input-output pairs as aliases (using `tf.aliasing_output` attribute) which read
    and write to the same resource. This aliasing information can then be propagated
    to XLA compiler for input/output buffer space optimizations.
  }];

  let constructor = "TFDevice::CreateMarkInputOutputAliasesPass()";
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

def HoistReplicateInvariantResourceWritesPass : FunctionPass<"tf-hoist-replicate-invariant-resource-writes"> {
  let summary = "Hoists writes to replicate invariant resource variables.";

  let description = [{
    This pass hoists replicate invariant resource variable writes outside
    tf_device.replicate op. These may have been inserted by other passes such as
    resource op lifting. However, if the resource variable is not replicated, writes
    to such variables for each replica are redundant and can be replaced by writing
    a single value from first replica.

    The benefit of this optimization is reduced memory requirement on host. For
    multiple writes (one from each replica) to such variables, the host would
    allocate buffer space to recieve the device output from all replicas, which is
    not required. We can use the output of first replica in such cases.
  }];

  let constructor = "TF::CreateHoistReplicateInvariantResourceWritesPass()";
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
    ```

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

def VerifySuitableForExportPass : Pass<"tf-verify-for-export", "ModuleOp"> {
  let summary = "Verify module is suitable for export back to TF Graph";
  let description = [{
    Verifies whether all functions in module are of single tf_executor.graph and
    each tf_executor.island in tf_executor.graph only has a single op.
  }];

  let constructor = "TF::CreateVerifySuitableForExportPass()";
}

def PrepareTpuComputationForTfExportPass : Pass<"prepare-tpu-computation-for-tf-export", "ModuleOp"> {
  let summary = "Prepare TPU computation to be legal for export to TensorFlow";
  let description = [{
    Prepares TPU computation module attached to _TPUCompileMlir op for
    TensorFlow graph export by making transformation such as replacing or
    removing MLIR or XLA specific attributes that are not legal in TensorFlow
    graph.
  }];

  let constructor = "TF::CreatePrepareTpuComputationForTfExportPass()";
}

def TensorDeviceCopyConversionPass : FunctionPass<"tf-tensor-device-copy"> {
  let summary = "Fold the tf.Identity op and the tf.IdentityN op if the op has the same device as its operand";

  let constructor = "TF::CreateTensorDeviceCopyConversionPass()";
}

def TensorArrayOpsDecompositionPass : Pass<"tf-tensor-array-ops-decomposition", "ModuleOp"> {
  let summary = "Decompose tensor array operations into local variable operations.";

  let description = [{
    A pass that converts tensor array operations to tensor operations and
    read/assign ops on local variables. A later resource lifting pass can further
    remove the local variables.

    This pass requires that the full shape of the tensor array can be inferred:
    1) the size needs to be a constant, 2) it specifies the full element shape,
    or that can be inferred from a later write, and 3) all elements have the same
    shape.
  }];
  let dependentDialects = ["tensor::TensorDialect"];

  let constructor = "TF::CreateTensorArrayOpsDecompositionPass()";
}

def TensorFlowOptimizePass : FunctionPass<"tf-optimize"> {
  let summary = "Optimize TensorFlow module";

  let constructor = "TF::CreateTFOptimizePass()";
}

def PromoteResourcesToArgsPass : Pass<"tf-promote-resources-to-args", "ModuleOp"> {
  let summary = "Promote resources reads/writes to function inputs/outputs.";
  let description = [{
    This pass promotes resource accesses in the main function to input arguments
    and outputs of the main function.

    Two types of resources are supported:
    (1) A function argument of TF::ResourceType type (this pass).
    (2) A VarHandleOp in the function (tf-promote-var-handles-to-args).

    After the pass,

     . The function will have an input argument for each resource that is
       already provided as an input argument or is read. The type of the input
       argument will become the shape of the value represented by the resource.

     . The function will have an output for each resource that is written. The
       type of the output will become the shape of the resource.

    The information of variable identification and input-output alising is
    recorded as named attributes of the input argument or output:

     . 'tf.resource_name' matches 'shared_name' of VarHandleOp, which represents
       the identifier of the corresponding resource. This attribute is added to
       an input argument if the initial value of the resource is read, or to the
       output if the initial value is not read.

     . 'tf.aliasing_output' is the index of the function output that is an alias
       of the input argument. This attribute is added only to the input argument
       when the initial value of the corresponding resource is read, and the
       resource is written later.

    Assumption of this pass:
     . Compound resource operations have already been decomposed.
     . Dead functions have already been removed, as resource arguments in dead
       functions can cause the pass to fail.
  }];

  let constructor = "TF::CreatePromoteResourcesToArgsPass()";
}

def PromoteVarHandlesToArgsPass : Pass<"tf-promote-var-handles-to-args", "ModuleOp"> {
  let summary = "Promote tf.VarHandleOps to function arguments.";
  let description = [{See joint description in promote resources to args.}];

  let constructor = "TF::CreatePromoteVarHandlesToArgsPass()";
}

def TPUMergeVariablesWithExecutePass : Pass<"tf-tpu-merge-variables-with-execute", "ModuleOp"> {
  let summary = "Merges device variable reads and updates into TPU execute ops";

  let description = [{
    This pass finds on-device resource variable reads and updates surrounding a
    `tf.TPUExecute` op and merges them into a `tf.TPUExecuteAndUpdateVariables`
    op. This allows the TPU execution to perform more efficient in-place
    variable updates.

    For example,

    ```mlir
      %0 = "tf.ReadVariableOp"(%arg0)
      %1 = "tf.ReadVariableOp"(%arg1)
      %2 = "tf.TPUExecute"(%0, %1, %compile)
      %3 = "tf.AssignVariableOp"(%arg0, %2)
    ```

    will be transformed into

    ```mlir
      %2 = "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1, %compile)
        { device_var_reads_indices = [0, 1],
          device_var_updates_indices = [0, -1] }
    ````

    The transformation happens only for on-device variables. The above
    transformation requires `%arg0`, `%arg1` to have the same device assignment
    as the `TPUExecute` op.
  }];

  let constructor = "TFTPU::CreateTPUMergeVariablesWithExecutePass()";
}

def ReplicateInvariantOpHoistingPass : FunctionPass<"tf-replicate-invariant-op-hoisting"> {
  let summary = "Hoists replicate invariant operations out of replicate";

  let description = [{
    This pass looks for replicate invariant ops in a `tf_device.replicate` op
    region and hoists them out. It also makes `tf.Shape` ops replicate invariant
    if possible. This currently updates or replaces `tf.Shape` ops of replicated
    arguments, either tensors or resources.

    For example, the following

    ```mlir
    tf_device.replicate([%0, %1] as %ri: tensor<*xi32>) {n = 2 : i32} {
      %2 = "tf.Shape"(%ri) : (tensor<*xi32>) -> tensor<?xi32>
      tf_device.return
    }
    ```

    gets converted to

    ```mlir
    tf_device.replicate([%0, %1] as %ri: tensor<*xi32>) {n = 2 : i32} {
      %2 = "tf.Shape"(%0) : (tensor<*xi32>) -> tensor<?xi32>
      tf_device.return
    }
    ```

    and for resource variables the following

    ```mlir
    tf_device.replicate([%0, %1] as %ri: tensor<*x!tf.resource>) {n = 2 : i32} {
      %2 = "tf.ReadVariableOp"(%ri) : tensor<*x!tf.resource> -> tensor<*xi32>
      %3 = "tf.Shape"(%2) : (tensor<*xi32>) -> tensor<?xi32>
      tf_device.return
    }
    ```

    gets converted to

    ```mlir
    tf_device.replicate([%0, %1] as %ri: tensor<*x!tf.resource>) {n = 2 : i32} {
      %2 = "tf.ReadVariableOp"(%ri) : tensor<*x!tf.resource> -> tensor<*xi32>
      %3 = "tf.VariableShape"(%0) : (tensor<*x!tf.resource>) -> tensor<?xi32>
      tf_device.return
    }
    ```
  }];

  let constructor = "TFDevice::CreateReplicateInvariantOpHoistingPass()";
}

def LowerQuantizedPass : FunctionPass<"tf-lower-quantized"> {
  let summary = "Lowers ops that require quantized input or output.";

  let description = [{
    This pass rewrites all ops that have at least one input or output that must
    be a quantized type to ops whose inputs and outputs allow non-quantized
    types. Examples of quantized types are TF_Qint8 or TF_Quint8.

    An example is TF_DequantizeOp, which converts a quantized type to a float.
    This op is rewritten to generic ops that perform the scale and shift
    and can operate on non-quantized types.

    Currently, TF_DequantizeOp is the only op with a lowering that falls
    in this category. When more lowerings are added (e.g. QuantizeV2Op),
    they should be added to this pass.
  }];

  let constructor = "TF::CreateLowerQuantizedPass()";
}

def OutsideCompiledToHostLaunchPass : Pass<"tf-outside-compiled-to-host-launch", "ModuleOp"> {
  let summary = "Wraps each op with the _xla_outside_compiled attribute in a separate tf_device.launch on replicated host device.";

  let description = [{
    This pass wraps ops with the same `_xla_outside_compilation`
    attribute value in a tf_device.launch op with host device assignment.

    A simple example:

    ```mlir
      "tf_device.cluster"() ( {
        "tf.A"()
        "tf.B"() {_xla_outside_compilation = "cluster1"}
        "tf.C"()
        tf_device.return
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
    ```

    Would become the following ops (unimportant attribute, type are omitted):

    ```mlir
      "tf_device.cluster"() ( {
        "tf.A"()
        "tf_device.launch"() {
          "tf.B"() {_xla_outside_compilation = "cluster1"}
          tf_device.return
        } {device = "TPU_REPLICATED_HOST"} : () -> ()
        "tf.C"()
        tf_device.return
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
    ```
  }];

  let constructor = "TFTPU::CreateOutsideCompiledToHostLaunchPass()";
}

def TPUHostComputationExpansionPass : FunctionPass<"tf-tpu-host-computation-expansion"> {
  let summary = "Expands host computation before and after TPU computation.";

  let description = [{
    This pass expands outside compilation attributes to Identity/Cast ops
    at the head of TPU computation if it's only used by outside compiled ops.
  }];

  let constructor = "TFTPU::CreateTPUHostComputationExpansionPass()";
}

def TPUUpdateEmbeddingEnqueueOpInputsPass : FunctionPass<"tf-tpu-update-embedding-enqueue-op-inputs"> {
  let summary = "Updates inputs to TPU embedding enqueue ops depending on whether graph is in training mode or in evaluation mode.";

  let description = [{
    Updates inputs to TPU embedding enqueue ops depending on whether graph
    is in training mode or in evaluation mode.
  }];

  let constructor = "TFTPU::CreateTPUUpdateEmbeddingEnqueueOpInputsPass()";
}

def DropWhileShapeInvariantPass : FunctionPass<"tf-drop-while-shape-invariant"> {
  let summary = "Drop `shape_invariant` attrbute from While/WhileRegion ops.";

  let description = [{
    Drop `shape_invariant` attribute from tf.While and tf.WhileRegion op. This
    would allow shape inference pass to further refine operand/result shapes of
    these ops. This is only safe to do when compiling to XLA.
  }];

  let constructor = "TF::CreateDropWhileShapeInvariantPass()";
}

def DropWhileShapeInvariantInDeviceClusterPass : FunctionPass<"tf-drop-while-shape-invariant-in-device-cluster"> {
  let summary = "Drop `shape_invariant` attrbute from While/WhileRegion ops inside device cluster.";

  let description = [{
    Drop `shape_invariant` attribute from tf.While and tf.WhileRegion op only
    inside device cluster. This would allow shape inference pass to further
    refine operand/result shapes of these ops. This is only safe to do when
    compiling to XLA.
  }];

  let constructor = "TF::CreateDropWhileShapeInvariantInDeviceClusterPass()";
}

def TPUCleanupClusterAttributesPass : Pass<"tf-tpu-cleanup-cluster-attributes", "ModuleOp"> {
  let summary = "Eliminate _tpu_replicate and other attributes from ops in a cluster";

  let description = [{
    This pass eliminate `_tpu_replicate` and `device` attribute on operations
    that are contained in a tf_device.cluster op.
  }];

  let constructor = "TFTPU::CreateTPUClusterCleanupAttributesPass()";
}

def TPUExtractHeadTailOutsideCompilationPass : Pass<"tf-tpu-extract-head-tail-outside-compilation", "ModuleOp"> {
  let summary = "Extracts TPU head or tail outside compilation to separate host launches before/after device cluster.";

  let description = [{
    This pass extracts a CPU computation cluster with `_xla_outside_compilation`
    annotation from the head or tail of a TPU cluster.

    For example:

    ```mlir
      %cluster = "tf_device.cluster"() ( {
        %a = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
        %b = "tf.B"(%a) : (tensor<i32>) -> tensor<i32>
        %c = "tf.C"(%b) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
        tf_device.return %c : tensor<i32>
      }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> tensor<i32>
      return %cluster : tensor<i32>
    ```

    becomes:

    ```mlir
    %0 = "tf_device.launch"() ( {
      %3 = "tf.A"(%arg0) : (tensor<i32>) -> tensor<i32>
      tf_device.return %3 : tensor<i32>
    }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> tensor<i32>
    %1 = "tf_device.cluster"() ( {
      %3 = "tf.B"(%0) : (tensor<i32>) -> tensor<i32>
      tf_device.return %3 : tensor<i32>
    }) {device_assignment = [], num_cores_per_replica = 1 : i64, padding_map = [], step_marker_location = "", topology = ""} : () -> tensor<i32>
    %2 = "tf_device.launch"() ( {
      %3 = "tf.C"(%1) : (tensor<i32>) -> tensor<i32>
      tf_device.return %3 : tensor<i32>
    }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> tensor<i32>
    return %2 : tensor<i32>

    ```
  }];

  let constructor = "TFTPU::CreateTPUExtractHeadTailOutsideCompilationPass()";
}

def TPUSpaceToDepthPass : Pass<"tf-tpu-space-to-depth-pass", "ModuleOp"> {
  let summary = "Applies automatic space to depth transform for the first or frontier convolutions consume host inputs on TPU.";

  let description = [{
    Automatic space to depth transform is done by adding space to depth transform op after host input
    and applying space to depth transform for the first convolution and its backprop filter on TPU.

    For example, original program:

    ```mlir
    module {
      func @while_body {
        %input = "tf.IteratorGetNext"(...) {device = "/CPU:0"}: -> tensor<2x224x224x3xf32>
        %device_launch = "tf_device.cluster_func"(%input,...) {func = @_func,...)
        return ...
      }
      func @_func(%input: tensor<2x224x224x3xf32>, %filter: tensor<7x7x3x64xf32>) {
        %6 = "tf.Conv2D"(%input, %filter)  {strides = [1, 2, 2, 1]}: (tensor<2x230x230x3xf32>, tensor<7x7x3x64xf32>) -> tensor<2x112x112x64xf32>
      }
    }
    ```

    The program will be transformed into:

    ```mlir
    module {
      func @while_body {
        %input = "tf.IteratorGetNext"(...) {device = "/CPU:0"} -> tensor<2x224x224x3xf32>
        %space_to_depth = "tf.SpaceToDepth"(%input) {block_size = 2, ...}: (tensor<2x224x224x3xf32>) -> tensor<2x112x112x12xf32>
        %device_launch = "tf_device.cluster_func"(%space_to_depth,...) {func = @_func,...)
        return ...
      }
      func @_func(%input: tensor<2x112x112x12xf32>, %filter: tensor<7x7x3x64xf32>) {
        %filter_transform = "tf.Pad/tf.Transpose/tf.Reshape"(%filter): tensor<7x7x3x64xf32>) -> tensor<4x4x12x64xf32>
        %conv = "tf.Conv2D"(%input, %filter_transfrom) {strides = [1, 1, 1, 1]}: (tensor<2x112x112x12xf32>, tensor<4x4x12x64xf32>) -> tensor<2x112x112x64xf32>
      }
    }
    ```

    This way, the first convolution with 3 feature dimension will be transformed
    to 12 feature dimension, which has better performance on TPU.
  }];

  let constructor = "TFTPU::CreateTPUSpaceToDepthPass()";
}

def MergeControlFlowPass : Pass<"tf-merge-control-flow", "ModuleOp"> {
  let summary = "Merges IfRegion ops together with a common predicate.";

  let description = [{
    This pass merges IfRegion ops together if they have the same predicate and it
    is safe to do so (there are no intermediate dependencies, they are in the
    same block, etc).

    For example:

    ```mlir
    "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) { is_stateless = true } : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ( {
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
      }) { is_stateless = true } : (tensor<i1>) -> ()
    ```

    Would be transformed to:

    ```mlir
    "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      %3 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
      }) { is_stateless = true } : (tensor<i1>) -> ()
    ```
  }];

  let constructor = "TFDevice::CreateMergeControlFlowPass()";
}

def TensorListOpsDecompositionPass : Pass<"tf-tensor-list-ops-decomposition", "ModuleOp"> {
  let summary = "Decomposes TensorList operations into generic operations on tensors.";

  let description = [{
This pass rewrites TensorList operations into generic and non-mutating
operations on tensors. This results in operations that can be legalized to XLA.

The list is converted to a single large tensor that includes all list elements,
with a new first dimension for the list index. List update operations are
converted to operations that create a new tensor representing the list.

In the current implementation, the resulting operations are statically shaped,
which means it must be possible to infer a bound on the full shape of the
TensorList. That is, the `element_shape` and `num_elements` arguments to a
tensor list creation op are constant.

A tensor list creation op `tf.EmptyTensorList`/`tf.TensorListReserve` will be
turned in to a zero-initialized buffer, and the size is initialized to 0
for `tf.EmptyTensorList` or the specified size for `tf.TensorListReserve`.
Each push will be turned into `tf.XlaDynamicUpdateSlice` with the incremented
size, and each pop will be turned into a `tf.Slice` and a copy of the buffer
with decremented size. Each `tf.TensorListSetItem` will be turned into a
`tf.XlaDynamicUpdateSlice` with unchanged size, and each `tf.TensorListGetItem`
will be rewritten to a `tf.Slice`.

The pass also works across control flow and functional calls.

For example, the TensorList ops in the following function:

```mlir
func @main(%arg0: tensor<8x4xf32>) {
  %elem_shape = "tf.Const"() {value = dense<[8, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %max_size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %tl = "tf.EmptyTensorList"(%elem_shape, %max_size) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<8x4xf32>>>
  %push = "tf.TensorListPushBack"(%tl, %arg0) : (tensor<!tf.variant<tensor<8x4xf32>>>, tensor<8x4xf32>) -> tensor<!tf.variant<tensor<8x4xf32>>>
  return
}
```

will be transformed to:

```mlir
func @main(%arg0: tensor<8x4xf32>) {
  // EmptyTensorList lowering
  %emptyi = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %emptyf = "tf.Cast"(%emptyi) : (tensor<i32>) -> tensor<f32>
  %size_shape = "tf.Const"() {value = dense<[10, 8, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
  %tl = "tf.BroadcastTo"(%emptyf, %size_shape) : (tensor<f32>, tensor<3xi32>) -> tensor<10x8x4xf32>
  // TensorListPushBack lowering
  %index_in_list = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %arg0_shape = "tf.Const"() {value = dense<[1, 8, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
  %arg0_reshaped = "tf.Reshape"(%arg0, %arg0_shape) : (tensor<8x4xf32>, tensor<3xi32>) -> tensor<1x8x4xf32>
  %zeroi2 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %axis = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %start_indices = "tf.ConcatV2"(%index_in_list, %zeroi2, %axis) : (tensor<1xi32>, tensor<2xi32>, tensor<i32>) -> tensor<3xi32>
  %push = "tf.XlaDynamicUpdateSlice"(%tl, %arg0_reshaped, %start_indices) : (tensor<10x8x4xf32>, tensor<1x8x4xf32>, tensor<3xi32>) -> tensor<10x8x4xf32>
  %one = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %next_index_in_list = "tf.AddV2"(%index_in_list, %one) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return
}
```
  }];

  let constructor = "TF::CreateTensorListOpsDecompositionPass()";
}
