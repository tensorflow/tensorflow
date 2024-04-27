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
include "mlir/Pass/PassBase.td"

def VerifyClusteringPass : Pass<"verify-clustering-pass", "mlir::func::FuncOp"> {

  let summary = "Verify that the Bridge output is correct and errors if verification fails.";

  let description = [{
    Verifies whether clustering has resulted in the expected invariants. These
    include verifying that clusters have been created and have been outside
    compiled, the result is device agnostic and in TF functional dialect &
    that the device attribute exists.
  }];

  let constructor = "tensorflow::tf2xla::internal::CreateVerifyClusteringPass()";
}

def TPUClusterFormationPass : Pass<"tf-tpu-cluster-formation", "ModuleOp"> {
  let summary = "Forms clusters from operations assigned to the same TPU computation";

  let description = [{
    TPU computations from the frontend are composed of a `tf.TPUReplicateMetadata`
    op, a subgraph of ops (TensorFlow Dialect) each with a matching
    `_replication_info` attribute relative to the associated
    `tf.TPUReplicateMetadata` op, and optionally `tf.TPUReplicatedInput` and
    `tf.TPUReplicatedOutput` ops feeding in inputs and outputs to and from a
    replicated TPU computation. The number of times a TPU computation is
    replicated is defined in the `tf.TPUReplicateMetadata` op (`num_replicas`
    attribute) and operand and result sizes of `tf.TPUReplicatedInput` and
    `tf.TPUReplicatedOutput` respectively must match, excluding packed tensors.
    It is also assumed ops of the same TPU computation do not have ops outside
    of the TPU computation that are both inputs and outputs to the same TPU
    computation. Furthermore, we assume that every node has either none or both
    of `_replication_info` and `_xla_compile_device_type` attributes defined.

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
      "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "cluster", num_relicas = 1, num_cores_per_replica = 1, topology = "<topology>", device_assignment = [], padding_map = []} : () -> ()
      %replicated_input = "tf.TPUReplicatedInput"(%arg0) : (tensor<i32>) -> tensor<i32>
      %identity = "tf.Identity"(%replicated_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i32>) -> tensor<i32>
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
      }) {_xla_compile_device_type = "TPU", _replication_info = "cluster", num_cores_per_replica = 1, topology = "topology", device_assignment = [], padding_map = []} : () -> (tensor<i32>)
      return %cluster : tensor<i32>
    }
    ```

    The following replicated computation:

    ```mlir
    func @tpu_computation(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
      "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "cluster", num_relicas = 2, num_cores_per_replica = 1, topology = "topology", device_assignment = [], padding_map = []} : () -> ()
      %replicated_input = "tf.TPUReplicatedInput"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %identity = "tf.Identity"(%replicated_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : (tensor<i32>) -> tensor<i32>
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
        }) {_xla_compile_device_type = "TPU", _replication_info = "cluster", num_cores_per_replica = 1, topology = "topology", device_assignment = [], padding_map = []} : () -> (tensor<i32>)
        tf_device.return %cluster : tensor<i32>
      }
      return %replicate#0, %replicate#1 : tensor<i32>, tensor<i32>
    }
    ```
  }];

  let constructor = "tensorflow::tf2xla::internal::CreateTPUClusterFormationPass()";
}

def ExtractHeadTailOutsideCompilationPass : Pass<"tf-extract-head-tail-outside-compilation", "ModuleOp"> {
  let summary = "Extracts head or tail outside compilation to separate host launches before/after device cluster.";

  let description = [{
    This pass extracts a CPU computation cluster with `_xla_outside_compilation`
    annotation from the head or tail of a Device cluster.

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

  let constructor = "tensorflow::tf2xla::internal::CreateExtractHeadTailOutsideCompilationPass()";
}

def ExtractOutsideCompilationPass : Pass<"tf-extract-outside-compilation", "ModuleOp"> {
  let summary = "Extracts device outside compilation computation to a separate tf_device.parallel_execute region.";

  let description = [{
    This pass extracts a CPU computation cluster with `_xla_outside_compilation`
    annotation, which denotes ops that should be run on CPU/host, from a device cluster.
    Each outside compilation cluster is moved to
    a tf_device.parallel_execute region. The device cluster is also moved to a
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
          %1 = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
          %2 = "tf._XlaRecvAtHost"(%1) {device_ordinal = 0 : i64, key = "host_compute_channel_0_0_args"} : (tensor<3x!tf_type.string>) -> tensor<f32>
          %3 = "tf.Identity"(%2) : (tensor<f32>) -> tensor<f32>
          "tf._XlaSendFromHost"(%3, %1) {device_ordinal = 0 : i64, key = "host_compute_channel_0_0_retvals"} : (tensor<f32>, tensor<3x!tf_type.string>) -> ()
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

  let constructor = "tensorflow::tf2xla::internal::CreateExtractOutsideCompilationPass()";
}

def XlaClusterFormationPass : Pass<"tf-xla-cluster-formation", "ModuleOp"> {
  let summary = "Encapsulate partitioned calls within a Cluster op";
  let description = [{
    This pass clusters `tf.PartitionedCall` and `tf.StatefulPartitionedCall`
    with `_xla_compile_device_type` attribute into a `tf_device.cluster`.
    Notice this pass will only rewrite the outermost call if there are nested
    calls to avoid nested `tf.XlaLaunch` operations from being created later.

    For example, the following code

    ```mlir
    func.func @main() -> tensor<i32> {
      %0 = "tf.StatefulPartitionedCall"() {_xla_compile_device_type = "CPU", f = @stateful_pcall_func} : () -> (tensor<i32>)
      func.return %0 : tensor<i32>
    }

    func.func @stateful_pcall_func() -> tensor<i32> {
      %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      func.return %0 : tensor<i32>
    }
    ```

    will be transformed into,

    ```mlir
    func.func @main() -> tensor<i32> {
      %0 = "tf_device.cluster"() ({
        %1 = "tf.StatefulPartitionedCall"() {_xla_compile_device_type = "CPU", f = @stateful_pcall_func} : () -> tensor<i32>
        tf_device.return %1 : tensor<i32>
      }) : () -> tensor<i32>
      func.return %0 : tensor<i32>
    }

    func.func @stateful_pcall_func() -> tensor<i32> {
      %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      func.return %0 : tensor<i32>
    }

    ```
  }];
  let constructor = "tensorflow::tf2xla::internal::CreateXlaClusterFormationPass()";
  let dependentDialects = ["mlir::tf_device::TensorFlowDeviceDialect"];
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

  let constructor = "tensorflow::tf2xla::internal::CreateMarkOpsForOutsideCompilationPass()";
}

def HoistBroadcastReadPass : Pass<"tf-hoist-broadcast-read", "mlir::func::FuncOp"> {
  let summary = "Hoist reads out of a replicate that are on a resource that is broacast to all replicas.";

  let description = [{
    Some `ReadVariableOp`s that are within a `tf_device.replicate` read the same
    value across all replicas. These reads can be hoisted out of the
    `tf_device.replicate` so there's one read for all replicas, and each replica
    depends on the result of the read. This transform enables the
    xla-broadcast-pass to optimize the broadcast value.

    For example, the following:

    ```mlir
    tf_device.replicate {n = 2 : i32} {
      %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
      "tf.OpA"(%0) : (tensor<f32>) -> ()
    }
    ```

    will be transformed into:

    ``mlir
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    tf_device.replicate {n = 2 : i32} {
      "tf.OpA"(%0) : (tensor<f32>) -> ()
    }
    ```

    We must ensure that there is a single underlying resource that not
    distributed across replicas. There is a single underlying resource when the
    resource device type is CPU, so we cautiously only apply in this case.

    To be cautious we never hoist a read that comes after a write to the same
    resource.
  }];

  let constructor = "tensorflow::tf2xla::internal::CreateHoistBroadcastReadPass()";
}

def XlaBroadcastPass : Pass<"tf-xla-broadcast", "mlir::func::FuncOp"> {
  let summary = "Moves a broadcast from host into XLA, encoded as XlaAllReduce";

  let description = [{
    This pass moves brodcasts from host TF ops into XLA. This enables use of
    the inter-device network, which is faster than the inter-host network.
    Broadcasts in XLA are encoded as XlaAllReduce. An all_reduce with all 0
    inputs except for one real input produces the same result as a broadcast.

    For example, the following:

    ```mlir
    func.func @main(%arg0: tensor<f32>) -> () {
      tf_device.replicate {n = 2 : i32} {
        "tf_device.cluster"() ({
          "tf.OpA"(%arg0) : (tensor<f32>) -> ()
          tf_device.return
        }) : () -> ()
      }
      func.return
    }
    ```

    will be transformed into

    ```mlir
    func.func @main(%arg0: tensor<f32>) {
      %elem = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %shape = "tf.Const"() <{value = dense<> : tensor<0xi64>}> : () -> tensor<0xi64>
      %zero = "tf.Fill"(%shape, %elem) : (tensor<0xi64>, tensor<f32>) -> tensor<f32>
      tf_device.replicate([%arg0, %zero] as %arg1: tensor<f32>) {n = 2 : i32} {
        %arg1_id = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
          %1 = "tf.Identity"(%arg1) : (tensor<f32>) -> tensor<f32>
          tf_device.return %1 : tensor<f32>
        }) : () -> tensor<f32>
        "tf_device.cluster"() ({
          %group = "tf.Const"() <{value = dense<[[0, 1]]> : tensor<1x2xi32>}> : () -> tensor<1x2xi32>
          %arg1_reduced = "tf.XlaAllReduce"(%arg1_id, %group) <{mode = "CrossReplica", reduce_op = "Add"}> : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
          "tf.OpA"(%arg1_reduced) : (tensor<f32>) -> ()
          tf_device.return
        }) : () -> ()
        tf_device.return
      }
      return
    }
    ```
  }];

  let constructor = "tensorflow::tf2xla::internal::CreateXlaBroadcastPass()";
  let dependentDialects = ["mlir::tf_device::TensorFlowDeviceDialect"];
}
