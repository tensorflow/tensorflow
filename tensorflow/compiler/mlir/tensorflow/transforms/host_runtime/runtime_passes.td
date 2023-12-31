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

def TPURewritePass : Pass<"tf-tpu-rewrite", "mlir::ModuleOp"> {
  let summary = "Rewrites a `tf_device.cluster_func` on TPUs into TPU runtime operations.";

  let description = [{
    This pass rewrites a `tf_device.cluster_func` operation into a sequence of `tf._TPUCompileMlir`
    and `tf.TPUExecute` operations. `tf._TPUCompileMlir` contains a MLIR module that is
    functionally equivalent to the function referenced by `tf_device.cluster_func`.
    This makes the module to be jit-compiled and executed on TPU.
    If it is not possible to rewrite the operation or device assignment fails,
    a failure will be returned.

    Note, many parameters to the `tf_device.cluster_func` are omitted in this
    and following examples.
    For example, a non replicated `tf_device.cluster_func`:

    ```mlir
    func @tf_tpu_rewrite(%arg0: tensor<i8>) {
      %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @func} : (tensor<i8>) -> tensor<i8>
      return
    }
    ```

    will be rewritten as:

    ```mlir
    func @tf_tpu_rewrite(%arg0: tensor<i8>) {
      %0:2 = "tf_device.launch"() ( {
        %compilation_status, %program = "tf._TPUCompileMlir"() {mlir_module = "<serialized func>"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
        tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
      }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
      "tf_device.launch"() ( {
        "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
        tf_device.return
      }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
      %1 = "tf_device.launch"() ( {
        %2 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<i8>, tensor<3x!tf_type.string>) -> tensor<i8>
        tf_device.return %2 : tensor<i8>
      }) {device = "/job:worker/replica:0/task:0/device:TPU:0"} : () -> tensor<i8>
      return
    }
    ```

    A replicated `tf_device.cluster_func`:

    ```mlir
    func @tf_tpu_rewrite(%arg0: tensor<i8>, %arg1: tensor<i8>) {
      %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<i8>) {n = 2 : i32} {
        %1 = "tf_device.cluster_func"(%ri) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @func} : (tensor<i8>) -> tensor<i8>
        tf_device.return %1 : tensor<i8>
      }
      return
    }
    ```

    will be rewritten as:

    ```mlir
    func @tf_tpu_rewrite(%arg0: tensor<i8>, %arg1: tensor<i8>) {
      %0:2 = tf_device.replicate([%arg0, %arg1] as %arg2: tensor<i8>) {devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"], TPU_REPLICATED_HOST_0 = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:CPU:0"]}, n = 2 : i32} {
        %1:2 = "tf_device.launch"() ( {
          %compilation_status, %program = "tf._TPUCompileMlir"() {mlir_module = "<serialized func>"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
          tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
        "tf_device.launch"() ( {
          "tf.TPUCompileSucceededAssert"(%1#0) : (tensor<!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
        %2 = "tf_device.launch"() ( {
          %3 = "tf.TPUExecute"(%arg2, %1#1) : (tensor<i8>, tensor<3x!tf_type.string>) -> tensor<i8>
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
      %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @func, num_cores_per_replica = 2, input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"]} : (tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    ```

    will be rewritten as:

    ```mlir
    func @tf_tpu_rewrite(%arg0: tensor<8xi32>) -> tensor<8xi32> {
      %0:3 = "tf_device.launch"() ( {
        %compilation_status, %program:2 = "tf._TPUCompileMlir"() {mlir_module = "<serialized func>"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>)
        tf_device.return %compilation_status, %program#0, %program#1 : tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>
      }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>)
      "tf_device.launch"() ( {
        "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
        tf_device.return
      }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
      %1 = "tf_device.parallel_execute"() ( {
        %2 = "tf_device.launch"() ( {
          %3 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<8xi32>, tensor<3x!tf_type.string>) -> tensor<8xi32>
          tf_device.return %3 : tensor<8xi32>
        }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> tensor<8xi32>
        tf_device.return %2 : tensor<8xi32>
      },  {
        "tf_device.launch"() ( {
          "tf.TPUExecute"(%0#2) : (tensor<3x!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:localhost/replica:0/task:0/device:TPU:1"} : () -> ()
        tf_device.return
      }) : () -> tensor<8xi32>
      return %1 : tensor<8xi32>
    }
    ```
  }];

  let options = [
    Option<"tpu_compile_metadata_debug_", "tpu-compile-metadata-debug", "bool", /*default=*/"false",
           "Whether to serialize TPUCompileMetadataProto metadata in 'tf._TPUCompileMlir' op as a proto debug string">
  ];

  let dependentDialects = [
    "mlir::mhlo::MhloDialect",
    "mlir::tf_device::TensorFlowDeviceDialect"
  ];
  let constructor = "mlir::TFTPU::CreateTPURewritePass()";
}