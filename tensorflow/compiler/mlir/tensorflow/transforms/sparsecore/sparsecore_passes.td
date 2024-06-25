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

def EmbeddingPipeliningPass : Pass<"tf-embedding-pipelining", "mlir::ModuleOp"> {
  let summary = "Rewrite graph for embedding pipelining";
  let constructor = "TFDevice::CreateEmbeddingPipeliningPass()";
    let description = [{
    For architectures that support accelerated embedding lookups, this pass will
    rewrite the graph to use pipelining for better device utilization.
  }];
}

def EmbeddingSequencingPass : Pass<"tf-embedding-sequencing", "mlir::ModuleOp"> {
  let summary = "Rewrite graph for sequential execution of embeddings";
  let constructor = "TFDevice::CreateEmbeddingSequencingPass()";
    let description = [{
    This is a strictly sequential and formally correct fallback option for the
    embedding pipelining pass intended for debugging during pipelining
    development.
  }];
}

def EmbeddingProgramKeyPass : Pass<"tf-embedding-program-key", "mlir::func::FuncOp"> {
  let summary = "Sets the program key for embedding ops.";
  let constructor = "TFDevice::CreateEmbeddingProgramKeyPass()";
    let description = [{
    Passes in the program key to embedding ops. Will move the embedding ops
    after a _TPUCompileMlir op if there is no predecessor _TPUCompileMlir op.
    Both the embedding op and compile op are assumed to be wrapped in separate
    tf_device.launch() ops. This is because the embedding op is head outside
    compiled and the compile op is wrapped in launch to execute on host
    during TPURewritePass.

    For example, the tf.OpA with the `mini_batch_splits` attribute will be
    moved after _TPUCompileMlir and the first input will use the
    _TPUCompileMlir program output:

    ```mlir
    "tf_device.launch"() ({
     %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
     "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> ()
     tf_device.return
   }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
   %0:2 = "tf_device.launch"() ({
     %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
     tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
   }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
   ```

   becomes:

   ```mlir
     %0:2 = "tf_device.launch"() ({
       %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
       tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
     }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
     "tf_device.launch"() ({
       %cst = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
       "tf.OpA"(%0#1) {mini_batch_splits = ""} : (tensor<3x!tf_type.string>) -> ()
       tf_device.return
     }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
   ```
  }];

  let dependentDialects = [
    "mhlo::MhloDialect",
    "tf_device::TensorFlowDeviceDialect"
  ];
}
