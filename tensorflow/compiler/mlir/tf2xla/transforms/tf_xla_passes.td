/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

def LegalizeTFCommunicationPass : Pass<"xla-legalize-tf-communication", "ModuleOp"> {
  let summary = "Legalize TF/XLA communication ops (TensorFlow dialect) to the HLO "
           "dialect";
  let dependentDialects = ["mhlo::MhloDialect", "sparse_tensor::SparseTensorDialect"];
  let constructor = "::mlir::mhlo::CreateLegalizeTFCommunicationPass()";
  let description = [{
    A pass that legalizes TF/XLA communication ops, propagates their respective
    tokens (for ordering), and rewrites their respective functions and control
    flow ops when necessary.

    For example, given the program

    ```mlir
      func @send_to_host(%arg0: tensor<i32>) {
        "tf.XlaSendToHost"(%arg0) {key = "send_key"} : (tensor<i32>) -> ()
        return
      }
    ```

    This might be legalized like this:

    ```mlir
      func @send_to_host(%arg0: tensor<i32>) {
        %0 = "mhlo.create_token"() : () -> !mhlo.token
        %1 = "mhlo.send"(%arg0, %0) {...}} : (tensor<i32>, !mhlo.token) -> !mhlo.token
        return
      }
    '''
  }];
}
