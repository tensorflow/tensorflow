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

def LegalizeTFWithTF2XLA
    : Pass<"xla-legalize-tf-with-tf2xla", "mlir::func::FuncOp"> {
  let summary = "Legalize from TensorFlow to the HLO dialect using tf2xla kernels";
  let dependentDialects = ["mhlo::MhloDialect"];
  let constructor = "::mlir::mhlo::createLegalizeTfWithTf2XlaPass()";
  let options = [
  // TODO(hinsu): Support finer grained device type assignment instead of a
  // global device type for all TensorFlow ops.
  Option<"device_type_", "device-type", "std::string", "",
         "XLA device type for execution of TensorFlow ops.">,
  Option<"prefer_tf2xla_", "prefer-tf2xla", "bool", "",
         "Enable legalization when it is not in the list of "
                     "MLIR-legalized ops.">,
  Option<"legalize_test_only_ops_", "legalize-test-only-ops", "bool", "",
         "Enable tf2xla legalizations for some ops that are "
                     "enabled only for testing.">
  ];
}

def LegalizeTFCommunicationPass : Pass<"xla-legalize-tf-communication", "ModuleOp"> {
  let summary = "Legalize TF/XLA communication ops (TensorFlow dialect) to the HLO "
           "dialect";
  let dependentDialects = ["mhlo::MhloDialect"];
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
