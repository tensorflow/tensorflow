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
