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

def TestClustering : FunctionPass<"tf-cpurt-test-clustering"> {
  let summary = "Creates `tf_device.cluster` operations according to the TF "
                "CPURT clustering policy. Add value constraints remarks to the "
                "function inputs inferred from the clustering policy.";

  let options = [
   Option<"min_cluster_size", "min-cluster-size", "int" , /*default=*/"1",
          "Do not form clusters smaller of the given size.">,
  ];

  let constructor = "tensorflow::CreateTestTfCpurtClusteringPass()";

  let dependentDialects = ["mlir::tf_device::TensorFlowDeviceDialect"];
}

def TestClusteringPolicy : FunctionPass<"tf-cpurt-test-clustering-policy"> {
  let summary = "Annotates operations with clustering constraints inferred "
                "from the TF CPURT clustering policies";
  let constructor = "tensorflow::CreateTestTfCpurtClusteringPolicyPass()";
}
