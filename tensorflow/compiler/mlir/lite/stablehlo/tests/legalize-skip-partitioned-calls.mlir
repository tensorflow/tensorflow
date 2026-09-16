// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: odml-to-stablehlo-opt %s --tf-stablehlo=skip-partitioned-calls=true | FileCheck %s --check-prefix=CHECK-SKIP
// RUN: odml-to-stablehlo-opt %s --tf-stablehlo=skip-partitioned-calls=false | FileCheck %s --check-prefix=CHECK-NOSKIP

module {
  func.func @partitioned_call(%arg0: tensor<1x2x2x3xf32>) -> (tensor<1x2x2x3xf32>) {
    %0 = "tf.StatefulPartitionedCall"(%arg0) <{
      config = "", config_proto = "", executor_type = "", f = @some_func
    }> {
      _collective_manager_ids = [], device = ""
    } : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    // CHECK-SKIP: tf.StatefulPartitionedCall
    // CHECK-NOSKIP: call @some_func
    // CHECK-NOSKIP-NOT: tf.StatefulPartitionedCall
    %1 = "tf.PartitionedCall"(%0) <{
      config = "", config_proto = "", executor_type = "", f = @some_other_func
    }> {
      _collective_manager_ids = [], device = ""
    } : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    // CHECK-SKIP: tf.PartitionedCall
    // CHECK-NOSKIP: call @some_other_func
    // CHECK-NOSKIP-NOT: tf.PartitionedCall
    func.return %1: tensor<1x2x2x3xf32>
  }

  // CHECK-SKIP: func.func private @some_func
  func.func private @some_func(%arg0: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32> attributes {tf._noinline = true} {
    return %arg0 : tensor<1x2x2x3xf32>
  }

  // CHECK-SKIP: func.func private @some_other_func
  func.func private @some_other_func(%arg0: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32> attributes {tf._noinline = true} {
    return %arg0 : tensor<1x2x2x3xf32>
  }
}
