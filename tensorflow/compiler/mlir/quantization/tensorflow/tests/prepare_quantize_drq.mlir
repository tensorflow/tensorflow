// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-quant-opt %s -split-input-file -quant-prepare-quantize-drq | FileCheck %s

// -----

module {
  func.func @matmul(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn} : (tensor<1x2x2x3xf32>, tensor<2x3xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }
  func.func private @composite_matmul_fn(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x3xf32>, tensor<2x3xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

// CHECK-LABEL: func @matmul
// CHECK-DAG: %cst = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK: %0 = "quantfork.qcast"(%cst) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 3.9370078740157481E-9>>
// CHECK: %1 = "quantfork.dcast"(%0) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32, 3.9370078740157481E-9>>) -> tensor<2x3xf32>
// CHECK: %2 = "tf.PartitionedCall"(%arg0, %1) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn} : (tensor<1x2x2x3xf32>, tensor<2x3xf32>) -> tensor<*xf32>
// CHECK: return %2 : tensor<*xf32>

// CHECK-LABEL: func private @composite_matmul_fn
// CHECK: %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x3xf32>, tensor<2x3xf32>) -> tensor<*xf32>
// CHECK: return %0 : tensor<*xf32>
}

// -----
