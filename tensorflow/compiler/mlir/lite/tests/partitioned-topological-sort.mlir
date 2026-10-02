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
// RUN: litert-opt %s -tfl-partitioned-topological-sort | FileCheck %s

// CHECK-LABEL: @tf_ops_will_be_partitioned
func.func @tf_ops_will_be_partitioned() -> tensor<1xf32> {
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %tmp1 = "tfl.add"(%const, %const) { fused_activation_function = "NONE" } : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
  %tmp2 = "tf.AddV2"(%tmp1, %tmp1) { device = "" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp3 = "tfl.add"(%const, %tmp2) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp4 = "tf.AddV2"(%tmp2, %tmp2) { device = "" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %result = "tfl.add"(%tmp3, %tmp4) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  func.return %result : tensor<1xf32>
}
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[TMP1:.*]] = tfl.add %[[CONST]], %[[CONST]]
// CHECK-NEXT: %[[TMP2:.*]] = "tf.AddV2"(%[[TMP1]], %[[TMP1]])
// CHECK-NEXT: %[[TMP4:.*]] = "tf.AddV2"(%[[TMP2]], %[[TMP2]])
// CHECK-NEXT: %[[TMP3:.*]] = tfl.add %[[CONST]], %[[TMP2]]
// CHECK-NEXT: %[[RESULT:.*]] = tfl.add %[[TMP3]], %[[TMP4]]
// CHECK-NEXT: return %[[RESULT]]

// CHECK-LABEL: @tf_ops_will_not_be_partitioned_if_not_schedulable
func.func @tf_ops_will_not_be_partitioned_if_not_schedulable() -> tensor<1xf32> {
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %tmp1 = "tfl.add"(%const, %const) { fused_activation_function = "NONE" } : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
  %tmp2 = "tf.AddV2"(%tmp1, %tmp1) { device = "" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp3 = "tfl.add"(%const, %tmp2) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp4 = "tf.AddV2"(%tmp2, %tmp3) { device = "" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %result = "tfl.add"(%tmp3, %tmp4) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  func.return %result : tensor<1xf32>
}
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[TMP1:.*]] = tfl.add %[[CONST]], %[[CONST]]
// CHECK-NEXT: %[[TMP2:.*]] = "tf.AddV2"(%[[TMP1]], %[[TMP1]])
// CHECK-NEXT: %[[TMP3:.*]] = tfl.add %[[CONST]], %[[TMP2]]
// CHECK-NEXT: %[[TMP4:.*]] = "tf.AddV2"(%[[TMP2]], %[[TMP3]])
// CHECK-NEXT: %[[RESULT:.*]] = tfl.add %[[TMP3]], %[[TMP4]]
// CHECK-NEXT: return %[[RESULT]]

// CHECK-LABEL: @wrapped_tf_ops_will_be_partitioned
func.func @wrapped_tf_ops_will_be_partitioned() -> tensor<1xf32> {
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %tmp1 = "tfl.add"(%const, %const) { fused_activation_function = "NONE" } : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
  %tmp2 = "tfl.custom_tf"(%tmp1, %tmp1) ({
    ^bb0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) :
      %wrap_result = "tf.AddV2"(%arg0, %arg1) { device = "" } : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
      "tfl.yield"(%wrap_result): (tensor<1xf32>)->()
      }): (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp3 = "tfl.add"(%const, %tmp2) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp4 = "tf.AddV2"(%tmp2, %tmp2) { device = "" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %result = "tfl.add"(%tmp3, %tmp4) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  func.return %result : tensor<1xf32>
}
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[TMP1:.*]] = tfl.add %[[CONST]], %[[CONST]]
// CHECK-NEXT: %[[TMP2:.*]] = "tfl.custom_tf"(%[[TMP1]], %[[TMP1]]) ({
// CHECK-NEXT:   ^bb0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>):
// CHECK-NEXT:   %[[WRAP_RESULT:.*]] = "tf.AddV2"(%arg0, %arg1)
// CHECK-NEXT:   "tfl.yield"(%[[WRAP_RESULT]])
// CHECK-NEXT: })
// CHECK-NEXT: %[[TMP4:.*]] = "tf.AddV2"(%[[TMP2]], %[[TMP2]])
// CHECK-NEXT: %[[TMP3:.*]] = tfl.add %[[CONST]], %[[TMP2]]
// CHECK-NEXT: %[[RESULT:.*]] = tfl.add %[[TMP3]], %[[TMP4]]
// CHECK-NEXT: return %[[RESULT]]

func.func @id(%arg0:tensor<1xf32>) -> tensor<1xf32> {
  func.return %arg0: tensor<1xf32>
}

// CHECK-LABEL: @tf_if_wont_be_partitioned
func.func @tf_if_wont_be_partitioned() -> tensor<1xf32> {
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %tmp1 = "tfl.add"(%const, %const) { fused_activation_function = "NONE" } : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
  %tmp2 = "tf.If"(%tmp1, %tmp1) { then_branch=@id, else_branch=@id, is_stateless = true } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp3 = "tfl.add"(%const, %tmp2) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp4 = "tf.AddV2"(%tmp2, %tmp2) { device = "" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %result = "tfl.add"(%tmp3, %tmp4) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  func.return %result : tensor<1xf32>
}
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[TMP1:.*]] = tfl.add %[[CONST]], %[[CONST]]
// CHECK-NEXT: %[[TMP2:.*]] = "tf.If"(%[[TMP1]], %[[TMP1]])
// CHECK-NEXT: %[[TMP3:.*]] = tfl.add %[[CONST]], %[[TMP2]]
// CHECK-NEXT: %[[TMP4:.*]] = "tf.AddV2"(%[[TMP2]], %[[TMP2]])
// CHECK-NEXT: %[[RESULT:.*]] = tfl.add %[[TMP3]], %[[TMP4]]
// CHECK-NEXT: return %[[RESULT]]

// CHECK-LABEL: @tfl_custom_flex_ops_will_be_partitioned
func.func @tfl_custom_flex_ops_will_be_partitioned() -> tensor<1xf32> {
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %tmp1 = "tfl.add"(%const, %const) { fused_activation_function = "NONE" } : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
  %tmp2 = "tfl.custom"(%tmp1, %tmp1) { custom_code="FlexSomething", custom_option=#tfl<const_bytes : "0x00"> } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp3 = "tfl.add"(%const, %tmp2) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp4 = "tfl.custom"(%tmp2, %tmp2) { custom_code="FlexSomething", custom_option=#tfl<const_bytes : "0x00"> } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %result = "tfl.add"(%tmp3, %tmp4) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  func.return %result : tensor<1xf32>
}
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[TMP1:.*]] = tfl.add %[[CONST]], %[[CONST]]
// CHECK-NEXT: %[[TMP2:.*]] = "tfl.custom"(%[[TMP1]], %[[TMP1]])
// CHECK-NEXT: %[[TMP4:.*]] = "tfl.custom"(%[[TMP2]], %[[TMP2]])
// CHECK-NEXT: %[[TMP3:.*]] = tfl.add %[[CONST]], %[[TMP2]]
// CHECK-NEXT: %[[RESULT:.*]] = tfl.add %[[TMP3]], %[[TMP4]]
// CHECK-NEXT: return %[[RESULT]]


// CHECK-LABEL: @tfl_custom_non_flex_ops_will_not_be_partitioned
func.func @tfl_custom_non_flex_ops_will_not_be_partitioned() -> tensor<1xf32> {
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %tmp1 = "tfl.add"(%const, %const) { fused_activation_function = "NONE" } : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
  %tmp2 = "tfl.custom"(%tmp1, %tmp1) { custom_code="SomethingElse", custom_option=#tfl<const_bytes : "0x00"> } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp3 = "tfl.add"(%const, %tmp2) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp4 = "tfl.custom"(%tmp2, %tmp2) { custom_code="FlexSomething", custom_option=#tfl<const_bytes : "0x00"> } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %result = "tfl.add"(%tmp3, %tmp4) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  func.return %result : tensor<1xf32>
}
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[TMP1:.*]] = tfl.add %[[CONST]], %[[CONST]]
// CHECK-NEXT: %[[TMP2:.*]] = "tfl.custom"(%[[TMP1]], %[[TMP1]])
// CHECK-NEXT: %[[TMP3:.*]] = tfl.add %[[CONST]], %[[TMP2]]
// CHECK-NEXT: %[[TMP4:.*]] = "tfl.custom"(%[[TMP2]], %[[TMP2]])
// CHECK-NEXT: %[[RESULT:.*]] = tfl.add %[[TMP3]], %[[TMP4]]
// CHECK-NEXT: return %[[RESULT]]


