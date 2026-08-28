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
// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s


// CHECK:      attr {
// CHECK:        key: "dtypes"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:       type: DT_FLOAT
// CHECK-NEXT:       type: DT_INT16

// CHECK:      attr {
// CHECK-NEXT:   key: "shapes"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       shape {
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 3
// CHECK:            shape {
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 4
// CHECK-NEXT:         }
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: -1
// CHECK:            shape {
// CHECK-NEXT:         unknown_rank: true


func.func @main() {
  tf_executor.graph {
    %0:4 = tf_executor.island wraps "tf.InfeedDequeueTuple"() : () -> (tensor<3xi32>, tensor<4x?xf32>, tensor<*xi16>)
    tf_executor.fetch
  }
  func.return
}
