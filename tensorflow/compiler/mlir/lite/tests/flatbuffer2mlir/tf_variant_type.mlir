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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

func.func @main(%arg0 : tensor<!tf_type.variant<tensor<2xi32>>>, %arg1: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>> {
  func.return %arg0 : tensor<!tf_type.variant<tensor<2xi32>>>
}

// CHECK:          func.func @main(%[[ARG0:.*]]: tensor<!tf_type.variant<tensor<2xi32>>>, %[[ARG1:.*]]: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>>
// CHECK-NEXT:       return %[[ARG0]] : tensor<!tf_type.variant<tensor<2xi32>>>
