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

func.func @add_const_custom_dense_i32_10(%c0: i32) -> i32 {
  %c1 = "test_custom.add.const.i32"(%c0) {value = dense<[0]> : tensor<1xi32>} : (i32) -> i32
  %c2 = "test_custom.add.const.i32"(%c1) {value = dense<[1]> : tensor<1xi32>} : (i32) -> i32
  %c3 = "test_custom.add.const.i32"(%c2) {value = dense<[2]> : tensor<1xi32>} : (i32) -> i32
  %c4 = "test_custom.add.const.i32"(%c3) {value = dense<[3]> : tensor<1xi32>} : (i32) -> i32
  %c5 = "test_custom.add.const.i32"(%c4) {value = dense<[4]> : tensor<1xi32>} : (i32) -> i32
  %c6 = "test_custom.add.const.i32"(%c5) {value = dense<[5]> : tensor<1xi32>} : (i32) -> i32
  %c7 = "test_custom.add.const.i32"(%c6) {value = dense<[6]> : tensor<1xi32>} : (i32) -> i32
  %c8 = "test_custom.add.const.i32"(%c7) {value = dense<[7]> : tensor<1xi32>} : (i32) -> i32
  %c9 = "test_custom.add.const.i32"(%c8) {value = dense<[8]> : tensor<1xi32>} : (i32) -> i32
  %c10 = "test_custom.add.const.i32"(%c9) {value = dense<[9]> : tensor<1xi32>} : (i32) -> i32
  func.return %c10 : i32
}
