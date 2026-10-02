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
// RUN: odml-to-stablehlo-opt %s -tfl-parse-stablehlo-ops | FileCheck %s

module {
func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.concatenate", custom_option = #tfl<const_bytes : "0x64696D656E73696F6E00010B0101010004022401">} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  func.return %0 : tensor<6x3xf32>
}
}

// CHECK:       module {
// CHECK-NEXT:  func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:    return %0 : tensor<6x3xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
