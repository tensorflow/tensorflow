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
// RUN: odml-to-stablehlo-opt %s -tfl-parse-stablehlo-ops | FileCheck -dump-input always %s

module {
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tfl.custom"(%arg0) {custom_code = "stablehlo.rsqrt", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}

// CHECK:       module
// CHECK-NEXT:  func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:  %0 = stablehlo.rsqrt %arg0 : tensor<2xf32>
// CHECK-NEXT:  return %0 : tensor<2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
