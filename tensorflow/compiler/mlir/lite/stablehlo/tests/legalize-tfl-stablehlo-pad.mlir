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
    func.func @main(%arg0: tensor<8x128xf32>, %arg1: tensor<f32>) -> tensor<11x131xf32> {
      %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.pad", custom_option = #tfl<const_bytes : "0x656467655F70616464696E675F6869676800020203656467655F70616464696E675F6C6F7700020100696E746572696F725F70616464696E6700020000033E2A17030103311E0B2C2C2C062401">} : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
      func.return %0 : tensor<11x131xf32>
    }
  }

// CHECK:       module {
// CHECK-NEXT:    func @main(%arg0: tensor<8x128xf32>, %arg1: tensor<f32>) -> tensor<11x131xf32> {
// CHECK-NEXT:      %0 = stablehlo.pad %arg0, %arg1, low = [1, 0], high = [2, 3], interior = [0, 0] : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
// CHECK-NEXT:      return %0 : tensor<11x131xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

