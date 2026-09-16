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
// RUN: odml-to-stablehlo-opt %s --tf-stablehlo=skip-quantization-ops=true | FileCheck %s --check-prefix=CHECK-SKIP
// RUN: odml-to-stablehlo-opt %s --tf-stablehlo=skip-quantization-ops=false | FileCheck %s --check-prefix=CHECK-NOSKIP

func.func @fake_quant_with_min_max_vars(%arg0: tensor<1x1x28x48xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<1x1x28x48xf32> {
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {device = "", narrow_range = true, num_bits = 8 : i64} : (tensor<1x1x28x48xf32>, tensor<f32>, tensor<f32>) -> tensor<1x1x28x48xf32>
  func.return %0 : tensor<1x1x28x48xf32>
  // CHECK-SKIP: tf.FakeQuantWithMinMaxVars
  // CHECK-NOSKIP-NOT: tf.
}
