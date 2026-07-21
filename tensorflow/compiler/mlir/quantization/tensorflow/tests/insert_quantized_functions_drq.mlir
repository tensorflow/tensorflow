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
// RUN: tf-quant-opt %s -quant-insert-quantized-functions='quantization-method=drq' | FileCheck %s
// RUN: tf-quant-opt %s -quant-insert-quantized-functions='quantization-method=drq target-opset=UNIFORM_QUANTIZED' | FileCheck --check-prefix=UQ-CHECK %s

// Empty module
module {
  func.func @simple_fn(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}

// CHECK-NOT: func private @internal_calculate_quant_params
// CHECK-NOT: func private @internal_dequantize_i32
// CHECK-NOT: func private @internal_quantize_i8
// CHECK-NOT: func private @internal_matmul_fn
// CHECK: func private @quantized_matmul_fn
// CHECK-SAME: tf_quant.quantized_ops = ["MatMul"]
// CHECK: func private @quantized_conv2d_fn
// CHECK-SAME: tf_quant.quantized_ops = ["Conv2D"]
// CHECK: func private @quantized_depthwise_conv2d_fn
// CHECK-SAME: tf_quant.quantized_ops = ["DepthwiseConv2D"]

// UQ-CHECK: func private @quantized_conv2d_fn
// UQ-CHECK: func private @quantized_depthwise_conv2d_fn
// UQ-CHECK: func private @quantized_matmul_fn
