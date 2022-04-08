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

// RUN: tf-quant-opt %s -quant-insert-quantized-functions | FileCheck %s

// Empty module
module {
  func.func @simple_fn(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}

// CHECK-NOT: func private @internal_rescale_fn
// CHECK-NOT: func private @internal_relu_fn
// CHECK-NOT: func private @internal_conv2d_fn
// CHECK-NOT: func private @internal_matmul_fn
// CHECK: func private @quantized_conv2d_with_bias_fn
// CHECK: func private @quantized_conv2d_with_bias_and_relu_fn
// CHECK: func private @quantized_conv2d_with_bias_and_relu6_fn
// CHECK: func private @quantized_matmul_with_bias_fn
// CHECK: func private @quantized_matmul_with_bias_and_relu_fn
// CHECK: func private @quantized_matmul_with_bias_and_relu6_fn
// CHECK: func private @quantize_i8
// CHECK: func private @dequantize_i8
