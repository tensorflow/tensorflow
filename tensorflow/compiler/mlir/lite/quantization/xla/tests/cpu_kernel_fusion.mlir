// RUN: tf-opt -xla-hlo-cpu-fusion %s | FileCheck %s

// CHECK-LABEL: @mul_add
func @mul_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %0 = "xla_hlo.multiply"(%arg0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "xla_hlo.add"(%0, %arg2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} :  (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>

// CHECK: %[[region:.*]] = "quant.region"(%arg0, %arg1, %arg2) ( {
// CHECK: ^bb0(%arg3: tensor<4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<4xf32>):	// no predecessors
// CHECK:   %[[mul:.*]] = xla_hlo.multiply %arg3, %arg4 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
// CHECK:   %[[add:.*]] = xla_hlo.add %[[mul]], %arg5 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
// CHECK:   "quant.return"(%[[add]]) : (tensor<4xf32>) -> ()
// CHECK: }) {input_specs = [f32, f32, f32], logical_kernel = "generic.mul_add", output_specs = [f32]} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[region]] : tensor<4xf32>
}
