// RUN: tf-opt -xla-hlo-cpu-fusion %s | FileCheck %s

// CHECK-LABEL: @mul_add_source
func @mul_add_source(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
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

// CHECK-LABEL: @mul_add_annotated
func @mul_add_annotated(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %cst = constant dense<0.0> : tensor<f32>
  %cst_0 = constant dense<255.0> : tensor<f32>
  %cst_1 = constant dense<8> : tensor<i32>
  %cst_2 = constant dense<false> : tensor<i1>
  %qin = "xla_hlo.custom_call"(%arg0, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars",
    has_side_effect = false, name = "custom-call.1"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
  %qw = "xla_hlo.custom_call"(%arg1, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars",
    has_side_effect = false, name = "custom-call.2"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
  %0 = "xla_hlo.multiply"(%qin, %qw) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = "xla_hlo.add"(%0, %arg2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} :  (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %r = "xla_hlo.custom_call"(%1, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars",
    has_side_effect = false, name = "custom-call.3"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>

// CHECK: %[[region:.*]] = "quant.region"
// CHECK: ^bb0(%arg3: tensor<2x4xf32>, %arg4: tensor<2x4xf32>, %arg5: tensor<2x4xf32>):	// no predecessors
// CHECK:   %[[mul:.*]] = xla_hlo.multiply %arg3, %arg4 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<2x4xf32>
// CHECK:   %[[add:.*]] = xla_hlo.add %[[mul]], %arg5 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<2x4xf32>
// CHECK:   "quant.return"(%[[add]]) : (tensor<2x4xf32>) -> ()
// CHECK: }) {input_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>, !quant.uniform<i8:f32, 1.000000e+00:-128>, f32],
// CHECK-SAME: logical_kernel = "generic.mul_add", output_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>]} :
// CHECK-SAME: (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:  %[[r:.*]] = "xla_hlo.custom_call"(%[[region]]
// CHECK: return %[[r]] : tensor<2x4xf32>
}


