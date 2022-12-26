// RUN: tf-opt %s -tfl-quantize-variables | FileCheck %s

// CHECK-LABEL: QuantizeReadAssign
func.func @QuantizeReadAssign(%arg0: tensor<1x32x1x3xf32>) -> (tensor<1x34x1x3xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x32x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>, volatile} : (tensor<1x32x1x3xf32>) -> tensor<1x32x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x32x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>) -> tensor<1x32x1x3xf32>
  %cst = arith.constant dense<1> : tensor<4xi32>
  %cst_0 = arith.constant dense<[0, 0, 0, 3]> : tensor<4xi32>
  %cst_1 = arith.constant dense<[0, -2, 0, 0]> : tensor<4xi32>
  %2 = "tfl.var_handle"() {container = "", shared_name = "read_assign2/states"} : () -> tensor<!tf_type.resource>
  %3 = "tfl.read_variable"(%2) : (tensor<!tf_type.resource>) -> tensor<1x2x1x3xf32>
  %4 = "tfl.concatenation"(%3, %1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x1x3xf32>, tensor<1x32x1x3xf32>) -> tensor<1x34x1x3xf32>
  %5 = "tfl.quantize"(%4) {qtype = tensor<1x34x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>, volatile} : (tensor<1x34x1x3xf32>) -> tensor<1x34x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x34x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>) -> tensor<1x34x1x3xf32>
  %7 = "tfl.strided_slice"(%6, %cst_1, %cst_0, %cst) {begin_mask = 13 : i32, ellipsis_mask = 0 : i32, end_mask = 15 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<1x34x1x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
  %8 = "tfl.quantize"(%7) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>
  %9 = "tfl.dequantize"(%8) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>) -> tensor<1x2x1x3xf32>
  "tfl.assign_variable"(%2, %9) : (tensor<!tf_type.resource>, tensor<1x2x1x3xf32>) -> ()
  func.return %6 : tensor<1x34x1x3xf32>

// TODO(b/261940892): Add more compact and intuitive unit-tests
// CHECK-NEXT:  %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x32x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>, volatile} : (tensor<1x32x1x3xf32>) -> tensor<1x32x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>
// CHECK-NEXT:  %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x32x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>) -> tensor<1x32x1x3xf32>
// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<1> : tensor<4xi32>
// CHECK-NEXT:  %[[cst_0:.*]] = arith.constant dense<[0, 0, 0, 3]> : tensor<4xi32>
// CHECK-NEXT:  %[[cst_1:.*]] = arith.constant dense<[0, -2, 0, 0]> : tensor<4xi32>
// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() {container = "", shared_name = "read_assign2/states"} : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>
// CHECK-NEXT:  %[[dq2:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[cc:.*]] = "tfl.concatenation"(%[[dq2]], %[[dq1]]) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x1x3xf32>, tensor<1x32x1x3xf32>) -> tensor<1x34x1x3xf32>
// CHECK-NEXT:  %[[q2:.*]] = "tfl.quantize"(%[[cc]]) {qtype = tensor<1x34x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>, volatile} : (tensor<1x34x1x3xf32>) -> tensor<1x34x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>
// CHECK-NEXT:  %[[dq3:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x34x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>) -> tensor<1x34x1x3xf32>
// CHECK-NEXT:  %[[ss:.*]] = "tfl.strided_slice"(%[[dq3]], %[[cst_1]], %[[cst_0]], %[[cst]]) {begin_mask = 13 : i32, ellipsis_mask = 0 : i32, end_mask = 15 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<1x34x1x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q3:.*]] = "tfl.quantize"(%[[ss]]) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>
// CHECK-NEXT:  "tfl.assign_variable"(%[[vh]], %[[q3]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>>>, tensor<1x2x1x3x!quant.uniform<i8:f32, 0.0078238183632493019:-1>>) -> ()
// CHECK-NEXT:  return %[[dq3]] : tensor<1x34x1x3xf32>
}

