// RUN: tf-opt %s -tfl-quantize-variables | FileCheck %s


// CHECK-LABEL: QuantizeReadVariable
func.func @QuantizeReadVariable() -> (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) {
  %1 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  %2 = "tfl.read_variable"(%1) : (tensor<!tf_type.resource>) -> tensor<1x2x1x3xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>
  func.return %3 : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() {container = "", shared_name = ""} : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q:.*]] = "tfl.quantize"(%[[dq]]) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  return %[[q]] : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
}

// CHECK-LABEL: QuantizeAssignVariableWithDequantAndEqualType
func.func @QuantizeAssignVariableWithDequantAndEqualType(%arg0 : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>> {
  %0 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x2x1x3xf32>
  "tfl.assign_variable"(%0, %1) : (tensor<!tf_type.resource>, tensor<1x2x1x3xf32>) -> ()
  func.return %arg0 : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() {container = "", shared_name = ""} : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  "tfl.assign_variable"(%[[vh]], %arg0) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>, tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> ()
// CHECK-NEXT:  return %arg0 : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
}

// CHECK-LABEL: QuantizeAssignVariableWithDequantAndNotEqualType
func.func @QuantizeAssignVariableWithDequantAndNotEqualType(%arg0 : tensor<1x2x1x3x!quant.uniform<i8:f64, 1.0>>) -> tensor<1x2x1x3x!quant.uniform<i8:f64, 1.0>> {
  %1 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  %2 = "tfl.read_variable"(%1) : (tensor<!tf_type.resource>) -> tensor<1x2x1x3xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>
  %5 = "tfl.dequantize"(%arg0) : (tensor<1x2x1x3x!quant.uniform<i8:f64, 1.0>>) -> tensor<1x2x1x3xf32>
  "tfl.assign_variable"(%1, %5) : (tensor<!tf_type.resource>, tensor<1x2x1x3xf32>) -> ()
  func.return %arg0 : tensor<1x2x1x3x!quant.uniform<i8:f64, 1.0>>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() {container = "", shared_name = ""} : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q1:.*]] = "tfl.quantize"(%[[dq]]) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[q2:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>} : (tensor<1x2x1x3x!quant.uniform<i8:f64, 1.000000e+00>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  "tfl.assign_variable"(%[[vh]], %[[q2]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>, tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> ()
// CHECK-NEXT:  return %arg0 : tensor<1x2x1x3x!quant.uniform<i8:f64, 1.000000e+00>>
}

// CHECK-LABEL: QuantizeAssignVariableWithoutDequant
func.func @QuantizeAssignVariableWithoutDequant(%arg0 : tensor<1x2x1x3xf32>) -> tensor<1x2x1x3xf32> {
  %0 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  %1 = "tfl.read_variable"(%0) : (tensor<!tf_type.resource>) -> tensor<1x2x1x3xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>
  %3 = "tfl.dequantize"(%2) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x2x1x3xf32>
  "tfl.assign_variable"(%0, %3) : (tensor<!tf_type.resource>, tensor<1x2x1x3xf32>) -> ()
  func.return %arg0 : tensor<1x2x1x3xf32>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() {container = "", shared_name = ""} : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q:.*]] = "tfl.quantize"(%[[dq]]) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  "tfl.assign_variable"(%[[vh]], %[[q]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>, tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> ()
// CHECK-NEXT:  return %arg0 : tensor<1x2x1x3xf32>
}

// CHECK-LABEL: VarHandleCase
func.func @VarHandleCase(%arg0 : tensor<1x2x1x3xf32>) -> tensor<1x2x1x3xf32> {
  %0 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  func.return %arg0 : tensor<1x2x1x3xf32>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() {container = "", shared_name = ""} : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  return %arg0 : tensor<1x2x1x3xf32>
}

// CHECK-LABEL: QuantizeReadAssign
func.func @QuantizeReadAssign(%arg0: tensor<1x32x1x3xf32>) -> (tensor<1x34x1x3xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x32x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x32x1x3xf32>) -> tensor<1x32x1x3x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x32x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x32x1x3xf32>
  %cst = arith.constant dense<1> : tensor<4xi32>
  %cst_0 = arith.constant dense<[0, 0, 0, 3]> : tensor<4xi32>
  %cst_1 = arith.constant dense<[0, -2, 0, 0]> : tensor<4xi32>
  %2 = "tfl.var_handle"() {container = "", shared_name = "read_assign2/states"} : () -> tensor<!tf_type.resource>
  %3 = "tfl.read_variable"(%2) : (tensor<!tf_type.resource>) -> tensor<1x2x1x3xf32>
  %4 = "tfl.concatenation"(%3, %1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x1x3xf32>, tensor<1x32x1x3xf32>) -> tensor<1x34x1x3xf32>
  %5 = "tfl.quantize"(%4) {qtype = tensor<1x34x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x34x1x3xf32>) -> tensor<1x34x1x3x!quant.uniform<i8:f32, 1.0>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x34x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x34x1x3xf32>
  %7 = "tfl.strided_slice"(%6, %cst_1, %cst_0, %cst) {begin_mask = 13 : i32, ellipsis_mask = 0 : i32, end_mask = 15 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<1x34x1x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
  %8 = "tfl.quantize"(%7) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>
  %9 = "tfl.dequantize"(%8) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x2x1x3xf32>
  "tfl.assign_variable"(%2, %9) : (tensor<!tf_type.resource>, tensor<1x2x1x3xf32>) -> ()
  func.return %6 : tensor<1x34x1x3xf32>

// CHECK-NEXT:  %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x32x1x3x!quant.uniform<i8:f32, 1.000000e+00>>, volatile} : (tensor<1x32x1x3xf32>) -> tensor<1x32x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x32x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x32x1x3xf32>
// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<1> : tensor<4xi32>
// CHECK-NEXT:  %[[cst_0:.*]] = arith.constant dense<[0, 0, 0, 3]> : tensor<4xi32>
// CHECK-NEXT:  %[[cst_1:.*]] = arith.constant dense<[0, -2, 0, 0]> : tensor<4xi32>
// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() {container = "", shared_name = "read_assign2/states"} : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq2:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[cc:.*]] = "tfl.concatenation"(%[[dq2]], %[[dq1]]) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x1x3xf32>, tensor<1x32x1x3xf32>) -> tensor<1x34x1x3xf32>
// CHECK-NEXT:  %[[q2:.*]] = "tfl.quantize"(%[[cc]]) {qtype = tensor<1x34x1x3x!quant.uniform<i8:f32, 1.000000e+00>>, volatile} : (tensor<1x34x1x3xf32>) -> tensor<1x34x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq3:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x34x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x34x1x3xf32>
// CHECK-NEXT:  %[[ss:.*]] = "tfl.strided_slice"(%[[dq3]], %[[cst_1]], %[[cst_0]], %[[cst]]) {begin_mask = 13 : i32, ellipsis_mask = 0 : i32, end_mask = 15 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<1x34x1x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q3:.*]] = "tfl.quantize"(%[[ss]]) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  "tfl.assign_variable"(%[[vh]], %[[q3]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>, tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> ()
// CHECK-NEXT:  return %[[dq3]] : tensor<1x34x1x3xf32>
}
