// RUN: tf-opt %s -tfl-quantize-variables | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize -tfl-quantize -tfl-post-quantize -tfl-quantize-variables -tfl-quantize -tfl-post-quantize | FileCheck --check-prefix=WHOLE-PASSES %s

// CHECK-LABEL: QuantizeReadVariable
func.func @QuantizeReadVariable() -> (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) {
  %1 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  %2 = "tfl.read_variable"(%1) : (tensor<!tf_type.resource>) -> tensor<1x2x1x3xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>
  func.return %3 : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() <{container = "", shared_name = ""}> : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q:.*]] = "tfl.quantize"(%[[dq]]) <{qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>}> {volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  return %[[q]] : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
}

// CHECK-LABEL: QuantizeAssignVariableWithDequantAndEqualType
func.func @QuantizeAssignVariableWithDequantAndEqualType(%arg0 : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>> {
  %0 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x2x1x3xf32>
  "tfl.assign_variable"(%0, %1) : (tensor<!tf_type.resource>, tensor<1x2x1x3xf32>) -> ()
  func.return %arg0 : tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() <{container = "", shared_name = ""}> : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
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

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() <{container = "", shared_name = ""}> : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q1:.*]] = "tfl.quantize"(%[[dq]]) <{qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>}> {volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[q2:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>}> : (tensor<1x2x1x3x!quant.uniform<i8:f64, 1.000000e+00>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
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

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() <{container = "", shared_name = ""}> : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q:.*]] = "tfl.quantize"(%[[dq]]) <{qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>}> {volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  "tfl.assign_variable"(%[[vh]], %[[q]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>, tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> ()
// CHECK-NEXT:  return %arg0 : tensor<1x2x1x3xf32>
}

// CHECK-LABEL: VarHandleCase
func.func @VarHandleCase(%arg0 : tensor<1x2x1x3xf32>) -> tensor<1x2x1x3xf32> {
  %0 = "tfl.var_handle"() : () -> tensor<!tf_type.resource>
  func.return %arg0 : tensor<1x2x1x3xf32>

// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() <{container = "", shared_name = ""}> : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
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
  %7 = "tfl.strided_slice"(%6, %cst_1, %cst_0, %cst) {begin_mask = 13 : i32, ellipsis_mask = 0 : i32, end_mask = 15 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<1x34x1x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
  %8 = "tfl.quantize"(%7) {qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>, volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>
  %9 = "tfl.dequantize"(%8) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x2x1x3xf32>
  "tfl.assign_variable"(%2, %9) : (tensor<!tf_type.resource>, tensor<1x2x1x3xf32>) -> ()
  func.return %6 : tensor<1x34x1x3xf32>

// CHECK-NEXT:  %[[q1:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x32x1x3x!quant.uniform<i8:f32, 1.000000e+00>>}> {volatile} : (tensor<1x32x1x3xf32>) -> tensor<1x32x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x32x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x32x1x3xf32>
// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<1> : tensor<4xi32>
// CHECK-NEXT:  %[[cst_0:.*]] = arith.constant dense<[0, 0, 0, 3]> : tensor<4xi32>
// CHECK-NEXT:  %[[cst_1:.*]] = arith.constant dense<[0, -2, 0, 0]> : tensor<4xi32>
// CHECK-NEXT:  %[[vh:.*]] = "tfl.var_handle"() <{container = "", shared_name = "read_assign2/states"}> : () -> tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>
// CHECK-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq2:.*]] = "tfl.dequantize"(%[[rv]]) : (tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[cc:.*]] = "tfl.concatenation"(%[[dq2]], %[[dq1]]) <{axis = 1 : i32, fused_activation_function = "NONE"}> : (tensor<1x2x1x3xf32>, tensor<1x32x1x3xf32>) -> tensor<1x34x1x3xf32>
// CHECK-NEXT:  %[[q2:.*]] = "tfl.quantize"(%[[cc]]) <{qtype = tensor<1x34x1x3x!quant.uniform<i8:f32, 1.000000e+00>>}> {volatile} : (tensor<1x34x1x3xf32>) -> tensor<1x34x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  %[[dq3:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x34x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x34x1x3xf32>
// CHECK-NEXT:  %[[ss:.*]] = "tfl.strided_slice"(%[[dq3]], %[[cst_1]], %[[cst_0]], %[[cst]]) <{begin_mask = 13 : i32, ellipsis_mask = 0 : i32, end_mask = 15 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<1x34x1x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT:  %[[q3:.*]] = "tfl.quantize"(%[[ss]]) <{qtype = tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>}> {volatile} : (tensor<1x2x1x3xf32>) -> tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK-NEXT:  "tfl.assign_variable"(%[[vh]], %[[q3]]) : (tensor<*x!tf_type.resource<tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>>>, tensor<1x2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> ()
// CHECK-NEXT:  return %[[dq3]] : tensor<1x34x1x3xf32>
}

// WHOLE-PASSES-LABEL: QuantizeConvVariable
func.func @QuantizeConvVariable(%arg0: tensor<1x3x1x1xf32>) -> (tensor<1x3x1x1xf32>) {
  %cst = arith.constant dense<1> : tensor<4xi32>
  %cst_0 = arith.constant dense<[0, 3, 0, 1]> : tensor<4xi32>
  %cst_1 = arith.constant dense<0> : tensor<4xi32>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<1xf32>
  %0 = "tfl.quantize"(%cst_2) {qtype = tensor<1x!quant.uniform<i32:f32, 1.0>>, volatile} : (tensor<1xf32>) -> tensor<1x!quant.uniform<i32:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x!quant.uniform<i32:f32, 1.0>>) -> tensor<1xf32>
  %cst_3 = arith.constant dense<[[[[1.0]], [[1.0]], [[1.0]]]]> : tensor<1x3x1x1xf32>
  %2 = "tfl.quantize"(%cst_3) {qtype = tensor<1x3x1x1x!quant.uniform<i8<-127:127>:f32:0, {1.0}>>, volatile} : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1x!quant.uniform<i8<-127:127>:f32:0, {1.0}>>
  %3 = "tfl.dequantize"(%2) : (tensor<1x3x1x1x!quant.uniform<i8<-127:127>:f32:0, {1.0}>>) -> tensor<1x3x1x1xf32>
  %4 = "tfl.quantize"(%arg0) {qtype = tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:6>>, volatile} : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:6>>
  %5 = "tfl.dequantize"(%4) : (tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:6>>) -> tensor<1x3x1x1xf32>
  %6 = "tfl.var_handle"() {container = "", shared_name = "conv_variable/state"} : () -> tensor<!tf_type.resource>
  %7 = "tfl.read_variable"(%6) : (tensor<!tf_type.resource>) -> tensor<1x3x1x1xf32>
  %8 = "tfl.conv_2d"(%5, %3, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1xf32>) -> tensor<1x3x1x1xf32>
  %9 = "tfl.quantize"(%8) {qtype = tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:2>>, volatile} : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:2>>
  %10 = "tfl.dequantize"(%9) : (tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:2>>) -> tensor<1x3x1x1xf32>
  %11 = "tfl.concatenation"(%7, %10) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>) -> tensor<1x6x1x1xf32>
  %12 = "tfl.quantize"(%11) {qtype = tensor<1x6x1x1x!quant.uniform<i8:f32, 1.0:2>>, volatile} : (tensor<1x6x1x1xf32>) -> tensor<1x6x1x1x!quant.uniform<i8:f32, 1.0:2>>
  %13 = "tfl.dequantize"(%12) : (tensor<1x6x1x1x!quant.uniform<i8:f32, 1.0:2>>) -> tensor<1x6x1x1xf32>
  %14 = "tfl.strided_slice"(%13, %cst_1, %cst_0, %cst) {begin_mask = 15 : i32, ellipsis_mask = 0 : i32, end_mask = 13 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<1x6x1x1xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x3x1x1xf32>
  %15 = "tfl.quantize"(%14) {qtype = tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:2>>, volatile} : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:2>>
  %16 = "tfl.dequantize"(%15) : (tensor<1x3x1x1x!quant.uniform<i8:f32, 1.0:2>>) -> tensor<1x3x1x1xf32>
  "tfl.assign_variable"(%6, %16) : (tensor<!tf_type.resource>, tensor<1x3x1x1xf32>) -> ()
  func.return %10 : tensor<1x3x1x1xf32>

// WHOLE-PASSES:  %[[vh:.*]] = "tfl.var_handle"() <{container = "", shared_name = "conv_variable/state"}> : () -> tensor<*x!tf_type.resource<tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>>>
// WHOLE-PASSES-NEXT:  %[[rv:.*]] = "tfl.read_variable"(%[[vh]]) : (tensor<*x!tf_type.resource<tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>
// WHOLE-PASSES-DAG:  %[[cv:.*]] = "tfl.conv_2d"(%arg0, {{.*}}) <{{{.*}}}> : (tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>, tensor<1x3x1x1x!quant.uniform<i8<-127:127>:f32:0, {{.*}}>>, tensor<1x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  %[[cc:.*]] = "tfl.concatenation"(%[[rv]], %[[cv]]) <{{{.*}}}> : (tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>, tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x6x1x1x!quant.uniform<i8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  %[[ss:.*]] = "tfl.strided_slice"(%[[cc]], {{.*}}) <{{{.*}}}> : (tensor<1x6x1x1x!quant.uniform<i8:f32, {{.*}}>>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  "tfl.assign_variable"(%[[vh]], %[[ss]]) : (tensor<*x!tf_type.resource<tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>>>, tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>) -> ()
// WHOLE-PASSES-NEXT:  return %[[cv]] : tensor<1x3x1x1x!quant.uniform<i8:f32, {{.*}}>>
}

// WHOLE-PASSES-LABEL: QuantizeTwoVariable
func.func @QuantizeTwoVariable(%arg0: tensor<1x2x3xf32>) -> (tensor<1x2x3xf32>) {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>

  %1 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tfl.pseudo_const"() {value = dense<[0, 2, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tfl.pseudo_const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>

  %4 = "tfl.var_handle"() {container = "", shared_name = "read_assign/states0"} : () -> tensor<!tf_type.resource>
  %5 = "tfl.var_handle"() {container = "", shared_name = "read_assign/states1"} : () -> tensor<!tf_type.resource>
  
  %40 = "tfl.read_variable"(%4) : (tensor<!tf_type.resource>) -> tensor<1x2x3xf32>
  %41 = "quantfork.stats"(%40) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  %42 = "tfl.concatenation"(%41, %0) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x4x3xf32>
  %43 = "quantfork.stats"(%42) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x4x3xf32>) -> tensor<1x4x3xf32>
  %44 = "tfl.strided_slice"(%43, %1, %2, %3) {begin_mask = 7 : i32, ellipsis_mask = 0 : i32, end_mask = 5 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<1x4x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x3xf32>
  %45 = "quantfork.stats"(%44) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  "tfl.assign_variable"(%4, %45) : (tensor<!tf_type.resource>, tensor<1x2x3xf32>) -> ()

  %50 = "tfl.read_variable"(%5) : (tensor<!tf_type.resource>) -> tensor<1x2x3xf32>
  %51 = "quantfork.stats"(%50) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  %52 = "tfl.concatenation"(%51, %0) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x4x3xf32>
  %53 = "quantfork.stats"(%52) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x4x3xf32>) -> tensor<1x4x3xf32>
  %54 = "tfl.strided_slice"(%53, %1, %2, %3) {begin_mask = 7 : i32, ellipsis_mask = 0 : i32, end_mask = 5 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<1x4x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x3xf32>
  %55 = "quantfork.stats"(%54) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  "tfl.assign_variable"(%5, %55) : (tensor<!tf_type.resource>, tensor<1x2x3xf32>) -> ()

  func.return %0 : tensor<1x2x3xf32>

// WHOLE-PASSES:  %[[q1:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>}> : (tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>) -> tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-DAG:  %[[vh1:.*]] = "tfl.var_handle"() <{container = "", shared_name = "read_assign/states0"}> : () -> tensor<*x!tf_type.resource<tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>>>
// WHOLE-PASSES-DAG:  %[[vh2:.*]] = "tfl.var_handle"() <{container = "", shared_name = "read_assign/states1"}> : () -> tensor<*x!tf_type.resource<tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>>>

// WHOLE-PASSES-DAG:  %[[rv1:.*]] = "tfl.read_variable"({{.*}}) : (tensor<*x!tf_type.resource<tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>>>) -> tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  %[[cc1:.*]] = "tfl.concatenation"(%[[rv1]], {{.*}}) {{.*}} : (tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>, tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>) -> tensor<1x4x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  %[[q2:.*]] = "tfl.quantize"(%[[cc1]]) <{qtype = tensor<1x4x3x!quant.uniform<u8:f32, {{.*}}>>}> : (tensor<1x4x3x!quant.uniform<u8:f32, {{.*}}>>) -> tensor<1x4x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  %[[ss1:.*]] = "tfl.strided_slice"(%[[q2]], {{.*}}) <{{{.*}}}> : (tensor<1x4x3x!quant.uniform<u8:f32, {{.*}}>>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  "tfl.assign_variable"(%[[vh1]], %[[ss1]]) : (tensor<*x!tf_type.resource<tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>>>, tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>) -> ()

// WHOLE-PASSES-DAG:  %[[rv2:.*]] = "tfl.read_variable"({{.*}}) : (tensor<*x!tf_type.resource<tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>>>) -> tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  %[[cc2:.*]] = "tfl.concatenation"(%[[rv2]], {{.*}}) {{.*}} : (tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>, tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>) -> tensor<1x4x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  %[[ss2:.*]] = "tfl.strided_slice"(%[[cc2]], {{.*}}) <{{{.*}}}> : (tensor<1x4x3x!quant.uniform<u8:f32, {{.*}}>>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>
// WHOLE-PASSES-NEXT:  "tfl.assign_variable"(%[[vh2]], %[[ss2]]) : (tensor<*x!tf_type.resource<tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>>>, tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>) -> ()

// WHOLE-PASSES-NEXT:  return %arg0 : tensor<1x2x3x!quant.uniform<u8:f32, {{.*}}>>
}
