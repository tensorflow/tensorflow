// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions-drq="min-num-elements-for-weights=2500000" | FileCheck %s

// CHECK-LABEL: lift_float_matmul
func.func @lift_float_matmul(%arg0: tensor<1x12x12x512xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
  %out_1 = "tf.MatMul"(%arg0, %cst) {
    device = "", transpose_a = false, transpose_b = false
  } : (tensor<1x12x12x512xf32>, tensor<512x512xf32>) -> tensor<*xf32>
  %out_2 = "tf.MatMul"(%arg0, %arg0) {
    device = "", transpose_a = false, transpose_b = true
  } : (tensor<1x12x12x512xf32>, tensor<1x12x12x512xf32>) -> tensor<*xf32>
  func.return %out_1, %out_2 : tensor<*xf32>, tensor<*xf32>

// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
// CHECK: %[[PARTITIONEDCALL:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST]])
// CHECK-NOT: {_tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: {config = "",
// CHECK-SAME: f = @composite_matmul_fn_1}
// CHECK: %[[UNQUANTIZED_OUTPUT:.*]] = "tf.MatMul"(%arg0, %arg0)
// CHECK: }

// CHECK-LABEL: private @composite_matmul_fn_1
}

// -----

// CHECK-LABEL: not_lift_float_conv
func.func @not_lift_float_conv(%arg0: tensor<1x3x4x512xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x512x512xf32>} : () -> tensor<2x3x512x512xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x512xf32>, tensor<2x3x512x512xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>

// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x512x512xf32>} : () -> tensor<2x3x512x512xf32>
// CHECK: %[[PARTITIONEDCALL:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST]])
// CHECK-NOT: {_tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: {config = "",
// CHECK-SAME: f = @composite_conv2d_fn_1}
}
