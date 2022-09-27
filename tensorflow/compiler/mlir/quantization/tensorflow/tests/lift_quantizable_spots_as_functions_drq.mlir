// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions-drq | FileCheck %s

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
// CHECK-SAME: {_tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: f = @composite_matmul_fn_1}
// CHECK: %[[UNQUANTIZED_OUTPUT:.*]] = "tf.MatMul"(%arg0, %arg0)
// CHECK: }

// CHECK-LABEL: private @composite_matmul_fn_1
// CHECK-NEXT: %[[OUT:.*]] = "tf.MatMul"(%arg0, %arg1)
// CHECK-NEXT: return %[[OUT]]
}
