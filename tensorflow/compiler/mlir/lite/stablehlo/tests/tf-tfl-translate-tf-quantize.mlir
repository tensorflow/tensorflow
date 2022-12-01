//RUN: tf_tfl_translate --post-training-quantization --enable-stablehlo-conversion --input-mlir --output-mlir %s -o - | FileCheck %s


module {
func.func @tfInplaceUpdate(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> {
  %1 = arith.constant dense<1> : tensor<1xi32>
  %2 = arith.constant dense<2.0> : tensor<1x1x2xf32>
  %3 = "tf.InplaceUpdate"(%arg0, %1, %2) {device = ""}
    : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
  func.return %3 : tensor<2x1x2xf32>
}
}

//CHECK: module {
//CHECK-NEXT:  func.func @main(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> {
//CHECK-DAG:    %0 = mhlo.constant dense<2.000000e+00> : tensor<1x1x2xf32>
//CHECK-DAG:    %1 = mhlo.constant dense<1> : tensor<i32>
//CHECK-DAG:    %2 = mhlo.constant dense<0> : tensor<i32>
//CHECK-NEXT:    %3 = mhlo.dynamic_update_slice %arg0, %0, %1, %2, %2 : (tensor<2x1x2xf32>, tensor<1x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x2xf32>
//CHECK-NEXT:    return %3 : tensor<2x1x2xf32>
//CHECK-NEXT:  }
//CHECK-NEXT:}