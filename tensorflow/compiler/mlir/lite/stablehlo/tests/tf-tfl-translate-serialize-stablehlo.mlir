//RUN: tf_tfl_translate --enable-stablehlo-conversion --input-mlir %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s


module {
func.func @tfInplaceUpdate(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> {
  %1 = arith.constant dense<1> : tensor<1xi32>
  %2 = arith.constant dense<2.0> : tensor<1x1x2xf32>
  %3 = "tf.InplaceUpdate"(%arg0, %1, %2) {device = ""}
    : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
  func.return %3 : tensor<2x1x2xf32>
}
}

//CHECK: module attributes
//CHECK-SAME: keep_stablehlo_constant = "true"
//CHECK-NEXT:  func.func @main(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "vhlo.dynamic_update_slice_v1"}} {
//CHECK-DAG:    %[[c0:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<1x1x2xf32>
//CHECK-DAG:    %[[c1:.+]] = stablehlo.constant dense<1> : tensor<i32>
//CHECK-DAG:    %[[c2:.+]] = stablehlo.constant dense<0> : tensor<i32>
//CHECK-NEXT:   %[[c3:.+]] = stablehlo.dynamic_update_slice %arg0, %[[c0]], %[[c1]], %[[c2]], %[[c2]] : (tensor<2x1x2xf32>, tensor<1x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x2xf32>
//CHECK-NEXT:   return %[[c3]] : tensor<2x1x2xf32>
//CHECK-NEXT:  }