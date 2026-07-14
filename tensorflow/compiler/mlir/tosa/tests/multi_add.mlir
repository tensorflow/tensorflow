// RUN: tf-tosa-opt --tfl-to-tosa-pipeline=target-compilation-backend %s | FileCheck %s


// CHECK:      tensor<1x8x8x3xf32> {ml_program.identifier = "a"}
// CHECK-SAME: tensor<1x8x8x3xf32> {ml_program.identifier = "b"}
// CHECK-SAME: tensor<1x8x8x3xf32> {ml_program.identifier = "c"}
// CHECK-SAME: tensor<1x8x8x3xf32> {ml_program.identifier = "d"}
// CHECK-SAME: -> (tensor<1x8x8x3xf32> {ml_program.identifier = "x"}, tensor<1x8x8x3xf32> {ml_program.identifier = "y"})

module attributes {tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x8x8x3xf32>, %arg1: tensor<1x8x8x3xf32>, %arg2: tensor<1x8x8x3xf32>, %arg3: tensor<1x8x8x3xf32>) -> (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) attributes {tf.entry_function = {inputs = "a,b,c,d", outputs = "x,y"}} {
    %0 = tfl.add %arg1, %arg2 {fused_activation_function = "NONE"} : tensor<1x8x8x3xf32>
    %1 = tfl.add %arg0, %0 {fused_activation_function = "NONE"} : tensor<1x8x8x3xf32>
    %2 = tfl.add %arg3, %0 {fused_activation_function = "NONE"} : tensor<1x8x8x3xf32>
    return %1, %2 : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>
  }
}
