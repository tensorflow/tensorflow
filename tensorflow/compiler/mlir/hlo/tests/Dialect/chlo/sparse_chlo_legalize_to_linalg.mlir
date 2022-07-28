// RUN: mlir-hlo-opt --sparse-chlo-legalize-to-linalg %s | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL: @asinh_scalar(
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:         %[[RESULT:.*]] = chlo.asinh %[[ARG]] : tensor<f32> -> tensor<f32>
// CHECK:         return %[[RESULT]] : tensor<f32>
func.func @asinh_scalar(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.asinh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// CHECK-LABEL: @asinh_tensor(
// CHECK-SAME: %[[ARG:.*]]: tensor<10x20xf32, #{{.*}}>) ->
// CHECK-SAME:   tensor<10x20xf32, #{{.*}}> {
// CHECK:         %[[OUT:.*]] = bufferization.alloc_tensor() :
// CHECK-SAME:      tensor<10x20xf32, #{{.*}}>
// CHECK:         %[[VAL:.*]] = linalg.generic
// CHECK-SAME:        ins(%[[ARG]] : tensor<10x20xf32,
// CHECK-SAME:        #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ],
// CHECK-SAME:        outs(%[[OUT]]
// CHECK:           sparse_tensor.unary %{{.*}} : f32 to f32
// CHECK:           present = {
// CHECK:             tensor.from_elements
// CHECK:             chlo.asinh
// CHECK:             tensor.extract
// CHECK:             sparse_tensor.yield %{{.*}} : f32
// CHECK:           }
// CHECK:           absent = {
// CHECK:           }
// CHECK:         }
func.func @asinh_tensor(%arg : tensor<10x20xf32, #CSR>)
                        -> tensor<10x20xf32, #CSR> {
  %result = "chlo.asinh"(%arg) : (tensor<10x20xf32, #CSR>)
                                 -> tensor<10x20xf32, #CSR>
  func.return %result : tensor<10x20xf32, #CSR>
}
