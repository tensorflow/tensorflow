// RUN: xla-opt -split-input-file "-xla-hlo-to-lhlo-with-xla=platform=CUDA" %s | FileCheck %s

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<3x3xi32>
// CHECK-SAME: %[[ARG1:.*]]: memref<2xi32>
// CHECK-SAME: %[[ARG2:.*]]: memref<2x3xi32>
// CHECK-SAME: %[[ARG3:.*]]: memref<36xi8> {
// CHECK: %[[VIEW0:.*]] = memref.view %[[ARG3]]{{.*}} : memref<36xi8> to memref<3x3xi32>
// CHECK: "lmhlo.copy"(%[[ARG0]], %[[VIEW0]])
// CHECK: %[[VIEW1:.*]] = memref.view %[[ARG3]]{{.*}} : memref<36xi8> to memref<3x3xi32>
// CHECK:  "lmhlo.scatter"(%[[VIEW0]], %[[ARG1]], %[[ARG2]], %[[VIEW1]])
// CHECK:  mhlo.add
// CHECK: indices_are_sorted = false
// CHECK: index_vector_dim = 1 : i64
// CHECK: inserted_window_dims = dense<0> : tensor<1xi64>
// CHECK: scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>
// CHECK: update_window_dims = dense<1> : tensor<1xi64>
// CHECK: unique_indices = false
// CHECK: (memref<3x3xi32>, memref<2xi32>, memref<2x3xi32>, memref<3x3xi32>) -> ()
func @main(%operand:tensor<3x3xi32>, %indices: tensor<2xi32>, %updates: tensor<2x3xi32>) -> tensor<3x3xi32> {
  %result = "mhlo.scatter"(%operand, %indices, %updates) ( {
    ^bb0(%x: tensor<i32>, %y : tensor<i32>):
      %result = "mhlo.add"(%x, %y): (tensor<i32>, tensor<i32>) -> tensor<i32>
      "mhlo.return"(%result) : (tensor<i32>) -> ()
    }) { scatter_dimension_numbers = {index_vector_dim = 1 : i64,
                inserted_window_dims = dense<0> : tensor<1xi64>,
                scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
                update_window_dims = dense<1> : tensor<1xi64>},
         indices_are_sorted = false,
         unique_indices = false} : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  return %result : tensor<3x3xi32>
}

