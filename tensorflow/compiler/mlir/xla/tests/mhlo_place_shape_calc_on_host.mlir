// RUN: tf-opt --mhlo-place-shape-calc %s | FileCheck %s

// CHECK: module
module {
  func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x6x2xi64>) -> tensor<?x6x?xf32> attributes {tf.entry_function = {input_placements = "host", inputs = "input0", output_placements = "host", outputs = "output0"}} {
    // CHECK: memref.dim %arg0, %c2 : tensor<?x?x?xf32> 
    // CHECK: index_cast %0 : index to i64 
    // CHECK: tensor.from_elements %c1_i64, %c1_i64_0, %1 {mhlo_place_type = "host"} : tensor<3xi64>
    // CHECK: "mhlo.h2d"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    // CHECK: "mhlo.dynamic_gather"(%3, %arg1, %2) {dimension_numbers = {collapsed_slice_dims = dense<[0, 1]> : tensor<2xi64>, index_vector_dim = 2 : i64, offset_dims = dense<2> : tensor<1xi64>, start_index_map = dense<[0, 1]> : tensor<2xi64>}, indices_are_sorted = false} : (tensor<?x?x?xf32>, tensor<?x6x2xi64>, tensor<3xi64>) -> tensor<?x6x?xf32>
    // CHECK: "mhlo.d2h"(%4) {mhlo_place_type = "host"} : (tensor<?x6x?xf32>) -> tensor<?x6x?xf32>
    // CHECK: return
    %c1_i64 = constant 1 : i64
    %c1_i64_0 = constant 1 : i64
    %c2 = constant 2 : index
    %0 = memref.dim %arg0, %c2 : tensor<?x?x?xf32>
    %1 = index_cast %0 : index to i64
    %2 = tensor.from_elements %c1_i64, %c1_i64_0, %1 : tensor<3xi64>
    %3 = "mhlo.dynamic_gather"(%arg0, %arg1, %2) {dimension_numbers = {collapsed_slice_dims = dense<[0, 1]> : tensor<2xi64>, index_vector_dim = 2 : i64, offset_dims = dense<2> : tensor<1xi64>, start_index_map = dense<[0, 1]> : tensor<2xi64>}, indices_are_sorted = false} : (tensor<?x?x?xf32>, tensor<?x6x2xi64>, tensor<3xi64>) -> tensor<?x6x?xf32>
    return %3 : tensor<?x6x?xf32>
  }
}
