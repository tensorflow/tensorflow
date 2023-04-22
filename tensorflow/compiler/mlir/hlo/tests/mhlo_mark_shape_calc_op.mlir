// RUN: mlir-hlo-opt --mhlo-mark-shape-calc --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func @main
module {
  // Test placement of operation returning tuples when function result is placed on CPU.
  // TODO(disc): the "input_placements" has no impact on this pass, remove from this test?
  func @main(%arg : tensor<i64> {hlo.input_placements = "cpu"}) -> (tensor<i64> {hlo.output_placements = "cpu"}) {
    // CHECK: mhlo.tuple
    // CHECK-SAME: disc.shape_op = [true]
    // CHECK: mhlo.get_tuple_element
    // CHECK-SAME: disc.shape_op = true
    %tuple = "mhlo.tuple"(%arg) : (tensor<i64>) -> tuple<tensor<i64>>
    %element = "mhlo.get_tuple_element"(%tuple) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    return %element : tensor<i64>
  }
}

// -----

// CHECK-LABEL: func @main
module {
  // Test placement of operation returning tuples when function result is NOT placed on CPU.
  func @main(%arg : tensor<i64>) -> (tensor<i64>) {
    // CHECK: mhlo.tuple
    // CHECK-NOT: disc.shape_op = [true]
    // CHECK: mhlo.get_tuple_element
    // CHECK-NOT: disc.shape_op = true
    %tuple = "mhlo.tuple"(%arg) : (tensor<i64>) -> tuple<tensor<i64>>
    %element = "mhlo.get_tuple_element"(%tuple) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    return %element : tensor<i64>
  }
}

// -----

// CHECK-LABEL: func @main
module {
  // Test a chain of operations.
  // TODO(disc): FIX the "DISABLED" check below. This is broken, the algorithm
  // does not propagate through arguments until fixed point right now.
  func @main(%arg : tensor<i64>) -> (tensor<i64> {hlo.output_placements = "cpu"}) {
    // CHECK: mhlo.tuple
    // DISABLED-CHECK-SAME: disc.shape_op = [true]
    // CHECK: mhlo.get_tuple_element
    // DISABLED-CHECK-SAME: disc.shape_op = true
    // CHECK: mhlo.tuple
    // CHECK-SAME: disc.shape_op = [true]
    // CHECK: mhlo.get_tuple_element
    // CHECK-SAME: disc.shape_op = true
    %tuple = "mhlo.tuple"(%arg) : (tensor<i64>) -> tuple<tensor<i64>>
    %element = "mhlo.get_tuple_element"(%tuple) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    %tuple2 = "mhlo.tuple"(%element) : (tensor<i64>) -> tuple<tensor<i64>>
    %element2 = "mhlo.get_tuple_element"(%tuple2) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    return %element2 : tensor<i64>
  }
}

// -----

// CHECK-LABEL: func @main
module  {
  func @main(%arg0: tensor<?x8xf32>) -> (tensor<?x24xf32> {hlo.output_placements = "cpu"}) {
    // Test various conditions: in particular only i32/i64 operations are placed on CPU.
    // CHECK: mhlo.constant
    // CHECK-SAME: disc.shape_op = true
    // CHECK: tensor.from_elements
    // CHECK-NOT: disc.shape_op = true
    // CHECK: mhlo.dynamic_broadcast_in_dim
    // CHECK-NOT: disc.shape_op = true
    // CHECK: tensor.from_elements
    // CHECK-NOT: disc.shape_op = true
    // CHECK: mhlo.dynamic_reshape
    // CHECK-NOT: disc.shape_op = true
    // CHECK: return
    %0 = mhlo.constant dense<[7, 3]> : tensor<2xi32>
    %c0 = constant 0 : index
    %1 = tensor.dim %arg0, %c0 : tensor<?x8xf32>
    %c8 = constant 8 : index
    %c0_0 = constant 0 : index
    %2 = tensor.extract %0[%c0_0] : tensor<2xi32>
    %3 = index_cast %2 : i32 to index
    %c1 = constant 1 : index
    %4 = tensor.extract %0[%c1] : tensor<2xi32>
    %5 = index_cast %4 : i32 to index
    %6 = tensor.from_elements %3, %1, %5, %c8 : tensor<4xindex>
    %7 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %6) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<?x8xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
    %8 = muli %3, %1 : index
    %9 = muli %5, %c8 : index
    %10 = tensor.from_elements %8, %9 : tensor<2xindex>
    %11 = "mhlo.dynamic_reshape"(%7, %10) : (tensor<?x?x?x?xf32>, tensor<2xindex>) -> tensor<?x24xf32>
    return %11 : tensor<?x24xf32>
  }
}

// -----

// CHECK-LABEL: func @main
module {
  func @main(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x6x2xi64>) -> (tensor<?x6x?xi32> {hlo.output_placements = "cpu"}) {
    // CHECK: tensor.from_elements
    // CHECK-NOT: disc.shape_op = true
    // CHECK: "mhlo.dynamic_gather"
    // CHECK-NOT: disc.shape_op = true
    // CHECK-SAME: disc.shape_op = true
    %c1_i64 = constant 1 : i64
    %c1_i64_0 = constant 1 : i64
    %c2 = constant 2 : index
    %0 = tensor.dim %arg0, %c2 : tensor<?x?x?xi32>
    %1 = index_cast %0 : index to i64
    %2 = tensor.from_elements %c1_i64, %c1_i64_0, %1 : tensor<3xi64>
    %3 = "mhlo.dynamic_gather"(%arg0, %arg1, %2) {dimension_numbers = {collapsed_slice_dims = dense<[0, 1]> : tensor<2xi64>, index_vector_dim = 2 : i64, offset_dims = dense<2> : tensor<1xi64>, start_index_map = dense<[0, 1]> : tensor<2xi64>}, indices_are_sorted = false} : (tensor<?x?x?xi32>, tensor<?x6x2xi64>, tensor<3xi64>) -> tensor<?x6x?xi32>
    return %3 : tensor<?x6x?xi32>
  }
}

