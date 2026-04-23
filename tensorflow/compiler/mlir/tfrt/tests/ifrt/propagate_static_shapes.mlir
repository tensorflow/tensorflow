// RUN: tf-tfrt-opt -split-input-file -propagate-static-shapes %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @callee(%arg0: tensor<?x?xi32> {tf._static_shape_arg_idx = 1 : i32}, %arg1: tensor<2xi64>) -> tensor<?x?xi32> attributes {tfrt_ifrt_serving.program_id = 123 : i64}
// CHECK: return %arg0
// CHECK-LABEL: func.func @main
// CHECK-NEXT: %[[C0:.*]] = "tf.Const"
// CHECK-NEXT: %[[C1:.*]] = "tf.IfrtCall"(%arg0, %[[C0]]) <{operandSegmentSizes = array<i32: 1, 1>, program_id = 123 : i64, variable_arg_indices = []}> : (tensor<?x?xi32>, tensor<2xi64>) -> tensor<?x?xi32>
// CHECK-NEXT: return %[[C1]]

module {
  func.func @callee(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {tfrt_ifrt_serving.program_id = 123 : i64} {
    func.return %arg0 : tensor<?x?xi32>
  }
  func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %0 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "tf.SetStaticDimensionBounds"(%arg0, %0) : (tensor<?x?xi32>, tensor<2xi64>) -> tensor<?x?xi32>
    %2 = "tf.IfrtCall"(%1) {program_id = 123 : i64, variable_arg_indices = [], operandSegmentSizes = array<i32: 1, 0>} : (tensor<?x?xi32>) -> tensor<?x?xi32>
    func.return %2 : tensor<?x?xi32>
  }
}

// -----
// CHECK-LABEL: func.func @callee(%arg0: tensor<?x?xi32> {tf._static_shape_arg_idx = 3 : i32}, %arg1: tensor<?xi32> {tf._static_shape_arg_idx = 4 : i32}, %arg2: tensor<?x?xf32>, %arg3: tensor<2xi64>, %arg4: tensor<1xi64>) -> (tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>) attributes {tfrt_ifrt_serving.program_id = 456 : i64}
// CHECK: return %arg0, %arg1, %arg2
// CHECK-LABEL: func.func @main
// CHECK-NEXT: %[[C0:.*]] = "tf.Const"
// CHECK-NEXT: %[[C1:.*]] = "tf.Const"
// CHECK-NEXT: %[[R:.*]]:3 = "tf.IfrtCall"(%arg0, %arg1, %arg2, %[[C0]], %[[C1]]) <{operandSegmentSizes = array<i32: 3, 2>, program_id = 456 : i64, variable_arg_indices = []}> : (tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>, tensor<2xi64>, tensor<1xi64>) -> (tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>)
// CHECK-NEXT: return %[[R]]#0, %[[R]]#1, %[[R]]#2

module {
  func.func @callee(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>) attributes {tfrt_ifrt_serving.program_id = 456 : i64} {
    func.return %arg0, %arg1, %arg2 : tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>
  }
  func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>) {
    %c0 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %c1 = "tf.Const"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "tf.SetStaticDimensionBounds"(%arg0, %c0) : (tensor<?x?xi32>, tensor<2xi64>) -> tensor<?x?xi32>
    %1 = "tf.SetStaticDimensionBounds"(%arg1, %c1) : (tensor<?xi32>, tensor<1xi64>) -> tensor<?xi32>
    %2:3 = "tf.IfrtCall"(%0, %1, %arg2) {program_id = 456 : i64, variable_arg_indices = [], operandSegmentSizes = array<i32: 3, 0>} : (tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>) -> (tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>)
    func.return %2#0, %2#1, %2#2 : tensor<?x?xi32>, tensor<?xi32>, tensor<?x?xf32>
  }
}
