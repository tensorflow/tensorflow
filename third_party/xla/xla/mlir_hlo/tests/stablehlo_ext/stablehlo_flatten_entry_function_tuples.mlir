// RUN: mlir-hlo-opt %s -split-input-file -stablehlo-ext-expand-flatten-entry-function-tuples='entry-function=main' -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x224x224x3xf16>, %arg1: tensor<f32>) -> tensor<1x224x224x3xf16> {
  // CHECK: return %arg0 : tensor<1x224x224x3xf16>
  func.return %arg0 : tensor<1x224x224x3xf16>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x1xf32>, %[[ARG1:.*]]: tensor<1x8x8x16xf32>) -> (tensor<1024xf32>, tensor<1xf32>)
func.func @main(%arg0: tensor<1x1xf32>, %arg1: tensor<1x8x8x16xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  // CHECK-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<1x1xf32>) -> tensor<1xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1xf32>
  // CHECK-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  %1 = stablehlo.reshape %arg1 : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  // CHECK-NEXT: %[[TUPLE:.*]] = stablehlo.tuple %[[RESHAPE1]], %[[RESHAPE0]] {name = "tuple.374"} : tuple<tensor<1024xf32>, tensor<1xf32>>
  %2 = stablehlo.tuple %1, %0 {name = "tuple.374"} : tuple<tensor<1024xf32>, tensor<1xf32>>
  // CHECK-NEXT: %[[RES0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<1024xf32>, tensor<1xf32>>) -> tensor<1024xf32>
  // CHECK-NEXT: %[[RES1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<1024xf32>, tensor<1xf32>>) -> tensor<1xf32>
  // CHECK-NEXT: return %[[RES0]], %[[RES1]] : tensor<1024xf32>, tensor<1xf32>
  return %2 : tuple<tensor<1024xf32>, tensor<1xf32>>
}

// -----

// CHECK-LABEL: func @main
// CEHCK-SAME: () -> (tensor<1xf32>, tensor<1xi32>)
func.func @main() -> tuple<tensor<1xf32>, tensor<1xi32>> {
  // CHECK-NEXT: %[[TUPLE:.*]] = "test.dummy"() : () -> tuple<tensor<1xf32>, tensor<1xi32>>
  %0 = "test.dummy"() : () -> tuple<tensor<1xf32>, tensor<1xi32>>
  // CHECK-NEXT: %[[RES0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<1xf32>, tensor<1xi32>>) -> tensor<1xf32>
  // CHECK-NEXT: %[[RES1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<1xf32>, tensor<1xi32>>) -> tensor<1xi32>
  // CHECK-NEXT: return %[[RES0]], %[[RES1]] : tensor<1xf32>, tensor<1xi32>
  func.return %0 : tuple<tensor<1xf32>, tensor<1xi32>>
}

// -----

// CHECK-LABEL: func @main
func.func @main() -> tuple<> {
  // CHECK-NEXT: %[[TUPLE:.*]] = stablehlo.tuple {xla_shape = "()"} : tuple<>
  %0 = "stablehlo.tuple"() {xla_shape = "()"} : () -> tuple<>
  // CHECK-NEXT: return{{$}}
  func.return %0 : tuple<>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<1024xf32>, %[[ARG1:.*]]: tensor<1xf32>) -> (tensor<1024xf32>, tensor<1xf32>)
func.func @main(%arg0: tuple<tensor<1024xf32>, tensor<1xf32>>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  // CHECK-NEXT: %[[TUPLE:.*]] = stablehlo.tuple %[[ARG0]], %[[ARG1]] : tuple<tensor<1024xf32>, tensor<1xf32>>
  // CHECK-NEXT: %[[RES0:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<1024xf32>, tensor<1xf32>>) -> tensor<1024xf32>
  // CHECK-NEXT: %[[RES1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<1024xf32>, tensor<1xf32>>) -> tensor<1xf32>
  // CHECK-NEXT: return %[[RES0]], %[[RES1]] : tensor<1024xf32>, tensor<1xf32>
  func.return %arg0 : tuple<tensor<1024xf32>, tensor<1xf32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG:.*]]: tensor<1xi8>) -> (tensor<1xf32>, tensor<1xi32>)
func.func @main(%arg0: tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>> {
  // CHECK: %[[T0:.*]] = stablehlo.tuple %[[ARG]] : tuple<tensor<1xi8>>
  // CHECK: %[[T1:.*]] = stablehlo.tuple %[[T0]] : tuple<tuple<tensor<1xi8>>>
  // CHECK: %[[T:.*]] = "test.dummy"(%[[T1]]) : (tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
  %0 = "test.dummy"(%arg0) : (tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
  // CHECK: %[[GTE0:.*]] = stablehlo.get_tuple_element %[[T]][0] : (tuple<tuple<tensor<1xf32>>, tensor<1xi32>>) -> tuple<tensor<1xf32>>
  // CHECK: %[[GTE1:.*]] = stablehlo.get_tuple_element %[[T]][1] : (tuple<tuple<tensor<1xf32>>, tensor<1xi32>>) -> tensor<1xi32>
  // CHECK: %[[GTE2:.*]] = stablehlo.get_tuple_element %[[GTE0]][0] : (tuple<tensor<1xf32>>) -> tensor<1xf32>
  // CHECK: return %[[GTE2]], %[[GTE1]] : tensor<1xf32>, tensor<1xi32>
  func.return %0 : tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
}
