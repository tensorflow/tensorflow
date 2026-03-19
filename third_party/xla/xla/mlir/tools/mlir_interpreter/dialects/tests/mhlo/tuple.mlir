// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @tuple() -> tuple<tensor<2xi1>, tensor<i32>> {
  %cst = mhlo.constant dense<[true, false]> : tensor<2xi1>
  %c1 = mhlo.constant dense<1> : tensor<i32>
  %ret = "mhlo.tuple"(%cst, %c1)
    : (tensor<2xi1>, tensor<i32>) -> tuple<tensor<2xi1>, tensor<i32>>
  return %ret : tuple<tensor<2xi1>, tensor<i32>>
}

// CHECK-LABEL: @tuple
// CHECK-NEXT: Results
// CHECK-NEXT: (TensorOrMemref<2xi1>: [true, false], TensorOrMemref<i32>: 1)

func.func @get_tuple_element() -> (tensor<2xi1>, tensor<i32>) {
  %cst = mhlo.constant dense<[true, false]> : tensor<2xi1>
  %c42 = mhlo.constant dense<42> : tensor<i32>
  %tuple = "mhlo.tuple"(%cst, %c42)
    : (tensor<2xi1>, tensor<i32>) -> tuple<tensor<2xi1>, tensor<i32>>
  %r0 = "mhlo.get_tuple_element"(%tuple) {index = 0 : i32}
    : (tuple<tensor<2xi1>, tensor<i32>>) -> tensor<2xi1>
  %r1 = "mhlo.get_tuple_element"(%tuple) {index = 1 : i32}
    : (tuple<tensor<2xi1>, tensor<i32>>) -> tensor<i32>
  return %r0, %r1 : tensor<2xi1>, tensor<i32>
}

// CHECK-LABEL: @get_tuple_element
// CHECK-NEXT: Results
// CHECK-NEXT: [true, false]
// CHECK-NEXT: 42
