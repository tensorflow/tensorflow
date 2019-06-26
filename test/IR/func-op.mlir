// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @external_func
func @external_func() {
  // CHECK-NEXT: func @external_func(i32, i64)
  func @external_func(i32, i64) -> ()

  // CHECK: func @external_func_with_result() -> (i1, i32)
  func @external_func_with_result() -> (i1, i32)
  return
}

// CHECK-LABEL: func @complex_func
func @complex_func() {
  // CHECK-NEXT: func @test_dimop(%i0: tensor<4x4x?xf32>) -> index {
  func @test_dimop(%i0: tensor<4x4x?xf32>) -> index {
    %0 = dim %i0, 2 : tensor<4x4x?xf32>
    "foo.return"(%0) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @func_attributes
func @func_attributes() {
  // CHECK-NEXT: func @foo()
  // CHECK-NEXT:   attributes {foo = true}
  func @foo() attributes {foo = true}
  return
}


// CHECK-LABEL: func @func_arg_attributes
func @func_arg_attributes() {
  // CHECK-NEXT: func @external_func_arg_attrs(i32, i1 {dialect.attr = 10 : i64}, i32)
  func @external_func_arg_attrs(i32, i1 {dialect.attr = 10 : i64}, i32)

  // CHECK: func @func_arg_attrs(%i0: i1 {dialect.attr = 10 : i64})
  func @func_arg_attrs(%i0: i1 {dialect.attr = 10 : i64}) {
    return
  }

  return
}
