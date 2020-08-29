// RUN: tf-opt %s -allow-unregistered-dialect -split-input-file -tf-rename-private-functions | FileCheck %s

// CHECK-LABEL: @simple
func @simple() {

// CHECK: "my.call"() {func = @[[NEW_FUNC_NAME:.+]]}
  "my.call"() {func = @my_func} : () -> ()
  return
}

// CHECK-NOT: func @my_func()
// CHECK: func @[[NEW_FUNC_NAME]]()
func @my_func() -> () attributes {sym_visibility = "private"}

// -----

// A stress test case to test uniquification logic

// CHECK-LABEL: @test_uniquification
func @test_uniquification() {
// CHECK: "my.call"() {func = @[[NEW_FUNC_NAME_0:.+]]}
  "my.call"() {func = @my_func} : () -> ()
// CHECK: "my.call"() {func = @[[NEW_FUNC_NAME_1:.+]]}
  "my.call"() {func = @my_func0} : () -> ()
  return
}

// CHECK-NOT: func @my_func()
// CHECK-NOT: func @my_func0()

// CHECK: func @[[NEW_FUNC_NAME_0]]()
func @my_func() -> () attributes {sym_visibility = "private"}
// CHECK: func @[[NEW_FUNC_NAME_1]]()
func @my_func0() -> () attributes {sym_visibility = "private"}


// -----

// Test for SymbolRefArrayAttr

// CHECK-LABEL: @test_case_op
func @test_case_op(%arg0: tensor<i32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch_one, @branch_two], is_stateless = false} : (tensor<i32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK: "tf.Case"(%arg0, %arg1) {branches = [@[[NEW_FUNC_NAME_1:.+]], @[[NEW_FUNC_NAME_2:.+]]]
  return %0 : tensor<2xf32>
}
// CHECK-NOT: func @branch_one()
// CHECK-NOT: func @branch_two()

// CHECK: func @[[NEW_FUNC_NAME_1]]
func @branch_one(tensor<2xf32>) -> tensor<2xf32> attributes {sym_visibility = "private"}
// CHECK: func @[[NEW_FUNC_NAME_2]]
func @branch_two(tensor<2xf32>) -> tensor<2xf32> attributes {sym_visibility = "private"}
