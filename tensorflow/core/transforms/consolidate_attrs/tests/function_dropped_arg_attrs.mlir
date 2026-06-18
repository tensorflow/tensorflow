// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_drop_dtype(
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>)
tfg.func @test_drop_dtype(%arg0: tensor<i32> {tfg.dtype = i32}) -> (tensor<i32>) {
  return(%arg0) : tensor<i32>
}

// -----

// CHECK-LABEL: tfg.func @test_drop_is_ref(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*x!tf_type.int32ref>)
tfg.func @test_drop_is_ref(%arg0: tensor<*x!tf_type.int32ref> {tfg.is_ref}) -> (tensor<*xi32>) {
  %DeRef, %ctl = DeRef(%arg0) : (tensor<*x!tf_type.int32ref>) -> (tensor<*xi32>)
  return(%DeRef) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_skip_ctl
// CHECK-SAME: tfg.name = "a"
// CHECK-NEXT: tfg.name = "b"
// CHECK-NEXT: tfg.name = "c"
tfg.func @test_skip_ctl(%a: tensor<*xi32> {tfg.name = "a"},
                        %b: tensor<*xi32> {tfg.name = "b"})
    -> (tensor<*xi32> {tfg.name = "c"}) {
  return(%a) : tensor<*xi32>
}
