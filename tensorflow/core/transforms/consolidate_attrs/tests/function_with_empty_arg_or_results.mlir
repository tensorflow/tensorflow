// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs | FileCheck %s
// RUN: tfg-transforms-opt %s --tfg-prepare-attrs-export | FileCheck %s

// This is used to ensure that the pass is handling empty arg_attrs/res_attrs,
// e.g., no crash happens.

// CHECK-LABEL: @test_no_arg
tfg.func @test_no_arg() -> (tensor<*xi32>) {
  %A, %ctl = A() : () -> (tensor<*xi32>)
  return(%A) : tensor<*xi32>
}

// CHECK-LABEL: @test_without_result
tfg.func @test_without_result(%arg0: tensor<*xi32>) -> () {
  return
}

// CHECK-LABEL: @test_without_arg_nor_result
tfg.func @test_without_arg_nor_result() -> () {
  return
}
