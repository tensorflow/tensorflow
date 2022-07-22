// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_drop_dtype
// CHECK: -> (tensor<i32>)
tfg.func @test_drop_dtype(%arg0: tensor<i32>) -> (tensor<i32> {tfg.dtype = i32}) {
  return(%arg0) : tensor<i32>
}

