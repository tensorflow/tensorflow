// RUN: tfg-transforms-opt --tfg-cf-sink %s | FileCheck %s

tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>) {
  %Add, %ctl = Add(%arg0, %arg1) name("add") {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  %Sub, %ctl_0 = Sub(%arg0, %arg1) name("sub") {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: IfRegion
  %IfRegion, %ctl_2 = IfRegion %cond then {
    // CHECK-NEXT: Add
    // CHECK-SAME: name("add_tfg_cf_sunk_if")
    yield(%Add) : tensor<i32>
  // CHECK: else
  } else {
    // CHECK-NEXT: Sub
    // CHECK-SAME: name("sub_tfg_cf_sunk_if")
    yield(%Sub) : tensor<i32>
  } {_mlir_name = "if"} : (tensor<i1>) -> (tensor<i32>)
  return(%IfRegion) : tensor<i32>
}
