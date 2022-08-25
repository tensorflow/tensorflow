// RUN: mlir-hlo-opt -hlo-legalize-sort %s | FileCheck %s

func.func @sort(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %result:2 = "mhlo.sort"(%arg0, %arg1) ({
    ^bb0(%00: tensor<i32>, %01: tensor<i32>, %10: tensor<i32>, %11: tensor<i32>):
      %50 = tensor.extract %00[] : tensor<i32>
      %51 = tensor.extract %01[] : tensor<i32>
      %52 = arith.cmpi sgt, %50, %51 : i32
      %cmp_result = tensor.from_elements %52 : tensor<i1>
      "mhlo.return"(%cmp_result) : (tensor<i1>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
  func.return %result#0, %result#1 : tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL: func @sort(
// CHECK-NOT:   mhlo.sort