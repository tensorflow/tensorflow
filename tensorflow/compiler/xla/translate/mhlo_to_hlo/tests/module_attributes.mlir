// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo %s | FileCheck %s
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo --via-builder=true %s | FileCheck %s

module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 1, indices = [0]> ] } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}
// CHECK-LABEL: hlo_module       {
// CHECK: cross_program_prefetches {
// CHECK-NEXT:    parameter: 1
// CHECK-NEXT:    index: 0
// CHECK-NEXT:  }
