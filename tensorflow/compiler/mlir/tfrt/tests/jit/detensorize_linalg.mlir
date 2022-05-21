// RUN: tf-tfrt-opt %s -tf-jitrt-detensorize-linalg | FileCheck %s

#id = affine_map<(d0) -> (d0)>
#empty = affine_map<(d0) -> ()>

// CHECK-LABEL: func @detensorize
func.func @detensorize(%arg : tensor<100xi32>) -> (tensor<100xi1>) attributes {} {
  %c10 = arith.constant 10 : i32
  %tensor = tensor.from_elements %c10 : tensor<i32>
  %init = linalg.init_tensor [100] : tensor<100xi1>
  %result = linalg.generic {
      indexing_maps = [#id, #empty, #id],
      iterator_types = ["parallel"]}
      ins(%arg, %tensor : tensor<100xi32>, tensor<i32>)
      outs(%init : tensor<100xi1>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i1):
      %0 = arith.cmpi slt, %arg0, %arg1 : i32
      linalg.yield %0 : i1
    } -> tensor<100xi1>
  func.return %result : tensor<100xi1>
}
// CHECK: %[[C10:.*]] = arith.constant 10 : i32
// CHECK: linalg.generic {
// CHECK-SAME: indexing_maps = [#{{map[0-9]*}}, #{{[map0-9]*}}, #{{[map0-9]*}}],
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-SAME: ins(%{{.*}}, %[[C10]] : tensor<100xi32>, i32)
