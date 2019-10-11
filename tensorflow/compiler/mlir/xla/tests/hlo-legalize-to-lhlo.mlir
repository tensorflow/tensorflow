// RUN: tf-opt -hlo-legalize-to-lhlo %s -o - | FileCheck %s

// CHECK-LABEL: func @fusion
func @fusion(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>, %arg3: memref<2x2xf32>) {
  // CHECK-NEXT:  %[[ADD_RESULT:.*]] = alloc() {temp = true} : memref<2x2xf32>
  %0 = tensor_load %arg1 : memref<2x2xf32>
  %1 = tensor_load %arg2 : memref<2x2xf32>
  %2 = "xla_hlo.add"(%0, %1) {name = "add"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.add"(%{{.*}}, %{{.*}}, %[[ADD_RESULT]])
  %3 = tensor_load %arg0 : memref<2x2xf32>
  %4 = "xla_hlo.mul"(%2, %3) {name = "multiply"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.mul"(%[[ADD_RESULT]], %{{.*}}, %{{.*}})
  tensor_store %4, %arg3 : memref<2x2xf32>
  // CHECK-NEXT:  dealloc %[[ADD_RESULT]] : memref<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
  "xla_lhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @exp
func @exp(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  %0 = tensor_load %arg0 : memref<2x2xf32>
  %1 = "xla_hlo.exp"(%0) {name = "exp"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.exp"(%{{.*}}, %{{.*}})
  tensor_store %1, %arg1 : memref<2x2xf32>
  "xla_lhlo.terminator"() : () -> ()
}
