// RUN: fusion_compiler_opt %s -xtile-cpu-linalg-elementwise-to-vector -split-input-file | FileCheck %s

func.func @elementwise_add_to_vector(
    %lhs : memref<8x1024xf32>,
    %rhs : memref<8x1024xf32>,
    %out : memref<8x1024xf32>) {
  // CHECK: %1 = vector.transfer_read %arg0
  // CHECK: %2 = vector.transfer_read %arg1
  // CHECK: %3 = arith.addf {{.*}} : vector<8x1024xf32>
  // CHECK: vector.transfer_write %{{.*}}, %arg2{{.*}} :
  // CHECK-SAME: vector<8x1024xf32>, memref<8x1024xf32>
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%lhs, %rhs : memref<8x1024xf32>, memref<8x1024xf32>)
    outs(%out : memref<8x1024xf32>)
  return
}
