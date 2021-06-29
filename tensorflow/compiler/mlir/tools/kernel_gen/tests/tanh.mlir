// RUN: tf-opt %s --xla-legalize-tf | \
// RUN: mlir-hlo-opt --mhlo-rank-specialization-cluster \
// RUN: --mhlo-rank-specialization-to-scf --hlo-legalize-to-linalg | \
// RUN: kernel-gen-opt -allow-unregistered-dialect --computeop-and-func-bufferize \
// RUN: --canonicalize --shape-to-descriptors --canonicalize --final-bufferize \
// RUN: | FileCheck %s

// Test whether all shape computations required for tanh can be lowered to
// the standard dialect, scf and descriptors. We check for a sparse pattern here,
// as each lowering pattern is already tested and we just care for the
// integration.
// TODO: Expand this pattern once things have stabilized.
// CHECK-LABEL: @tanh
func @tanh(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: scf.for
  // CHECK: alloc
  // CHECK: memref.reshape
  // CHECK: alloc
  // CHECK: linalg.generic
  // CHECK: memref.reshape
  %0 = "tf.Tanh"(%arg0) { } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
