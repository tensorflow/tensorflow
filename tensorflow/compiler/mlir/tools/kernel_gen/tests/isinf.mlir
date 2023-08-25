// RUN: tf-opt %s --test-tf-lower-tf --xla-legalize-tf | \
// RUN: mlir-hlo-opt --mhlo-rank-specialization-cluster \
// RUN: --mhlo-rank-specialization-to-scf --hlo-legalize-to-linalg \
// RUN: --empty-tensor-to-alloc-tensor \
// RUN: --computeop-and-func-bufferize --canonicalize | \
// RUN: kernel-gen-opt -allow-unregistered-dialect \
// RUN: --shape-to-descriptors \
// RUN: --canonicalize --kernelgen-final-bufferize | \
// RUN: FileCheck %s

// Test whether all shape computations required for isinf can be lowered to
// the standard dialect, scf and descriptors.
// CHECK-LABEL: @isinf
func.func @isinf(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  // CHECK-NOT: shape
  %0 = "tf.IsInf"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}
