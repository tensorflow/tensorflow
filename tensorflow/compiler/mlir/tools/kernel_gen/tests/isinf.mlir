// RUN: tf-opt %s --test-tf-lower-tf --xla-legalize-tf | mlir-hlo-opt --transform-unranked-hlo | kernel-gen-opt -allow-unregistered-dialect --canonicalize --shape-to-descriptors --canonicalize --hlo-bufferize --std-bufferize --func-bufferize | FileCheck %s

// Test whether all shape computations required for isinf can be lowered to
// the standard dialect, scf and descriptors.
// CHECK-LABEL: @isinf
func @isinf(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  // CHECK-NOT: shape
  %0 = "tf.IsInf"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}
