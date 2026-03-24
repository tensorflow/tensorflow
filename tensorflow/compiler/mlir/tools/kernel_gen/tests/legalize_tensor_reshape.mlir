// RUN: kernel-gen-opt --split-input-file --legalize-tensor-reshape %s | FileCheck %s

// CHECK-LABEL: func @tanh
func.func @tanh(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {tf_entry} {
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %1 = shape.num_elements %0 : tensor<?xindex> -> index
  %from_elements = tensor.from_elements %1 : tensor<1xindex>
  // CHECK: mhlo.dynamic_reshape
  // CHECK-NOT: tensor.reshape
  %2 = tensor.reshape %arg0(%from_elements) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  %3 = mhlo.tanh %2 : tensor<?xf32>
  %4 = tensor.reshape %3(%0) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}
