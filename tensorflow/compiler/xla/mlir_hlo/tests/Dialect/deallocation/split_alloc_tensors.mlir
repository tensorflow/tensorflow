// RUN: mlir-hlo-opt %s -allow-unregistered-dialect -hlo-split-alloc-tensors | FileCheck %s

func.func @split() {
  %alloc_tensor = bufferization.alloc_tensor() : tensor<2xf32>
  %a = "some.op"(%alloc_tensor) : (tensor<2xf32>) -> (tensor<2xf32>)
  %b = "some.op"(%a, %alloc_tensor)
      : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>)
  "some.use"(%b) : (tensor<2xf32>) -> ()
  %c = "some.op"(%alloc_tensor) : (tensor<2xf32>) -> (tensor<2xf32>)
  return
}

// CHECK-LABEL: @split
// CHECK-NEXT: alloc_tensor
// CHECK-NEXT: some.op
// CHECK-NEXT: alloc_tensor
// CHECK-NEXT: some.op
// CHECK-NEXT: some.use
// CHECK-NEXT: alloc_tensor
// CHECK-NEXT: some.op