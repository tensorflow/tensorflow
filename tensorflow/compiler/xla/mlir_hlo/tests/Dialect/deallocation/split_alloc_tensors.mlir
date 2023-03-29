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

func.func @split_empty_region() {
  %alloc_tensor = bufferization.alloc_tensor() : tensor<2xf32>
  %cond = "test.cond"() : () -> (i1)
  scf.if %cond {
    %a = "some.op"(%alloc_tensor) : (tensor<2xf32>) -> (tensor<2xf32>)
  }
  // No else.
  return
}

// This is a regression test. Just check that this is processed successfully.
// CHECK-LABEL: @split_empty_region
