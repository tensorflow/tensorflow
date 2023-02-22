// RUN: mlir-hlo-opt --unbufferize %s | FileCheck %s

// CHECK-LABEL: func @unbufferize
// CHECK-SAME: (%arg0: tensor<8xf32>) -> (tensor<8xf32> {my.attr})
func.func @unbufferize(%arg0: memref<8xf32>, %arg1: memref<8xf32> {my.attr}) {
  %0 = bufferization.to_tensor %arg0 : memref<8xf32>
  memref.tensor_store %0, %arg1 : memref<8xf32>
  // CHECK-NEXT: return %arg0 : tensor<8xf32>
  return
}

// CHECK-LABEL: func @not_block_arg
func.func @not_block_arg() {
  %0 = memref.alloc() : memref<8xf32>
  // CHECK: bufferization.to_tensor
  %1 = bufferization.to_tensor %0 : memref<8xf32>
  // CHECK: memref.tensor_store
  memref.tensor_store %1, %0 : memref<8xf32>
  return
}
