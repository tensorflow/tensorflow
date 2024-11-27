// RUN: mlir-hlo-opt --unbufferize %s | FileCheck %s

// CHECK-LABEL: func @unbufferize
// CHECK-SAME: (%arg0: tensor<8xf32>) -> (tensor<8xf32> {my.attr})
func.func @unbufferize(%arg0: memref<8xf32>, %arg1: memref<8xf32> {my.attr}) {
  %0 = bufferization.to_tensor %arg0 : memref<8xf32> to tensor<8xf32>
  bufferization.materialize_in_destination %0 in writable %arg1
      : (tensor<8xf32>, memref<8xf32>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<8xf32>
  return
}

// CHECK-LABEL: func @not_block_arg
func.func @not_block_arg() {
  %0 = memref.alloc() : memref<8xf32>
  // CHECK: bufferization.to_tensor
  %1 = bufferization.to_tensor %0 : memref<8xf32> to tensor<8xf32>
  // CHECK: bufferization.materialize_in_destination
  bufferization.materialize_in_destination %1 in writable %0
      : (tensor<8xf32>, memref<8xf32>) -> ()
  return
}
