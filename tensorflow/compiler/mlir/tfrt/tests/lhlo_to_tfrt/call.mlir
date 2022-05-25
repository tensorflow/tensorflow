// RUN: lhlo-tfrt-opt %s \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK: func @main(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain {
func.func @main(%arg0: memref<8xf32>) {
  // CHECK: %[[ch:.*]] = tfrt.call @memcpy(%arg0, %arg1, %arg2)
  tfrt.call @memcpy(%arg0) : (memref<8xf32>) -> ()
  // CHECK: tfrt.call @nogpu(%arg2) : (!tfrt_gpu.buffer) -> ()
  tfrt.call @nogpu(%arg0) : (memref<8xf32>) -> ()
  // CHECK: tfrt.call @return(%arg2) : (!tfrt_gpu.buffer) -> !tfrt_gpu.buffer
  %0 = tfrt.call @return(%arg0) : (memref<8xf32>) -> memref<8xf32>
  // CHECK: tfrt.return %[[ch]] : !tfrt.chain
  func.return
}

// CHECK: func @memcpy(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain {
func.func @memcpy(%arg0: memref<8xf32>) {
  // CHECK: %[[ch:.*]] = tfrt_gpu.mem.copy %arg2, %arg2, %arg1, %arg0
  gpu.memcpy %arg0, %arg0 : memref<8xf32>, memref<8xf32>
  // CHECK: tfrt.return %[[ch]] : !tfrt.chain
  func.return
}

// CHECK: func @nogpu(%arg0: !tfrt_gpu.buffer)
func.func @nogpu(%arg0: memref<8xf32>) {
  // Prevents special empty-function rewrite.
  %ch = tfrt.new.chain
  %zero = tfrt.constant.i32 0
  tfrt.print.i32 %zero, %ch
  func.return
}

// CHECK: func @return(%arg0: !tfrt_gpu.buffer) -> !tfrt_gpu.buffer
func.func @return(%arg0: memref<8xf32>) -> memref<8xf32> {
  // CHECK: tfrt.return %arg0 : !tfrt_gpu.buffer
  tfrt.return %arg0 : memref<8xf32>
}


