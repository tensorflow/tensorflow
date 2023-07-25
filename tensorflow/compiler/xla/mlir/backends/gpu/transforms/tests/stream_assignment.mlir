// RUN: xla-gpu-opt %s --split-input-file -xla-gpu-stream-assignment \
// RUN:   | FileCheck %s

// -----
// Check that independent kernels are assigned to different streams.
// A   B--->C
// |        ^
// |        |
// +--------+
//
// Stream assignment: A->0 B->1 C->0

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn2(%arg0: memref<3x3xi64>, %arg1: memref<3x3xi64>) kernel { gpu.return }
  }

  // CHECK: func @xla.gpu.cuda.graph.capture
  func.func @xla.gpu.cuda.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 1 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn2
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn2 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>, %view_0 : memref<3x3xi64>)
    return
  }
}

// -----
// Check that the assignment for the following pattern correctly exploits
// parallelism.
// A--->B   C
// |        ^
// |        |
// +--------+
//
// Stream assignment: A->0 B->1 C->0
//

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn2(%arg0: memref<3x3xi64> {lmhlo.written}, %arg1: memref<3x3xi64> {lmhlo.written}) kernel { gpu.return }
  }


  // CHECK: func @xla.gpu.cuda.graph.capture
  func.func @xla.gpu.cuda.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: gpu.launch_func @gpu_module::@fn2
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn2 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>, %view_0 : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 1 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    return
  }
}
