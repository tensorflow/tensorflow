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

  // CHECK: func @xla.gpu.graph.capture
  func.func @xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>) {
    // CHECK: call @xla.gpu.concurrent_region.begin()
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
    // CHECK: call @xla.streams.await() {from = 0 : i64, to = [1]}
    // CHECK: gpu.launch_func @gpu_module::@fn2
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn2 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>, %view_0 : memref<3x3xi64>)
    // CHECK: call @xla.gpu.concurrent_region.end()
    // CHECK: return
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


  // CHECK: func @xla.gpu.graph.capture
  func.func @xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>) {
    // CHECK: call @xla.gpu.concurrent_region.begin()
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: gpu.launch_func @gpu_module::@fn2
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn2 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>, %view_0 : memref<3x3xi64>)
    // CHECK: call @xla.streams.await() {from = 1 : i64, to = [0]}
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 1 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    // CHECK: call @xla.gpu.concurrent_region.end()
    // CHECK: return
    return
  }
}

// -----
// Check that stream with multiple dependencies is handled correctly.
// A    B    C-->D
// |    |        ^
// |    |--------|
// +-------------+
//
// Stream assignment: A->0 B->1 C->2 D->0
//

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn2(%arg0: memref<3x3xi64> {lmhlo.written}, %arg1: memref<3x3xi64> {lmhlo.written}, %arg3: memref<3x3xi64>) kernel { gpu.return }
  }


  // CHECK: func @xla.gpu.graph.capture
  func.func @xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<72xi8>) {
    // CHECK: call @xla.gpu.concurrent_region.begin()
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view_0 = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_1 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_2 = memref.view %arg2[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 1 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_1 : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 2 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_2 : memref<3x3xi64>)
    // CHECK: call @xla.streams.await() {from = 0 : i64, to = [1, 2]}
    // CHECK: gpu.launch_func @gpu_module::@fn2
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn2 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>, %view_1 : memref<3x3xi64>, %view_2 : memref<3x3xi64>)
    // CHECK: call @xla.gpu.concurrent_region.end()
    // CHECK: return
    return
  }
}

// -----
// Check that stream synchronization only happens when two streams joins.
// A    B--->C-->D
// |         ^
// |         |
// +---------+
//
// Stream assignment: A->0 B->1 C->0 D->0
//

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn2(%arg0: memref<3x3xi64> {lmhlo.written}, %arg1: memref<3x3xi64>) kernel { gpu.return }
  }


  // CHECK: func @xla.gpu.graph.capture
  func.func @xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>) {
    // CHECK: call @xla.gpu.concurrent_region.begin()
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view_0 = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_1 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    // CHECK: gpu.launch_func @gpu_module::@fn1
    // CHECK-SAME: {stream = 1 : i64}
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_1 : memref<3x3xi64>)
    // CHECK: call @xla.streams.await() {from = 0 : i64, to = [1]}
    // CHECK: gpu.launch_func @gpu_module::@fn2
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn2 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>, %view_1 : memref<3x3xi64>)
    // CHECK-NEXT: gpu.launch_func @gpu_module::@fn2
    // CHECK-SAME: {stream = 0 : i64}
    gpu.launch_func  @gpu_module::@fn2 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>, %view_1 : memref<3x3xi64>)
    // CHECK: call @xla.gpu.concurrent_region.end()
    // CHECK: return
    return
  }
}
