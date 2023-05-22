// RUN: xla-gpu-opt %s --split-input-file -xla-gpu-add-concurrent-regions \
// RUN:   | FileCheck %s


// -----
// Check that two consecutive launch_funcs using different buffers is captured
// by a concurrent_region.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64>) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64>) kernel { gpu.return }
  }


  // CHECK: func @xla.gpu.cuda.graph.capture
  func.func @xla.gpu.cuda.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<328xi8>, %arg3: memref<72xi8>, %arg4: memref<72xi8>, %arg5: memref<72xi8>, %arg6: memref<72xi8>, %arg7: memref<72xi8>, %arg8: memref<72xi8>, %arg9: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: call @xla.gpu.concurrent_region.begin()
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: call @xla.gpu.concurrent_region.end()
    // CHECK-NEXT: return
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    return
  }
}

// -----
// Check that two consecutive launch_funcs using the same buffer is not
// captured.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64>) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64>) kernel { gpu.return }
  }


  // CHECK: func @xla.gpu.cuda.graph.capture
  func.func @xla.gpu.cuda.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<328xi8>, %arg3: memref<72xi8>, %arg4: memref<72xi8>, %arg5: memref<72xi8>, %arg6: memref<72xi8>, %arg7: memref<72xi8>, %arg8: memref<72xi8>, %arg9: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: gpu.launch_func
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: return
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    return
  }
}
