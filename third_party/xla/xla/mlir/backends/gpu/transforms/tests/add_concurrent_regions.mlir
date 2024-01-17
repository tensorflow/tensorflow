// RUN: xla-gpu-opt %s --split-input-file -xla-gpu-add-concurrent-regions \
// RUN:   | FileCheck %s


// -----
// Check that two consecutive launch_funcs using different buffers is captured
// by a concurrent_region.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<328xi8>, %arg3: memref<72xi8>, %arg4: memref<72xi8>, %arg5: memref<72xi8>, %arg6: memref<72xi8>, %arg7: memref<72xi8>, %arg8: memref<72xi8>, %arg9: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: call @local_xla.gpu.concurrent_region.begin()
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: call @local_xla.gpu.concurrent_region.end()
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
    gpu.func @fn0(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<328xi8>, %arg3: memref<72xi8>, %arg4: memref<72xi8>, %arg5: memref<72xi8>, %arg6: memref<72xi8>, %arg7: memref<72xi8>, %arg8: memref<72xi8>, %arg9: memref<72xi8>) {
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

// -----
// Check that there is no dependency from launch_funcs that do not write to
// buffers.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64> ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64> ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<328xi8>, %arg3: memref<72xi8>, %arg4: memref<72xi8>, %arg5: memref<72xi8>, %arg6: memref<72xi8>, %arg7: memref<72xi8>, %arg8: memref<72xi8>, %arg9: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: call @local_xla.gpu.concurrent_region.begin()
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: call @local_xla.gpu.concurrent_region.end()
    // CHECK-NEXT: return
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    return
  }
}

// -----
// Check that the i1 data type is handled correctly.
module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi1> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi1> {lmhlo.written} ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<328xi8>, %arg3: memref<72xi8>, %arg4: memref<72xi8>, %arg5: memref<72xi8>, %arg6: memref<72xi8>, %arg7: memref<72xi8>, %arg8: memref<72xi8>, %arg9: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi1>
    %view_0 = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi1>

    // CHECK-NOT: xla.gpu.concurrent_region.begin()
    // CHECK: gpu.launch_func
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: return
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi1>)
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi1>)
    return
  }
}

// -----
// Check that disjoint buffer slices does not introduce dependency.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<144xi8>) {
    %c0 = arith.constant 0 : index
    %c72 = arith.constant 72 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<144xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg0[%c72][] : memref<144xi8> to memref<3x3xi64>

    // CHECK: call @local_xla.gpu.concurrent_region.begin()
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: call @local_xla.gpu.concurrent_region.end()
    // CHECK-NEXT: return
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    return
  }
}

// -----
// Check that overlapping buffer slices creates dependency.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<144xi8>) {
    %c0 = arith.constant 0 : index
    %c36 = arith.constant 36 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<144xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg0[%c36][] : memref<144xi8> to memref<3x3xi64>

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

// -----
// Check that constant input buffer does not create dependency.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<144xi8> {lmhlo.constant_name = "cst0"}) {
    %c0 = arith.constant 0 : index
    %c36 = arith.constant 36 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<144xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg0[%c36][] : memref<144xi8> to memref<3x3xi64>

    // CHECK: call @local_xla.gpu.concurrent_region.begin()
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: call @local_xla.gpu.concurrent_region.end()
    // CHECK-NEXT: return
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    return
  }
}

// -----
// Check that two gemms that read the same buffer are moved into a concurrent
// region.

module attributes {gpu.container_module} {

  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<16xi8>,
                   %arg1: memref<16xi8>,
                   %arg2: memref<16xi8>,
                   %arg3: memref<16xi8>) {
    %c0 = arith.constant 0 : index
    %view_0 = memref.view %arg0[%c0][] : memref<16xi8> to memref<2x2xf32>
    %c1 = arith.constant 0 : index
    %view_1 = memref.view %arg1[%c1][] : memref<16xi8> to memref<2x2xf32>
    %c2 = arith.constant 0 : index
    %view_2 = memref.view %arg2[%c2][] : memref<16xi8> to memref<2x2xf32>
    %view_3 = memref.view %arg3[%c2][] : memref<16xi8> to memref<2x2xf32>

    // CHECK: call @local_xla.gpu.concurrent_region.begin()
    // CHECK-NEXT: lmhlo_gpu.gemm
    // CHECK-NEXT: lmhlo_gpu.gemm
    // CHECK-NEXT: call @local_xla.gpu.concurrent_region.end()
    // CHECK-NEXT: return
    "lmhlo_gpu.gemm"(%view_0, %view_1, %view_2) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 1 : i64, lhs_stride = 4 : i64, rhs_stride = 4 : i64, dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo_gpu.gemm"(%view_0, %view_1, %view_3) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 1 : i64, lhs_stride = 4 : i64, rhs_stride = 4 : i64, dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    return
  }

  func.func private @external()
}

// -----
// Check that lmhlo_gpu.gemm is not moved into the concurrent region if it
// uses a buffer used by a kernel launch.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<16xi8> {lmhlo.written} ) kernel { gpu.return }
  }

  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<16xi8>,
                   %arg1: memref<16xi8>,
                   %arg2: memref<16xi8>) {
    %c0 = arith.constant 0 : index
    %view_0 = memref.view %arg0[%c0][] : memref<16xi8> to memref<2x2xf32>
    %c1 = arith.constant 0 : index
    %view_1 = memref.view %arg1[%c1][] : memref<16xi8> to memref<2x2xf32>
    %c2 = arith.constant 0 : index
    %view_2 = memref.view %arg2[%c2][] : memref<16xi8> to memref<2x2xf32>

    // CHECK-NOT: @local_xla.gpu.concurrent_region.begin()
    // CHECK: lmhlo_gpu.gemm
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: return
    "lmhlo_gpu.gemm"(%view_0, %view_1, %view_2) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 1 : i64, lhs_stride = 4 : i64, rhs_stride = 4 : i64, dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c0, %c0, %c0)
      threads in (%c0, %c0, %c0) args(%arg0: memref<16xi8>)
    return
  }

  func.func private @external()
}

// -----
// Check that memcpies are added to concurrent regions.

module attributes {gpu.container_module} {

  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<16xi8>,
                   %arg1: memref<16xi8>,
                   %arg2: memref<16xi8>) {
    %c0 = arith.constant 0 : index
    %view_0 = memref.view %arg0[%c0][] : memref<16xi8> to memref<2x2xf32>
    %c1 = arith.constant 0 : index
    %view_1 = memref.view %arg1[%c1][] : memref<16xi8> to memref<2x2xf32>
    %c2 = arith.constant 0 : index
    %view_2 = memref.view %arg2[%c2][] : memref<16xi8> to memref<2x2xf32>

    // CHECK: @local_xla.gpu.concurrent_region.begin()
    // CHECK-NEXT: gpu.memcpy
    // CHECK-NEXT: gpu.memcpy
    // CHECK-NEXT: @local_xla.gpu.concurrent_region.end()
    // CHECK-NEXT: return
    gpu.memcpy %view_1, %view_0 : memref<2x2xf32>, memref<2x2xf32>
    gpu.memcpy %view_2, %view_0 : memref<2x2xf32>, memref<2x2xf32>
    return
  }

  func.func private @external()
}

// -----
// Check that region size is set correctly.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
    gpu.func @fn1(%arg0: memref<3x3xi64> {lmhlo.written} ) kernel { gpu.return }
  }


  // CHECK: func @local_xla.gpu.graph.capture
  func.func @local_xla.gpu.graph.capture(%arg0: memref<72xi8>, %arg1: memref<72xi8>, %arg2: memref<328xi8>, %arg3: memref<72xi8>, %arg4: memref<72xi8>, %arg5: memref<72xi8>, %arg6: memref<72xi8>, %arg7: memref<72xi8>, %arg8: memref<72xi8>, %arg9: memref<72xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][] : memref<72xi8> to memref<3x3xi64>
    %view_0 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>

    // CHECK: call @local_xla.gpu.concurrent_region.begin() {size = 2 : i64}
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: memref.view
    // CHECK-NEXT: gpu.launch_func
    // CHECK-NEXT: call @local_xla.gpu.concurrent_region.end()
    // CHECK-NEXT: return
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view : memref<3x3xi64>)
    %view_1 = memref.view %arg1[%c0][] : memref<72xi8> to memref<3x3xi64>
    gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%view_0 : memref<3x3xi64>)
    return
  }
}
