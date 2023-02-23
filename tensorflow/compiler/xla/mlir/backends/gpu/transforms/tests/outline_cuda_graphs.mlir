// RUN: xla-gpu-opt %s --split-input-file -xla-gpu-outline-cuda-graphs \
// RUN:   | FileCheck %s

module attributes {gpu.container_module} {

gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @fn0(%arg0: memref<?xf32>) kernel {
    gpu.return
  }
  gpu.func @fn1(%arg0: memref<?xf32>) kernel {
    gpu.return
  }
}

// CHECK: @func(
// CHECK:   %[[ARG0:.*]]: memref<?xf32>,
// CHECK:   %[[ARG1:.*]]: memref<?xf32>
// CHECK: )
func.func @func(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  // CHECK: call @xla.gpu.cuda.graph.launch(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: {capture = @xla.gpu.cuda.graph.capture}
  // CHECK-NEXT: return

  gpu.launch_func  @gpu_module::@fn0
    blocks in (%c1, %c2, %c3)
    threads in (%c4, %c5, %c6)
    args(%arg0 : memref<?xf32>)

  gpu.launch_func  @gpu_module::@fn1
    blocks in (%c3, %c2, %c1)
    threads in (%c6, %c5, %c4)
    args(%arg1 : memref<?xf32>)

  func.return
}

// CHECK: func @xla.gpu.cuda.graph.capture
// CHECK-NEXT:  %[[C1:.*]] = arith.constant 1
// CHECK-NEXT:  %[[C2:.*]] = arith.constant 2
// CHECK-NEXT:  %[[C3:.*]] = arith.constant 3
// CHECK-NEXT:  %[[C4:.*]] = arith.constant 4
// CHECK-NEXT:  %[[C5:.*]] = arith.constant 5
// CHECK-NEXT:  %[[C6:.*]] = arith.constant 6
// CHECK-NEXT:  gpu.launch_func @gpu_module::@fn0
// CHECK-SAME:    blocks in (%[[C1]], %[[C2]], %[[C3]])
// CHECK-SAME:    threads in (%[[C4]], %[[C5]], %[[C6]])
// CHECK-NEXT:  gpu.launch_func @gpu_module::@fn1
// CHECK-SAME:    blocks in (%[[C3]], %[[C2]], %[[C1]])
// CHECK-SAME:    threads in (%[[C6]], %[[C5]], %[[C4]])
// CHECK-NEXT:  return

// CHECK: func private @xla.gpu.cuda.graph.launch(memref<?xf32>, memref<?xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.cuda.graph.launch"}
}

// -----
// Check that single function launch was not outlined into graph capture.

module attributes {gpu.container_module} {

gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @fn0(%arg0: memref<?xf32>) kernel {
    gpu.return
  }
}

// CHECK: @func(%[[ARG0:.*]]: memref<?xf32>)
func.func @func(%arg0: memref<?xf32>) {
  %c1 = arith.constant 1 : index

  // CHECK: gpu.launch_func {{.*}} args(%[[ARG0]] : memref<?xf32>)
  // CHECK-NOT: call @xla.gpu.cuda.graph.launch
  gpu.launch_func  @gpu_module::@fn0
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<?xf32>)

  func.return
}

}

// -----
// Check that two different sequences are outlined in different capture
// functions.

module attributes {gpu.container_module} {

gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @fn0(%arg0: memref<?xf32>) kernel {
    gpu.return
  }
  gpu.func @fn1(%arg0: memref<?xf32>) kernel {
    gpu.return
  }
}

// CHECK: @func(%[[ARG0:.*]]: memref<?xf32>)
func.func @func(%arg0: memref<?xf32>) {
  // CHECK: %[[C1:.*]] = arith.constant 1
  %c1 = arith.constant 1 : index

  // CHECK: call @xla.gpu.cuda.graph.launch(%[[ARG0]])
  // CHECK-SAME: {capture = @[[CAPTURE:.*]]}

  gpu.launch_func  @gpu_module::@fn0
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<?xf32>)

  gpu.launch_func  @gpu_module::@fn1
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<?xf32>)

  // CHECK: %[[C2:.*]] = arith.constant 2
  %c2 = arith.constant 2 : index

  // Use function call to break the captured ops sequence.
  // CHECK: call @external
  call @external(): () -> ()

  // CHECK: call @xla.gpu.cuda.graph.launch(%[[ARG0]])
  // CHECK-SAME: {capture = @[[CAPTURE_0:.*]]}

  gpu.launch_func  @gpu_module::@fn1
    blocks in (%c2, %c2, %c2)
    threads in (%c2, %c2, %c2)
    args(%arg0 : memref<?xf32>)

  gpu.launch_func  @gpu_module::@fn0
    blocks in (%c2, %c2, %c2)
    threads in (%c2, %c2, %c2)
    args(%arg0 : memref<?xf32>)

  func.return
}

func.func private @external()

// CHECK: rt.export @[[CAPTURE]]
// CHECK: func.func @[[CAPTURE]](%arg0: memref<?xf32>)
// CHECK-NEXT: arith.constant 1
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn0
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn1

// CHECK: rt.export @[[CAPTURE_0]]
// CHECK: func.func @[[CAPTURE_0]](%arg0: memref<?xf32>)
// CHECK-NEXT: arith.constant 2
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn1
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn0

}

// -----
// Check that constants from the different basic blocks are cloned into the
// graph capture function.

module attributes {gpu.container_module} {

gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @fn0(%arg0: memref<?xf32>) kernel {
    gpu.return
  }
  gpu.func @fn1(%arg0: memref<?xf32>) kernel {
    gpu.return
  }
}

// CHECK: @func(
// CHECK:   %[[ARG0:.*]]: memref<?xf32>,
// CHECK:   %[[ARG1:.*]]: memref<?xf32>
// CHECK: )
func.func @func(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  cf.br ^bb2
^bb1:
  // CHECK: call @xla.gpu.cuda.graph.launch(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME: {capture = @xla.gpu.cuda.graph.capture}
  // CHECK-NEXT: return

  gpu.launch_func  @gpu_module::@fn0
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<?xf32>)

  gpu.launch_func  @gpu_module::@fn1
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg1 : memref<?xf32>)

  func.return

^bb2:
  %c1 = arith.constant 1 : index
  cf.br ^bb1
}
}

// CHECK: func @xla.gpu.cuda.graph.capture
// CHECK-NEXT: arith.constant 1
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn0
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn1
// CHECK-NEXT: return

// -----
// Check that memref.view operations are cloned into the graph capture function.

module attributes {gpu.container_module} {

gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @fn0(%arg0: memref<4xf32>) kernel { gpu.return }
  gpu.func @fn1(%arg0: memref<4xf32>) kernel { gpu.return }
}

// CHECK: @func(%[[ARG0:.*]]: memref<16xi8>)
func.func @func(%arg0: memref<16xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %view = memref.view %arg0[%c0][] : memref<16xi8> to memref<4xf32>

  call @external() : () -> ()

  // CHECK: call @xla.gpu.cuda.graph.launch(%[[ARG0]])
  // CHECK-SAME: {capture = @xla.gpu.cuda.graph.capture}
  // CHECK-NEXT: return
  gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1) args(%view : memref<4xf32>)
  gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1) args(%view : memref<4xf32>)

  func.return
}

func.func private @external()
}

// CHECK: func @xla.gpu.cuda.graph.capture
// CHECK-NEXT: arith.constant 0
// CHECK-NEXT: arith.constant 1
// CHECK-NEXT: memref.view
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn0
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn1
// CHECK-NEXT: return

// -----
// Check that memref.view not used by operations in the captured graph will not
// be moved into the graph capture function.

module attributes {gpu.container_module} {

gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @fn0(%arg0: memref<16xi8>) kernel { gpu.return }
  gpu.func @fn1(%arg0: memref<16xi8>) kernel { gpu.return }
}

// CHECK: @func(%[[ARG0:.*]]: memref<16xi8>)
func.func @func(%arg0: memref<16xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  call @external() : () -> ()

  // CHECK: call @xla.gpu.cuda.graph.launch(%[[ARG0]])
  // CHECK-SAME: {capture = @xla.gpu.cuda.graph.capture}
  // CHECK-NEXT: memref.view
  // CHECK-NEXT: return
  gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1) args(%arg0 : memref<16xi8>)
  %view = memref.view %arg0[%c0][] : memref<16xi8> to memref<4xf32>
  gpu.launch_func  @gpu_module::@fn1 blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1) args(%arg0 : memref<16xi8>)

  func.return
}

func.func private @external()
}

// CHECK: func @xla.gpu.cuda.graph.capture
// CHECK-NEXT: arith.constant 1
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn0
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn1
// CHECK-NEXT: return

// -----
// Check that lmhlo_gpu.gemm is moved into the graph capture function.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<16xi8>) kernel { gpu.return }
  }

  // CHECK: @func(%[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0 : index}
  // CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1 : index}
  // CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
  func.func @func(%raw_arg0: memref<16xi8> {lmhlo.params = 0 : index},
                   %raw_arg1: memref<16xi8> {lmhlo.params = 1 : index},
                   %raw_arg2: memref<16xi8> {lmhlo.output_index = dense<[0]> : tensor<1xindex>}) attributes {
                       result_xla_shape = "(f32[4]) "
                   } {
    %c0 = arith.constant 0 : index
    %arg0 = memref.view %raw_arg0[%c0][] : memref<16xi8> to memref<2x2xf32>
    %c1 = arith.constant 0 : index
    %arg1 = memref.view %raw_arg1[%c1][] : memref<16xi8> to memref<2x2xf32>
    %c2 = arith.constant 0 : index
    %arg2 = memref.view %raw_arg2[%c2][] : memref<16xi8> to memref<2x2xf32>

    // CHECK: call @xla.gpu.cuda.graph.launch(%[[ARG0]], %[[ARG1]], %[[ARG2]])
    // CHECK-SAME: {capture = @xla.gpu.cuda.graph.capture}
    "lmhlo_gpu.gemm"(%arg0, %arg1, %arg2) {alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 1 : i64, lhs_stride = 4 : i64, rhs_stride = 4 : i64, dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%raw_arg0 : memref<16xi8>)
    "lmhlo.terminator"() : () -> ()
  }

  func.func private @external()
}

// CHECK: func @xla.gpu.cuda.graph.capture
// CHECK-NEXT: arith.constant 0
// CHECK-NEXT: memref.view
// CHECK-NEXT: arith.constant 0
// CHECK-NEXT: memref.view
// CHECK-NEXT: arith.constant 0
// CHECK-NEXT: memref.view
// CHECK-NEXT: "lmhlo_gpu.gemm"
// CHECK-NEXT: gpu.launch_func @gpu_module::@fn0
// CHECK-NEXT: return

// -----
// Check that lmhlo_gpu.gemm with runtime autotuning is not captured by a CUDA
// graph.

module attributes {gpu.container_module} {

  gpu.module @gpu_module attributes {binary = "kernel binary"} {
    gpu.func @fn0(%arg0: memref<16xi8>) kernel { gpu.return }
  }

  // CHECK: @func(%[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0 : index}
  // CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1 : index}
  // CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
  func.func @func(%raw_arg0: memref<16xi8> {lmhlo.params = 0 : index},
                   %raw_arg1: memref<16xi8> {lmhlo.params = 1 : index},
                   %raw_arg2: memref<16xi8> {lmhlo.output_index = dense<[0]> : tensor<1xindex>}) attributes {
                       result_xla_shape = "(f32[4]) "
                   } {
    %c0 = arith.constant 0 : index
    %arg0 = memref.view %raw_arg0[%c0][] : memref<16xi8> to memref<2x2xf32>
    %c1 = arith.constant 0 : index
    %arg1 = memref.view %raw_arg1[%c1][] : memref<16xi8> to memref<2x2xf32>
    %c2 = arith.constant 0 : index
    %arg2 = memref.view %raw_arg2[%c2][] : memref<16xi8> to memref<2x2xf32>


    // CHECK-NOT: call @xla.gpu.cuda.graph.launch
    // CHECK: "lmhlo_gpu.gemm"
    "lmhlo_gpu.gemm"(%arg0, %arg1, %arg2) {algorithm = -5, alpha_imag = 0.000000e+00 : f64, alpha_real = 1.000000e+00 : f64, beta = 0.000000e+00 : f64, batch_size = 1 : i64, lhs_stride = 4 : i64, rhs_stride = 4 : i64, dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
    gpu.launch_func  @gpu_module::@fn0 blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1) args(%raw_arg0 : memref<16xi8>)
    "lmhlo.terminator"() : () -> ()
  }

  func.func private @external()
}
