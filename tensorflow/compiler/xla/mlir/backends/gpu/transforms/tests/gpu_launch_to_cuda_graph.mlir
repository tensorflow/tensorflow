// RUN: xla-gpu-opt %s --split-input-file -xla-gpu-launch-func-to-cuda-graphs \
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
  // CHECK: %[[C1:.*]] = arith.constant 1
  // CHECK: %[[C2:.*]] = arith.constant 2
  // CHECK: %[[C3:.*]] = arith.constant 3
  // CHECK: %[[C4:.*]] = arith.constant 4
  // CHECK: %[[C5:.*]] = arith.constant 5
  // CHECK: %[[C6:.*]] = arith.constant 6
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  // CHECK: call @xla.gpu.cuda.graph.launch(
  // CHECK:  %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]],
  // CHECK:  %[[ARG0]], %[[ARG1]])
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
// CHECK-NEXT:  gpu.launch_func @gpu_module::@fn0
// CHECK-NEXT:  gpu.launch_func @gpu_module::@fn1
// CHECK-NEXT:  return

// CHECK: func private @xla.gpu.cuda.graph.launch(
// CHECK-SAME:  index, index, index, index, index, index,
// CHECK-SAME:  memref<?xf32>, memref<?xf32>)
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

  // CHECK: call @xla.gpu.cuda.graph.launch(%[[C1]], %[[ARG0]])
  // CHECK-SAME: {capture = @[[CAPTURE:.*]]}

  gpu.launch_func  @gpu_module::@fn0
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<?xf32>)

  gpu.launch_func  @gpu_module::@fn1
    blocks in (%c1, %c1, %c1)
    threads in (%c1, %c1, %c1)
    args(%arg0 : memref<?xf32>)

  // Use constant to break the large function launch sequence.
  // CHECK: %[[C2:.*]] = arith.constant 2
  %c2 = arith.constant 2 : index

  // CHECK: call @xla.gpu.cuda.graph.launch(%[[C2]], %[[ARG0]])
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

// CHECK: rt.export @[[CAPTURE]]
// CHECK: func.func @[[CAPTURE]](%arg0: index, %arg1: memref<?xf32>)

// CHECK: rt.export @[[CAPTURE_0]]
// CHECK: func.func @[[CAPTURE_0]](%arg0: index, %arg1: memref<?xf32>)

}
