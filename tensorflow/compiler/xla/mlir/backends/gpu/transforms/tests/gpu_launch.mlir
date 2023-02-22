// RUN: xla-gpu-opt %s -xla-gpu-to-gpu-runtime | FileCheck %s

module attributes {gpu.container_module} {

// CHECK-NOT: gpu.module
gpu.module @gpu_module attributes {binary = "kernel binary"} {
  gpu.func @fn0(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) kernel {
    gpu.return
  }
  gpu.func @fn1(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) kernel {
    gpu.return
  }
}

// CHECK: @func(
// CHECK:   %[[ARG0:.*]]: memref<4x4xf32>,
// CHECK:   %[[ARG1:.*]]: memref<4x4xf32>
// CHECK: )
func.func @func(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) {
  // Launch dimensions converted to i32 as a part of the lowering.
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i32
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
  // CHECK-DAG: %[[C6:.*]] = arith.constant 6 : i32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C256:.*]] = arith.constant 256 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c256 = arith.constant 256 : i32

  // CHECK: call @[[LAUNCH:[_a-z.]+]](%[[C0]], %[[C1]], %[[C2]], %[[C3]],
  // CHECK-SAME: %[[C4]], %[[C5]], %[[C6]], %[[ARG0]], %[[ARG1]])
  // CHECK-SAME: kernel = "fn0"
  gpu.launch_func  @gpu_module::@fn0
    blocks in (%c1, %c2, %c3)
    threads in (%c4, %c5, %c6)
    args(%arg0 : memref<4x4xf32>, %arg1 : memref<4x4xf32>)

  // CHECK: call @[[LAUNCH]](%[[C256]], %[[C3]], %[[C2]], %[[C1]], %[[C6]],
  // CHECK-SAME: %[[C5]], %[[C4]], %[[ARG0]], %[[ARG1]])
  // CHECK-DAG: kernel = "fn1"
  gpu.launch_func  @gpu_module::@fn1
    blocks in (%c3, %c2, %c1)
    threads in (%c6, %c5, %c4)
    dynamic_shared_memory_size %c256
    args(%arg0 : memref<4x4xf32>, %arg1 : memref<4x4xf32>)

  func.return
}

// CHECK: func private @[[LAUNCH]](i32, i32, i32, i32, i32, i32,
// CHECK-SAME: memref<4x4xf32>, memref<4x4xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.func.launch"}

// Check that we have a single custom call declaration in the module.
// CHECK-NOT: rt.custom_call

}
