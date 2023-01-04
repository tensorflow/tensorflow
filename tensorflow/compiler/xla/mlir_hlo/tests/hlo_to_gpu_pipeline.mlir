// RUN: mlir-hlo-opt -split-input-file %s \
// RUN:  -hlo-to-gpu-pipeline="block-tile=1024 warp-tile=128 thread-tile=4" \
// RUN: | FileCheck %s

// CHECK:       gpu.container_module
// CHECK-LABEL: func.func @simple_op
func.func @simple_op(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>) {
  %0 = bufferization.to_tensor %arg0 : memref<2048xf32>
  %1 = mhlo.log %0 : tensor<2048xf32>
  // CHECK-DAG:  %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG:  %[[BLOCK:.*]] = arith.constant 8
  // CHECK-DAG:  %[[WARP:.*]] = arith.constant 32
  // CHECK-DAG:  %[[GRID:.*]] = arith.constant 2
  // CHECK:      gpu.launch_func @[[MODULE:.*]]::@[[KERNEL:.*]] blocks
  // CHECK-SAME: in (%[[GRID]], %[[ONE]], %[[ONE]])
  // CHECK-SAME: threads in (%[[WARP]], %[[BLOCK]], %[[ONE]])
  // CHECK-SAME: args({{.*}} : memref<2048xf32>, {{.*}} : memref<2048xf32>)
  memref.tensor_store %1, %arg1 : memref<2048xf32>
  "lmhlo.terminator"() : () -> ()
}
// CHECK: gpu.module @[[MODULE]] attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}
// CHECK: llvm.func @[[KERNEL]]({{.*}}) attributes {gpu.kernel, nvvm.kernel}
// CHECK: llvm.call @__nv_logf
// Make sure we successfully unrolled the loop 4 times
// CHECK: llvm.call @__nv_logf
// CHECK: llvm.call @__nv_logf
// CHECK: llvm.call @__nv_logf
// CHECK-NOT: llvm.call @__nv_logf


// -----

// CHECK:       gpu.container_module
// CHECK-LABEL: func.func @fusion
func.func @fusion(%arg0: memref<2048xf32>, %arg1: memref<2048xf32>) {
  %0 = bufferization.to_tensor %arg0 : memref<2048xf32>
  %1 = mhlo.abs %0 : tensor<2048xf32>
  %2 = mhlo.add %1, %1 : tensor<2048xf32>
  // CHECK-DAG:  %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG:  %[[BLOCK:.*]] = arith.constant 8
  // CHECK-DAG:  %[[WARP:.*]] = arith.constant 32
  // CHECK-DAG:  %[[GRID:.*]] = arith.constant 2
  // CHECK:      gpu.launch_func @[[MODULE:.*]]::@[[KERNEL:.*]] blocks
  // CHECK-SAME: in (%[[GRID]], %[[ONE]], %[[ONE]])
  // CHECK-SAME: threads in (%[[WARP]], %[[BLOCK]], %[[ONE]])
  // CHECK-SAME: args({{.*}} : memref<2048xf32>, {{.*}} : memref<2048xf32>)
  memref.tensor_store %2, %arg1 : memref<2048xf32>
  "lmhlo.terminator"() : () -> ()
}
// CHECK:     gpu.module @[[MODULE]]
// CHECK:     llvm.func @[[KERNEL]]({{.*}}) attributes {gpu.kernel, nvvm.kernel}
// CHECK:     %[[ABS:.*]] = llvm.call @__nv_fabsf
// CHECK-NOT: llvm.return
// CHECK:     %[[ADD:.*]] = llvm.fadd %[[ABS]], %[[ABS]]
// Make sure we successfully unrolled the loop 4 times:
// CHECK: llvm.call @__nv_fabsf
// CHECK: llvm.call @__nv_fabsf
// CHECK: llvm.call @__nv_fabsf
// CHECK-NOT: llvm.call @__nv_fabsf


// -----

// CHECK:       gpu.container_module
// CHECK-LABEL: func.func @imperfect_tiling
func.func @imperfect_tiling(%arg0: memref<2051xf32>, %arg1: memref<2051xf32>) {
  %0 = bufferization.to_tensor %arg0 : memref<2051xf32>
  %1 = mhlo.log %0 : tensor<2051xf32>
  // CHECK-DAG:  %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG:  %[[BLOCK:.*]] = arith.constant 8
  // CHECK-DAG:  %[[WARP:.*]] = arith.constant 32
  // CHECK-DAG:  %[[GRID:.*]] = arith.constant 3
  // CHECK:      gpu.launch_func @[[MODULE:.*]]::@[[KERNEL:.*]] blocks
  // CHECK-SAME: in (%[[GRID]], %[[ONE]], %[[ONE]])
  // CHECK-SAME: threads in (%[[WARP]], %[[BLOCK]], %[[ONE]])
  // CHECK-SAME: args({{.*}} : memref<2051xf32>, {{.*}} : memref<2051xf32>)
  memref.tensor_store %1, %arg1 : memref<2051xf32>
  "lmhlo.terminator"() : () -> ()
}
// CHECK: gpu.module @[[MODULE]] attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>}
// CHECK: llvm.func @[[KERNEL]]({{.*}}) attributes {gpu.kernel, nvvm.kernel}
// CHECK: llvm.call @__nv_logf
// Make sure we successfully unrolled the loop 4 times:
// CHECK: llvm.call @__nv_logf
// CHECK: llvm.call @__nv_logf
// CHECK: llvm.call @__nv_logf
// ... and that we have an imperfect-tile loop at the end:
// CHECK: llvm.cond_br
// CHECK: llvm.call @__nv_logf
// CHECK-NOT: llvm.call @__nv_logf
