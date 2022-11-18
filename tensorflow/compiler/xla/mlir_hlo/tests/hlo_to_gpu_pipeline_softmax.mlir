// TODO(b/255935104): Merge this with hlo_to_gpu_pipeline.mlir once we've
// unified the softmax and elementwise paths.
// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:   --hlo-to-gpu-pipeline="block-tile=8 warp-tile=1 experimental-softmax=true" \
// RUN: | FileCheck %s

// CHECK:       gpu.container_module
// CHECK-LABEL: @perfectly_tiled_softmax(
func.func @perfectly_tiled_softmax(%argbuffer : memref<2048x4096xf32>,
    %resbuffer : memref<2048x4096xf32>) {
  %arg = bufferization.to_tensor %argbuffer : memref<2048x4096xf32>
  %0 = mhlo.constant dense<-1> : tensor<1xi64>
  %1 = mhlo.convert %arg : tensor<2048x4096xf32>
  %2 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %3 = mhlo.reduce(%1 init: %2) applies mhlo.maximum across dimensions = [1]
      : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
  %4 = mhlo.convert %3 : tensor<2048xf32>
  %cst = arith.constant dense<1> : tensor<1xi32>
  %5 = mhlo.reshape %4 : (tensor<2048xf32>) -> tensor<2048x1xf32>
  %6 = chlo.broadcast_subtract %arg, %5
      : (tensor<2048x4096xf32>, tensor<2048x1xf32>) -> tensor<2048x4096xf32>
  %7 = mhlo.exponential %6 : tensor<2048x4096xf32>
  %8 = mhlo.convert %7 : tensor<2048x4096xf32>
  %9 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  %10 = mhlo.reduce(%8 init: %9) applies mhlo.add across dimensions = [1]
      : (tensor<2048x4096xf32>, tensor<f32>) -> tensor<2048xf32>
  %11 = mhlo.convert %10 : tensor<2048xf32>
  %cst_0 = arith.constant dense<1> : tensor<1xi32>
  %12 = mhlo.reshape %11 : (tensor<2048xf32>) -> tensor<2048x1xf32>
  %13 = chlo.broadcast_divide %7, %12
      : (tensor<2048x4096xf32>, tensor<2048x1xf32>) -> tensor<2048x4096xf32>
  // CHECK-DAG:  %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG:  %[[BLOCK:.*]] = arith.constant 8
  // CHECK-DAG:  %[[WARP:.*]] = arith.constant 32
  // CHECK-DAG:  %[[GRID:.*]] = arith.constant 256
  // CHECK:      gpu.launch_func @[[MODULE:.*]]::@[[KERNEL:.*]] blocks
  // CHECK-SAME: in (%[[GRID]], %[[ONE]], %[[ONE]])
  // CHECK-SAME: threads in (%[[WARP]], %[[BLOCK]], %[[ONE]])
  // CHECK-SAME: args({{.*}} : memref<2048x4096xf32>,
  // CHECK-SAME: {{.*}} : memref<2048x4096xf32>)
  memref.tensor_store %13, %resbuffer : memref<2048x4096xf32>
  "lmhlo.terminator"() : () -> ()
}
// CHECK: gpu.module @[[MODULE]]
// CHECK: llvm.func @[[KERNEL]]({{.*}}) attributes {gpu.kernel, nvvm.kernel}
// CHECK: nvvm.shfl.sync  bfly
// CHECK: llvm.fcmp
// CHECK: llvm.select
// CHECK: llvm.fsub
// CHECK: llvm.call @__nv_expf
// CHECK: nvvm.shfl.sync  bfly
// CHECK: llvm.fadd
// CHECK: llvm.fdiv

// -----

// CHECK:       gpu.container_module
// CHECK-LABEL: @imperfectly_tiled_softmax(
func.func @imperfectly_tiled_softmax(%argbuffer : memref<2047x4095xf32>,
    %resbuffer : memref<2047x4095xf32>) {
  %arg = bufferization.to_tensor %argbuffer : memref<2047x4095xf32>
  %0 = mhlo.constant dense<-1> : tensor<1xi64>
  %1 = mhlo.convert %arg : tensor<2047x4095xf32>
  %2 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %3 = mhlo.reduce(%1 init: %2) applies mhlo.maximum across dimensions = [1]
      : (tensor<2047x4095xf32>, tensor<f32>) -> tensor<2047xf32>
  %4 = mhlo.convert %3 : tensor<2047xf32>
  %cst = arith.constant dense<1> : tensor<1xi32>
  %5 = mhlo.reshape %4 : (tensor<2047xf32>) -> tensor<2047x1xf32>
  %6 = chlo.broadcast_subtract %arg, %5
      : (tensor<2047x4095xf32>, tensor<2047x1xf32>) -> tensor<2047x4095xf32>
  %7 = mhlo.exponential %6 : tensor<2047x4095xf32>
  %8 = mhlo.convert %7 : tensor<2047x4095xf32>
  %9 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  %10 = mhlo.reduce(%8 init: %9) applies mhlo.add across dimensions = [1]
      : (tensor<2047x4095xf32>, tensor<f32>) -> tensor<2047xf32>
  %11 = mhlo.convert %10 : tensor<2047xf32>
  %cst_0 = arith.constant dense<1> : tensor<1xi32>
  %12 = mhlo.reshape %11 : (tensor<2047xf32>) -> tensor<2047x1xf32>
  %13 = chlo.broadcast_divide %7, %12
      : (tensor<2047x4095xf32>, tensor<2047x1xf32>) -> tensor<2047x4095xf32>
  // CHECK-DAG:  %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG:  %[[BLOCK:.*]] = arith.constant 8
  // CHECK-DAG:  %[[WARP:.*]] = arith.constant 32
  // CHECK-DAG:  %[[GRID:.*]] = arith.constant 256
  // CHECK:      gpu.launch_func @[[MODULE:.*]]::@[[KERNEL:.*]] blocks
  // CHECK-SAME: in (%[[GRID]], %[[ONE]], %[[ONE]])
  // CHECK-SAME: threads in (%[[WARP]], %[[BLOCK]], %[[ONE]])
  // CHECK-SAME: args({{.*}} : memref<2047x4095xf32>,
  // CHECK-SAME: {{.*}} : memref<2047x4095xf32>)
  memref.tensor_store %13, %resbuffer : memref<2047x4095xf32>
  "lmhlo.terminator"() : () -> ()
}
// CHECK: gpu.module @[[MODULE]]
// CHECK: llvm.func @[[KERNEL]]({{.*}}) attributes {gpu.kernel, nvvm.kernel}

// -----

// CHECK-LABEL: @imperfectly_tiled_softmax_4d
// CHECK-SAME:  %[[ARG0:.*]]: memref<6x4x2047x4095xf32>, %[[ARG1:.*]]: memref<6x4x2047x4095xf32>
func.func @imperfectly_tiled_softmax_4d(%argbuffer : memref<6x4x2047x4095xf32>,
    %resbuffer : memref<6x4x2047x4095xf32>) {
  %arg = bufferization.to_tensor %argbuffer : memref<6x4x2047x4095xf32>
  %0 = mhlo.constant dense<-1> : tensor<1xi64>
  %1 = mhlo.convert %arg : tensor<6x4x2047x4095xf32>
  %2 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %3 = mhlo.reduce(%1 init: %2) applies mhlo.maximum across dimensions = [3]
      : (tensor<6x4x2047x4095xf32>, tensor<f32>) -> tensor<6x4x2047xf32>
  %4 = mhlo.convert %3 : tensor<6x4x2047xf32>
  %cst = arith.constant dense<1> : tensor<1xi32>
  %5 = mhlo.reshape %4 : (tensor<6x4x2047xf32>) -> tensor<6x4x2047x1xf32>
  %6 = chlo.broadcast_subtract %arg, %5
      : (tensor<6x4x2047x4095xf32>, tensor<6x4x2047x1xf32>)
      -> tensor<6x4x2047x4095xf32>
  %7 = mhlo.exponential %6 : tensor<6x4x2047x4095xf32>
  %8 = mhlo.convert %7 : tensor<6x4x2047x4095xf32>
  %9 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  %10 = mhlo.reduce(%8 init: %9) applies mhlo.add across dimensions = [3]
      : (tensor<6x4x2047x4095xf32>, tensor<f32>) -> tensor<6x4x2047xf32>
  %11 = mhlo.convert %10 : tensor<6x4x2047xf32>
  %cst_0 = arith.constant dense<1> : tensor<1xi32>
  %12 = mhlo.reshape %11 : (tensor<6x4x2047xf32>) -> tensor<6x4x2047x1xf32>
  %13 = chlo.broadcast_divide %7, %12
      : (tensor<6x4x2047x4095xf32>, tensor<6x4x2047x1xf32>)
      -> tensor<6x4x2047x4095xf32>
  // CHECK:         %[[COLLAPSE_SHAPE:.*]] = memref.collapse_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3{{\]\]}} : memref<6x4x2047x4095xf32> into memref<49128x4095xf32>
  // CHECK:         %[[COLLAPSE_SHAPE_2:.*]] = memref.collapse_shape %[[ARG1]] {{\[\[}}0, 1, 2], [3{{\]\]}} : memref<6x4x2047x4095xf32> into memref<49128x4095xf32>
  // CHECK:         gpu.launch_func  @imperfectly_tiled_softmax_4d_kernel::@imperfectly_tiled_softmax_4d_kernel
  // CHECK-SAME:        args(%[[COLLAPSE_SHAPE]] : memref<49128x4095xf32>, %[[COLLAPSE_SHAPE_2]] : memref<49128x4095xf32>)
  // CHECK:         return
  memref.tensor_store %13, %resbuffer : memref<6x4x2047x4095xf32>
  "lmhlo.terminator"() : () -> ()
}
// CHECK:       gpu.module @imperfectly_tiled_softmax_4d_kernel
// CHECK:         llvm.func @imperfectly_tiled_softmax_4d_kernel(%[[ARG0_0:.*]]: !llvm.ptr<f32>, %[[ARG1_0:.*]]: !llvm.ptr<f32>
