// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:   --hlo-to-triton-pipeline="block-tile=1" \
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
  // CHECK-DAG:  %[[GRID:.*]] = arith.constant 2048
  // CHECK:      gpu.launch_func @[[MODULE:.*]]::@[[KERNEL:.*]] blocks
  // CHECK-SAME: in (%[[GRID]], %[[ONE]], %[[ONE]])
  // CHECK-SAME: threads in (%[[ONE]], %[[ONE]], %[[ONE]])
  // CHECK-SAME: args({{.*}} : memref<2048x4096xf32>,
  // CHECK-SAME: {{.*}} : memref<2048x4096xf32>)
  memref.tensor_store %13, %resbuffer : memref<2048x4096xf32>
  "lmhlo.terminator"() : () -> ()
}
// CHECK:      gpu.module @[[MODULE]]
// CHECK:      gpu.func @[[KERNEL]]
  /// TODO(b/261710844): This should be Triton Dialect.
// CHECK-SAME: %[[IN:.*]]: memref<2048x4096xf32>,
// CHECK-SAME: %[[OUT:.*]]: memref<2048x4096xf32>) kernel
// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      %[[BID:.*]] = gpu.block_id  x
// CHECK:      %[[SV:.*]] = memref.subview %[[IN]][%[[BID]], 0] [1, 4096]
// CHECK:      %[[VEC:.*]] = vector.transfer_read %[[SV]][%[[C0]], %[[C0]]]
// CHECK:      %[[MAX:.*]] = vector.multi_reduction <maxf>, %[[VEC]]
// CHECK:      %[[BMAX:.*]] = vector.broadcast %[[MAX]]
// CHECK:      %[[SUB:.*]] = arith.subf %[[VEC]], %[[BMAX]]
// CHECK:      %[[EXP:.*]] = math.exp %[[SUB]]
// CHECK:      %[[SUM:.*]] = vector.multi_reduction <add>, %[[EXP]]
// CHECK:      %[[BSUM:.*]] = vector.broadcast %[[SUM]]
// CHECK:      %[[DIV:.*]] = arith.divf %[[EXP]], %[[BSUM]]
// CHECK:      vector.transfer_write %[[DIV]], %[[OUT]][%[[BID]], %[[C0]]]