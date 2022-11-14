// RUN: mlir-hlo-opt --split-input-file --gpu-fusion-rewrite %s | FileCheck %s

// CHECK: gpu.container_module
// CHECK: gpu.module @fusion_kernel
// CHECK: llvm.func @fusion_kernel
// CHECK-SAME: gpu.kernel
// CHECK-LABEL: func.func @log
// CHECK: gpu.launch_func @fusion_kernel::@fusion_kernel
func.func @log(
    %arg0: memref<8xf32> {lmhlo.params = 0 : index},
    %arg1: memref<8xf32> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8]{0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8xf32>
    %1 = mhlo.log %0 : tensor<8xf32>
    memref.tensor_store %1, %arg1 : memref<8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// Check that no index computations are emitted for flattened tensor.
// We do however, need some index computations to convert from warp and thread
// indices to offset in input/output that that thread should operate on.
// TODO(b/247482325): Optimize this better if needed.
// CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-DAG: %[[C32:.*]] = llvm.mlir.constant(32 : index)
// CHECK-DAG: %[[TIDX:.*]] = nvvm.read.ptx.sreg.tid.x
// CHECK-DAG: %[[TIDY:.*]] = nvvm.read.ptx.sreg.tid.y
// CHECK-DAG: %[[TMP1:.*]] = llvm.mul %[[TIDY]], %[[C32]]
// CHECK-DAG: %[[TMP2:.*]] = llvm.mul %[[TMP1]], %[[C1]]
// CHECK-DAG: %[[WARPOFS:.*]] = llvm.add %[[TMP2]], %[[C0]]
// CHECK: llvm.add %[[WARPOFS]], %[[TIDX]]
// CHECK-NOT: llvm.mul
// CHECK-NOT: llvm.add
// CHECK-LABEL: func.func @multidimensional
func.func @multidimensional(
    %arg0: memref<8x8xf32> {lmhlo.params = 0 : index},
    %arg1: memref<8x8xf32> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8,8]{1,0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8x8xf32>
    %1 = mhlo.abs %0 : tensor<8x8xf32>
    memref.tensor_store %1, %arg1 : memref<8x8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func.func @twice
// CHECK-DAG: gpu.launch_func @fusion_kernel::@fusion_kernel
// CHECK-DAG: gpu.launch_func @fusion_kernel_0::@fusion_kernel
func.func @twice(
    %arg0: memref<8xf32> {lmhlo.params = 0 : index},
    %arg1: memref<8xf32> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8]{0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8xf32>
    %1 = mhlo.log %0 : tensor<8xf32>
    memref.tensor_store %1, %arg1 : memref<8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8xf32>
    %1 = mhlo.log %0 : tensor<8xf32>
    memref.tensor_store %1, %arg1 : memref<8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func.func @empty
// CHECK-NOT: gpu.launch_func
func.func @empty(
    %arg0: memref<8xf32> {lmhlo.params = 0 : index},
    %arg1: memref<8xf32> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8]{0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8xf32>
    memref.tensor_store %0, %arg1 : memref<8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func.func @tanh
// CHECK: gpu.launch_func
func.func @tanh(
    %arg0: memref<8xf32> {lmhlo.params = 0 : index},
    %arg1: memref<8xf32> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8]{0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8xf32>
    %1 = mhlo.tanh %0 : tensor<8xf32>
    memref.tensor_store %1, %arg1 : memref<8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// Checks that binary ops are rewritten.
// CHECK-LABEL: func.func @add
// CHECK: gpu.launch_func
func.func @add(
    %arg0: memref<8xf32> {lmhlo.params = 0 : index},
    %arg1: memref<8xf32> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8]{0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8xf32>
    %1 = mhlo.add %0, %0 : tensor<8xf32>
    memref.tensor_store %1, %arg1 : memref<8xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// Checks that unsigned integer types are not rewritten.
// CHECK-LABEL: func.func @negate
// CHECK-NOT: gpu.launch_func
func.func @negate(
    %arg0: memref<8xui16> {lmhlo.params = 0 : index},
    %arg1: memref<8xui16> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8]{0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<8xui16>
    %1 = mhlo.negate %0 : tensor<8xui16>
    memref.tensor_store %1, %arg1 : memref<8xui16>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// Checks that complex typed ops are not rewritten.
// CHECK-LABEL: func.func @complex
// CHECK-NOT: gpu.launch_func
func.func @complex(
    %arg0: memref<4xcomplex<f32>> {lmhlo.params = 0 : index},
    %arg1: memref<4xcomplex<f32>> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[8]{0}"} {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %arg0 : memref<4xcomplex<f32>>
    %1 = mhlo.negate %0 : tensor<4xcomplex<f32>>
    memref.tensor_store %1, %arg1 : memref<4xcomplex<f32>>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK: gpu.container_module
// CHECK: gpu.module @fusion_kernel
// CHECK: llvm.func @fusion_kernel
// CHECK-SAME: gpu.kernel
// CHECK-LABEL: func.func @softmax
// CHECK: gpu.launch_func @fusion_kernel::@fusion_kernel
func.func @softmax(
    %arg0: memref<16384xi8> {lmhlo.params = 0 : index},
    %arg1: memref<16384xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[128,32]{1,0}"} {
  %c0 = arith.constant 0 : index
  %view = memref.view %arg0[%c0][] : memref<16384xi8> to memref<128x32xf32>
  %c0_0 = arith.constant 0 : index
  %view_1 = memref.view %arg1[%c0_0][] : memref<16384xi8> to memref<128x32xf32>
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %view : memref<128x32xf32>
    %1 = mhlo.constant dense<0.0> : tensor<f32>
    %2 = mhlo.reduce(%0 init: %1) across dimensions = [1] : (tensor<128x32xf32>, tensor<f32>) -> tensor<128xf32>
    reducer(%arg3: tensor<f32>, %arg4: tensor<f32>) {
      %5 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      mhlo.return %5 : tensor<f32>
    }
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<128x32xf32>
    %4 = mhlo.subtract %0, %3 : tensor<128x32xf32>
    memref.tensor_store %4, %view_1 : memref<128x32xf32>
    "lmhlo.terminator"() : () -> ()
  }) {fusion_type = "softmax_fusion"} : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK: gpu.container_module
// CHECK: gpu.module @fusion_kernel
// CHECK: llvm.func @fusion_kernel
// CHECK-SAME: gpu.kernel
// CHECK-LABEL: func.func @complete_softmax
// CHECK: gpu.launch_func @fusion_kernel::@fusion_kernel
func.func @complete_softmax(
    %arg0: memref<640xi8> {lmhlo.params = 0 : index},
    %arg1: memref<640xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[32,5]{1,0}"} {
  %c0 = arith.constant 0 : index
  %view = memref.view %arg0[%c0][] : memref<640xi8> to memref<32x5xf32>
  %c0_0 = arith.constant 0 : index
  %view_1 = memref.view %arg1[%c0_0][] : memref<640xi8> to memref<32x5xf32>
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %view : memref<32x5xf32>
    %1 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %2 = mhlo.reduce(%0 init: %1) across dimensions = [1] : (tensor<32x5xf32>, tensor<f32>) -> tensor<32xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %10 = mhlo.maximum %arg4, %arg5 : tensor<f32>
      mhlo.return %10 : tensor<f32>
    }
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<32x5xf32>
    %4 = mhlo.subtract %0, %3 : tensor<32x5xf32>
    %5 = mhlo.exponential %4 : tensor<32x5xf32>
    %6 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = mhlo.reduce(%5 init: %6) across dimensions = [1] : (tensor<32x5xf32>, tensor<f32>) -> tensor<32xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %10 = mhlo.add %arg4, %arg5 : tensor<f32>
      mhlo.return %10 : tensor<f32>
    }
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<32x5xf32>
    %9 = mhlo.divide %5, %8 : tensor<32x5xf32>
    memref.tensor_store %9, %view_1 : memref<32x5xf32>
    "lmhlo.terminator"() : () -> ()
  }) {fusion_type = "softmax_fusion"} : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

func.func @softmax_4d(
    %arg0: memref<16384xi8> {lmhlo.params = 0 : index},
    %arg1: memref<16384xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}
) attributes {result_xla_shape = "f32[128,32]{1,0}"} {
  %c0 = arith.constant 0 : index
  %view = memref.view %arg0[%c0][] : memref<16384xi8> to memref<4x4x8x32xf32>
  %c0_0 = arith.constant 0 : index
  %view_1 = memref.view %arg1[%c0_0][]
      : memref<16384xi8> to memref<4x4x8x32xf32>
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %view : memref<4x4x8x32xf32>
    %1 = mhlo.constant dense<0.0> : tensor<f32>
    %2 = mhlo.reduce(%0 init: %1) across dimensions = [3]
        : (tensor<4x4x8x32xf32>, tensor<f32>) -> tensor<4x4x8xf32>
    reducer(%arg3: tensor<f32>, %arg4: tensor<f32>) {
      %5 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      mhlo.return %5 : tensor<f32>
    }
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[0, 1, 2]>
        : tensor<3xi64>} : (tensor<4x4x8xf32>) -> tensor<4x4x8x32xf32>
    %4 = mhlo.subtract %0, %3 : tensor<4x4x8x32xf32>
    memref.tensor_store %4, %view_1 : memref<4x4x8x32xf32>
    "lmhlo.terminator"() : () -> ()
  }) {fusion_type = "softmax_fusion"} : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      module attributes {gpu.container_module}
// CHECK:        gpu.module @fusion_kernel
// CHECK:          llvm.func @fusion_kernel(%[[ARG0:.*]]: !llvm.ptr<f32>, %[[ARG1:.*]]: !llvm.ptr<f32>)

// CHECK:        func.func @softmax_4d(%[[ARG0_0:.*]]: memref<16384xi8> {lmhlo.params = 0 : index}, %[[ARG1_0:.*]]: memref<16384xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}) attributes {result_xla_shape = "f32[128,32]{1,0}"}
// CHECK:          %[[VIEW:.*]] = memref.view %[[ARG0_0]][%{{.*}}][] : memref<16384xi8> to memref<4x4x8x32xf32>
// CHECK:          %[[VIEW_1:.*]] = memref.view %[[ARG1_0]][%{{.*}}][] : memref<16384xi8> to memref<4x4x8x32xf32>
// CHECK:          %[[COLLAPSE_SHAPE:.*]] = memref.collapse_shape %[[VIEW]] {{\[\[}}0, 1, 2], [3{{\]\]}} : memref<4x4x8x32xf32> into memref<128x32xf32>
// CHECK:          %[[COLLAPSE_SHAPE_4:.*]] = memref.collapse_shape %[[VIEW_1]] {{\[\[}}0, 1, 2], [3{{\]\]}} : memref<4x4x8x32xf32> into memref<128x32xf32>
// CHECK:          gpu.launch_func  @fusion_kernel::@fusion_kernel
// CHECK-SAME:         args(%[[COLLAPSE_SHAPE]] : memref<128x32xf32>, %[[COLLAPSE_SHAPE_4]] : memref<128x32xf32>)
