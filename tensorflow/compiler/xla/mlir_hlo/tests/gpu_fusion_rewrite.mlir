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
// CHECK-NOT: gpu.launch_func
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
