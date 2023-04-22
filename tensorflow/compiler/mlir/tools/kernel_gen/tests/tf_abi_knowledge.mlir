// RUN: kernel-gen-opt %s -allow-unregistered-dialect -propagate-tf-abi-knowledge-to-kernels -split-input-file | FileCheck %s --check-prefixes=CHECK,ABI
// RUN: kernel-gen-opt %s -allow-unregistered-dialect -propagate-shape-knowledge-to-kernels -split-input-file | FileCheck %s --check-prefixes=CHECK,SHAPE

// The input is taken from what is actually used in kernel generator lowering
// for unary operations.

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @abs
  func @abs(%ctx: !tf_framework.op_kernel_context, %arg0: memref<*xf32>, %size: index)
      attributes {tf_entry} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %13 = memref.alloca() : memref<1xindex>
    memref.store %size, %13[%c0] : memref<1xindex>
    %14 = memref.reshape %arg0(%13) : (memref<*xf32>, memref<1xindex>) -> memref<?xf32>
    %15 = memref.dim %14, %c0 : memref<?xf32>
    %16 = tf_framework.alloc(%ctx, %15) : memref<?xf32>
    gpu.launch_func @abs_kernel::@abs_kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%14 : memref<?xf32>, %16 : memref<?xf32>)
    return
  }

  // CHECK-LABEL: gpu.module @abs_kernel
  gpu.module @abs_kernel {
    // CHECK-LABEL: llvm.func @abs_kernel
    // ABI-SAME: %[[ARG0:.*]]: !llvm.ptr<f32>, %[[ARG1:.*]]: !llvm.ptr<f32> {llvm.align = 16 : index},
    // ABI-SAME: %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: !llvm.ptr<f32>, %[[ARG6:.*]]: !llvm.ptr<f32> {llvm.align = 16 : index, llvm.noalias},
    // ABI-SAME: %[[ARG7:.*]]: i64, %[[ARG8:.*]]: i64, %[[ARG9:.*]]: i64
    // SHAPE-SAME: %[[ARG0:.*]]: !llvm.ptr<f32>, %[[ARG1:.*]]: !llvm.ptr<f32>, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: !llvm.ptr<f32>, %[[ARG6:.*]]: !llvm.ptr<f32>, %[[ARG7:.*]]: i64, %[[ARG8:.*]]: i64, %[[ARG9:.*]]: i64
    llvm.func @abs_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<f32>, %arg6: !llvm.ptr<f32>, %arg7: i64, %arg8: i64, %arg9: i64) attributes {gpu.kernel} {
      // ABI: %[[ZERO:.*]] = llvm.mlir.constant(0 : index)
      // ABI: %[[ONE:.*]] = llvm.mlir.constant(1 : index)
      // CHECK: llvm.mlir.undef
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ARG1]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG0]]
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.insertvalue %[[ARG1]]
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ZERO]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG2]]
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.insertvalue %[[ARG3]]
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ONE]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG4]]
      %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.mlir.undef
      %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ARG6]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG5]]
      %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // CHECK-NEXT: llvm.insertvalue %[[ARG6]]
      %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ZERO]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG7]]
      %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ARG8]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG3]]
      %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[ONE]]
      // SHAPE-NEXT: llvm.insertvalue %[[ARG4]]
      %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      llvm.return
      // CHECK-NEXT: llvm.return
    }
  }
}

// -----

// Binary op without broadcasting (same shape).

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @add_same_shape
  func @add_same_shape(%arg0: !tf_framework.op_kernel_context, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %size: index)
      attributes {tf_entry} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %82 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%size], strides: [%c1]: memref<*xf32> to memref<?xf32>
    %83 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%size], strides: [%c1]: memref<*xf32> to memref<?xf32>
    %84 = tf_framework.alloc(%arg0, %size) : memref<?xf32>
    gpu.launch_func  @AddV2_kernel_1::@AddV2_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%size : index, %82 : memref<?xf32>, %83 : memref<?xf32>, %84 : memref<?xf32>)
    return
  }

  // CHECK-LABEL: gpu.module @AddV2_kernel_1
  gpu.module @AddV2_kernel_1 {
    // CHECK-LABEL: llvm.func @AddV2_kernel
    // ABI-SAME: {llvm.align = 16 : index}
    // ABI-SAME: {llvm.align = 16 : index}
    // ABI-SAME: {llvm.align = 16 : index, llvm.noalias}
    llvm.func @AddV2_kernel(%arg0: i64, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr<f32>, %arg7: !llvm.ptr<f32>, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i64, %arg14: i64, %arg15: i64) attributes {gpu.kernel} {
      // ABI: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
      // ABI: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP:.*]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[STR:.*]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR1:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR1]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[STR]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %12 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %17 = llvm.insertvalue %arg15, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR2:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR2]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[STR]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      llvm.return
      // CHECK-NEXT: llvm.return
    }
  }
}

// -----

// Test binary op with broadcasting - 2d case.

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @add_same_shape
  func @add_same_shape(%arg0: !tf_framework.op_kernel_context, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %size0: index, %size1: index, %stride0: index, %stride1: index)
      attributes {tf_entry} {
    %c1 = constant 1 : index
    %216 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%size1, %size0], strides: [%size0, %c1]: memref<*xf32> to memref<?x?xf32>
    %241 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%size1, %size0], strides: [%size0, %c1]: memref<*xf32> to memref<?x?xf32>
    %304 = memref.reinterpret_cast %216 to offset: [0], sizes: [%size1, %size0], strides: [%stride1, %stride0]: memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>
    %309 = memref.reinterpret_cast %241 to offset: [0], sizes: [%size1, %size0], strides: [%stride0, %stride1]: memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>
    %310 = tf_framework.alloc(%arg0, %size1, %size0) : memref<?x?xf32>
    gpu.launch_func  @AddV2_kernel_3::@AddV2_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%size0 : index, %size1 : index, %310 : memref<?x?xf32>, %304 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>, %309 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>)
    return
  }

  // CHECK-LABEL: gpu.module @AddV2_kernel_3
  gpu.module @AddV2_kernel_3 {
    // CHECK-LABEL: llvm.func @AddV2_kernel
    // ABI-SAME: {llvm.align = 16 : index, llvm.noalias}
    // ABI-SAME: {llvm.align = 16 : index}
    // ABI-SAME: {llvm.align = 16 : index}
    llvm.func @AddV2_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr<f32>, %arg3: !llvm.ptr<f32> {llvm.align = 16 : index, llvm.noalias}, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr<f32>, %arg10: !llvm.ptr<f32> {llvm.align = 16 : index}, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr<f32>, %arg17: !llvm.ptr<f32> {llvm.align = 16 : index}, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel} {
      // ABI: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
      // ABI: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg6, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP0:.*]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[STR0:.*]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP1:.*]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[STR1:.*]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg13, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg15, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NOT: llvm.insertvalue %[[C1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP0]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: llvm.insertvalue %[[STR0]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: llvm.insertvalue %[[SHP1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: llvm.insertvalue %[[STR1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg16, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %arg17, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.insertvalue %arg18, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %20 = llvm.insertvalue %arg19, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %21 = llvm.insertvalue %arg21, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %22 = llvm.insertvalue %arg20, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %23 = llvm.insertvalue %arg22, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // ABI-NOT: llvm.insertvalue %[[C1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP0]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: llvm.insertvalue %[[STR0]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE: llvm.insertvalue %[[SHP1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      // SHAPE-NOT: llvm.insertvalue %[[STR1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      llvm.return
      // CHECK: llvm.return
    }
  }
}

// -----


// Test binary op with broadcasting a scalar.

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-LABEL: func @add_one_scalar
  func @add_one_scalar(%arg0: !tf_framework.op_kernel_context, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %size: index)
      attributes {tf_entry} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %13 = memref.cast %arg1 : memref<*xf32> to memref<f32>
    %26 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%size], strides: [%c1]: memref<*xf32> to memref<?xf32>
    %27 = memref.reinterpret_cast %13 to offset: [0], sizes: [%size], strides: [%c0]: memref<f32> to memref<?xf32, #map0>
    %28 = memref.reinterpret_cast %26 to offset: [0], sizes: [%size], strides: [%c1]: memref<?xf32> to memref<?xf32, #map0>
    %29 = tf_framework.alloc(%arg0, %size) : memref<?xf32>
    gpu.launch_func  @AddV2_kernel::@AddV2_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%size : index, %29 : memref<?xf32>, %27 : memref<?xf32, #map0>, %28 : memref<?xf32, #map0>)
    return
  }
  // CHECK-LABEL: gpu.module @AddV2_kernel
  gpu.module @AddV2_kernel {
    // CHECK-LABEL: llvm.func @AddV2_kernel
    // ABI-SAME: {llvm.align = 16 : index, llvm.noalias}
    // ABI-SAME: {llvm.align = 16 : index}
    // ABI-SAME: {llvm.align = 16 : index}
    llvm.func @AddV2_kernel(%arg0: i64, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr<f32>, %arg7: !llvm.ptr<f32>, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i64, %arg14: i64, %arg15: i64) attributes {gpu.kernel} {
      // ABI: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
      // ABI: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR0]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP:.*]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[STR:.*]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR1:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR1]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NOT: llvm.insertvalue %[[STR]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %12 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %17 = llvm.insertvalue %arg15, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR2:.*]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[PTR2]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // ABI-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NEXT: llvm.insertvalue %[[SHP]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      // SHAPE-NOT: llvm.insertvalue %[[STR]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      llvm.return
      // CHECK: llvm.return
    }
  }
}

