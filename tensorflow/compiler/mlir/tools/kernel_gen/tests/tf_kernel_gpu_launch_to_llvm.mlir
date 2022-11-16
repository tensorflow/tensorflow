// RUN: kernel-gen-opt %s -tf-kernel-to-llvm -split-input-file | FileCheck %s --dump-input=always

// CHECK-LABEL: module @main
module @main attributes {gpu.container_module} {

// CHECK-NOT: gpu.module @kernel_module
gpu.module @kernel_module attributes {gpu.binary_blob = "BLOB!"} {
  llvm.func @the_kernel() attributes {gpu.kernel} {
    llvm.return
  }
}

// CHECK: llvm.func @_mlir_ciface_tf_launch_kernel(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, !llvm.ptr<ptr<i8>>)
// CHECK-DAG: llvm.mlir.global internal constant @kernel_module_the_kernel_kernel_name("the_kernel\00")
// CHECK-DAG: llvm.mlir.global internal constant @kernel_module_blob("BLOB!")

// CHECK-LABEL: llvm.func @launch
// CHECK-SAME: (%[[CTX:.*]]: !llvm.ptr<i8>, %{{.*}}: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64
func.func @launch(%ctx: !tf_framework.op_kernel_context, %memref: memref<?x10xf32>) {
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[BLOB:.*]] = llvm.mlir.addressof @kernel_module_blob : !llvm.ptr<array<5 x i8>>
  // CHECK: %[[BLOB_PTR:.*]] = llvm.getelementptr %[[BLOB]][0, 0] : (!llvm.ptr<array<5 x i8>>) -> !llvm.ptr<i8>
  // CHECK: %[[NAME:.*]] = llvm.mlir.addressof @kernel_module_the_kernel_kernel_name : !llvm.ptr<array<11 x i8>>
  // CHECK: %[[NAME_PTR:.*]] = llvm.getelementptr %[[NAME]][0, 0] : (!llvm.ptr<array<11 x i8>>) -> !llvm.ptr<i8>
  // CHECK: %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK: %[[ARGS:.*]] = llvm.alloca %22 x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.call @_mlir_ciface_tf_launch_kernel(%[[CTX]], %[[BLOB_PTR]], %[[NAME_PTR]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[ARGS]])
  %c1 = arith.constant 1 : index
  gpu.launch_func  @kernel_module::@the_kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1)
      args(%memref: memref<?x10xf32>)
  func.return
}

}
