// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#linear = #ttg.linear<{register = [[0, 2], [2, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @reduce_linear_layout
tt.func private @reduce_linear_layout(%arg0: tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> {
  // CHECK-NEXT: [[SRC0:%.*]] = extractvalue {{.*}} %0, 0
  // CHECK-NEXT: [[SRC1:%.*]] = extractvalue {{.*}} %0, 1
  // CHECK-NEXT: [[SRC2:%.*]] = extractvalue {{.*}} %0, 2
  // CHECK-NEXT: [[SRC3:%.*]] = extractvalue {{.*}} %0, 3

  // The layout looks lke
  // [[  T0:0,  T32:0,   T0:1,  T32:1, ...
  // [   T4:0,  T36:0,   T4:1,  T36:1, ...
  // [   T0:2,  T32:2,   T0:3,  T32:3, ...
  // [   T4:2,  T36:2,   T4:3,  T36:3,
  // ...
  //
  // A reduction along axis=0 consists of adding registers (0, 2) and (1, 3)
  // before shuffling.
  //
  // Columns along axis=0 are contained within a warp, so reduction arcoss warps
  // is not needed.

  // Reduce within threads
  // CHECK-NEXT: [[SUM0:%.*]] = add i32 [[SRC0]], [[SRC2]]
  // CHECK-NEXT: [[SUM1:%.*]] = add i32 [[SRC1]], [[SRC3]]

  // Reduce within warp.
  // CHECK-NEXT: [[W0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SUM0]], i32 16, i32 31)
  // CHECK-NEXT: [[WSUM0:%.*]] = add i32 [[W0]], [[SUM0]]
  // CHECK-NEXT: [[W1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM0]], i32 8, i32 31)
  // CHECK-NEXT: [[WSUM1:%.*]] = add i32 [[WSUM0]], [[W1]]
  // CHECK-NEXT: [[W2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM1]], i32 4, i32 31)
  // CHECK-NEXT: [[WSUM2:%.*]] = add i32 [[WSUM1]], [[W2]]
  // CHECK-NEXT: [[W3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM2]], i32 2, i32 31)
  // CHECK-NEXT: [[WSUM3:%.*]] = add i32 [[WSUM2]], [[W3]]

  // CHECK-NEXT: [[W4:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SUM1]], i32 16, i32 31)
  // CHECK-NEXT: [[WSUM4:%.*]] = add i32 [[W4]], [[SUM1]]
  // CHECK-NEXT: [[W5:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM4]], i32 8, i32 31)
  // CHECK-NEXT: [[WSUM5:%.*]] = add i32 [[WSUM4]], [[W5]]
  // CHECK-NEXT: [[W6:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM5]], i32 4, i32 31)
  // CHECK-NEXT: [[WSUM6:%.*]] = add i32 [[WSUM5]], [[W6]]
  // CHECK-NEXT: [[W7:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM6]], i32 2, i32 31)
  // CHECK-NEXT: [[WSUM7:%.*]] = add i32 [[WSUM6]], [[W7]]

  // CHECK-NEXT: [[DST0:%.*]] = insertvalue { i32, i32 } undef, i32 [[WSUM3]], 0
  // CHECK-NEXT: [[DST1:%.*]] = insertvalue { i32, i32 } [[DST0]], i32 [[WSUM7]], 1

  %0 = "tt.reduce"(%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 0 : i32} : (tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>

  // CHECK-NEXT: ret { i32, i32 } [[DST1]]
  tt.return %0 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
}

tt.func @anchor(%ptr: !llvm.ptr, %arg0: tensor<32x16xi32, #linear>) {
  %0 = tt.call @reduce_linear_layout(%arg0) : (tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
  %1 = builtin.unrealized_conversion_cast %0 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> to !llvm.struct<(i32, i32)>
  llvm.store volatile %1, %ptr : !llvm.struct<(i32, i32)>, !llvm.ptr
  tt.return
}

}
