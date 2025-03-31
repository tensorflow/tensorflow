// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// CHECK-LABEL: amd_in_thread_transpose
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 1]], lane = [[0, 2], [0, 4], [0, 8], [2, 0], [4, 0], [8, 0]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose(%arg0: tensor<16x16xf16, #blocked>) {
    // CHECK-DAG:  [[VEC_UNDEF:%.*]] = llvm.mlir.undef : vector<2xf16>
    // CHECK-DAG: [[CST_0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: [[CST_1:%.*]] = llvm.mlir.constant(1 : i32) : i32

    // CHECK-DAG: [[VAL0:%.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(f16, f16, f16, f16)>
    // CHECK-DAG: [[VAL1:%.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(f16, f16, f16, f16)>
    // CHECK-DAG: [[VAL2:%.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(f16, f16, f16, f16)>
    // CHECK-DAG: [[VAL3:%.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(f16, f16, f16, f16)>

    // CHECK-DAG: [[VEC1_TMP:%.*]] = llvm.insertelement [[VAL0]], [[VEC_UNDEF]]{{\[}}[[CST_0]] : i32] : vector<2xf16>
    // CHECK-DAG: [[VEC1:%.*]] = llvm.insertelement [[VAL2]], [[VEC1_TMP]]{{\[}}[[CST_1]] : i32] : vector<2xf16>
    // CHECK-DAG: llvm.store [[VEC1]], {{.*}} {alignment = 4 : i64} : vector<2xf16>, !llvm.ptr<3>

    // CHECK-DAG: [[VEC2_TMP:%.*]] = llvm.insertelement [[VAL1]], [[VEC_UNDEF]]{{\[}}[[CST_0]] : i32] : vector<2xf16>
    // CHECK-DAG: [[VEC2:%.*]] = llvm.insertelement [[VAL3]], [[VEC2_TMP]]{{\[}}[[CST_1]] : i32] : vector<2xf16>
    // CHECK-DAG: llvm.store [[VEC2]], {{.*}} {alignment = 4 : i64} : vector<2xf16>, !llvm.ptr<3>

    %0 = amdgpu.in_thread_transpose %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #linear>
    ttg.local_alloc %0 : (tensor<16x16xf16, #linear>) -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
