// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx942" -split-input-file | FileCheck %s

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: shortcut_mfma16
  tt.func public @shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: no_shortcut_mfma16
  tt.func public @no_shortcut_mfma16(%arg0: tensor<16x16xf16, #mfma>) {
    // CHECK: store
    // CHECK: load
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #mfma> -> tensor<16x16xf16, #dotop>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_dot_cvt_f8_mfma32
  tt.func public @mfma_dot_cvt_f8_mfma32(%arg0: tensor<128x32xf8E4M3FNUZ, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load

    // CHECK: [[val3:%.*]] = llvm.extractvalue %arg0[3]
    // CHECK: [[val7:%.*]] = llvm.extractvalue %arg0[7]

    // CHECK-DAG: [[c32:%.*]] = llvm.mlir.constant(32 : i32)
    // CHECK-DAG: [[c64:%.*]] = llvm.mlir.constant(64 : i32)

    // CHECK: [[threadId:%.*]] = rocdl.workitem.id.x
    // CHECK: [[laneId:%.*]] = llvm.urem [[threadId]], [[c64]]
    // CHECK: [[mask0:%.*]] = llvm.icmp "slt" [[laneId]], [[c32]]

    // CHECK: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c32]]
    // CHECK: [[addr32:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // CHECK: [[vec0:%.*]] = llvm.insertelement [[val3]], {{.*}} : vector<4xi8>
    // CHECK: [[vec1:%.*]] = llvm.insertelement [[val7]], {{.*}} : vector<4xi8>

    // CHECK: [[bvec0:%.*]] = llvm.bitcast [[vec0]]
    // CHECK: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // CHECK: [[bShflVec0:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec0]]
    // CHECK: [[shflVec0:%.*]] = llvm.bitcast [[bShflVec0]]

    // CHECK: [[bvec1:%.*]] = llvm.bitcast [[vec1]]
    // CHECK: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // CHECK: [[bShflVec1:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec1]]
    // CHECK: [[shflVec1:%.*]] = llvm.bitcast [[bShflVec1]]

    // Input (8 values): (vec0, vec1)
    // Output (8 values shuffled, '>> n' - take the value from (lane + n) % 64):
    //                 resVec0     resVec1
    //   lanes  0-31: (vec0      , vec0 >> 32) (mask0=1)
    //   lanes 32-63: (vec1 >> 32, vec1      ) (mask0=0)

    // CHECK: [[resVec0:%.*]] = llvm.select [[mask0]], [[vec0]], [[shflVec1]]
    // CHECK: [[resVec1:%.*]] = llvm.select [[mask0]], [[shflVec0]], [[vec1]]

    // CHECK: [[c3:%.*]] = llvm.mlir.constant(3 : i32)
    // CHECK: [[resVal3:%.*]] = llvm.extractelement [[resVec0]][[[c3]] : i32] : vector<4xi8>
    // CHECK: [[c3:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: [[resVal7:%.*]] = llvm.extractelement [[resVec1]][[[c3]] : i32] : vector<4xi8>

    // CHECK: llvm.insertvalue [[resVal3]], {{.*}}[3]
    // CHECK: llvm.insertvalue [[resVal7]], {{.*}}[7]

    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E4M3FNUZ, #mfma> -> tensor<128x32xf8E4M3FNUZ, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_dot_cvt_bf8_mfma32
  tt.func public @mfma_dot_cvt_bf8_mfma32(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK: rocdl.ds_bpermute
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_dot_cvt_f8_mfma16
  tt.func public @mfma_dot_cvt_f8_mfma16(%arg0: tensor<128x32xf8E4M3FNUZ, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load

    // CHECK: [[val3:%.*]] = llvm.extractvalue %arg0[3]
    // CHECK: [[val7:%.*]] = llvm.extractvalue %arg0[7]

    // CHECK-DAG: [[c16:%.*]] = llvm.mlir.constant(16 : i32)
    // CHECK-DAG: [[c32:%.*]] = llvm.mlir.constant(32 : i32)
    // CHECK-DAG: [[c48:%.*]] = llvm.mlir.constant(48 : i32)
    // CHECK-DAG: [[c64:%.*]] = llvm.mlir.constant(64 : i32)

    // CHECK: [[threadId:%.*]] = rocdl.workitem.id.x
    // CHECK: [[laneId:%.*]] = llvm.urem [[threadId]], [[c64]]
    // CHECK: [[mask0:%.*]] = llvm.icmp "slt" [[laneId]], [[c32]]

    // CHECK: [[laneIdRem:%.*]] = llvm.urem [[laneId]], [[c32]]
    // CHECK: [[mask1:%.*]] = llvm.icmp "slt" [[laneIdRem]], [[c16]]

    // CHECK: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c16]]
    // CHECK: [[addr16:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // CHECK: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c32]]
    // CHECK: [[addr32:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // CHECK: [[shflLaneId:%.*]] = llvm.add [[laneId]], [[c48]]
    // CHECK: [[addr48:%.*]] = llvm.urem [[shflLaneId]], [[c64]]

    // CHECK: [[vec0:%.*]] = llvm.insertelement [[val3]], {{.*}} : vector<4xi8>
    // CHECK: [[vec1:%.*]] = llvm.insertelement [[val7]], {{.*}} : vector<4xi8>

    // CHECK: [[bvec0:%.*]] = llvm.bitcast [[vec0]]
    // CHECK: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[addr:%.*]] = llvm.shl [[addr16]], [[c2]]
    // CHECK: [[bShflVec0_16:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec0]]
    // CHECK: [[shflVec0_16:%.*]] = llvm.bitcast [[bShflVec0_16]]

    // CHECK: [[bvec0:%.*]] = llvm.bitcast [[vec0]]
    // CHECK: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // CHECK: [[bShflVec0_32:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec0]]
    // CHECK: [[shflVec0_32:%.*]] = llvm.bitcast [[bShflVec0_32]]

    // CHECK: [[bvec1:%.*]] = llvm.bitcast [[vec1]]
    // CHECK: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[addr:%.*]] = llvm.shl [[addr32]], [[c2]]
    // CHECK: [[bShflVec1_32:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec1]]
    // CHECK: [[shflVec1_32:%.*]] = llvm.bitcast [[bShflVec1_32]]

    // CHECK: [[bvec1:%.*]] = llvm.bitcast [[vec1]]
    // CHECK: [[c2:%.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: [[addr:%.*]] = llvm.shl [[addr48]], [[c2]]
    // CHECK: [[bShflVec1_48:%.*]] = rocdl.ds_bpermute [[addr]], [[bvec1]]
    // CHECK: [[shflVec1_48:%.*]] = llvm.bitcast [[bShflVec1_48]]

    // Input (8 values): (vec0, vec1)
    // Output (8 values shuffled, '>> n' - take the value from (lane + n) % 64):
    //                 resVec0     resVec1
    //   lanes  0-15: (vec0      , vec0 >> 16) (mask0=1, mask1=1)
    //   lanes 16-31: (vec0 >> 16, vec0 >> 32) (mask0=1, mask1=0)
    //   lanes 32-47: (vec1 >> 32, vec1 >> 48) (mask0=0, mask1=1)
    //   lanes 48-63: (vec1 >> 48, vec1      ) (mask0=0, mask1=0)

    // CHECK-DAG: [[mask0_true:%.*]] = llvm.select [[mask1]], [[vec0]], [[shflVec0_16]] : i1, vector<4xi8>
    // CHECK-DAG: [[mask0_false:%.*]] = llvm.select [[mask1]], [[shflVec1_32]], [[shflVec1_48]] : i1, vector<4xi8>
    // CHECK: [[resVec0:%.*]] = llvm.select [[mask0]], [[mask0_true]], [[mask0_false]] : i1, vector<4xi8>

    // CHECK-DAG: [[mask0_true:%.*]] = llvm.select [[mask1]], [[shflVec0_16]], [[shflVec0_32]] : i1, vector<4xi8>
    // CHECK-DAG: [[mask0_false:%.*]] = llvm.select [[mask1]], [[shflVec1_48]], [[vec1]] : i1, vector<4xi8>
    // CHECK: [[resVec1:%.*]] = llvm.select [[mask0]], [[mask0_true]], [[mask0_false]] : i1, vector<4xi8>

    // CHECK: [[c3:%.*]] = llvm.mlir.constant(3 : i32)
    // CHECK: [[resVal3:%.*]] = llvm.extractelement [[resVec0]][[[c3]] : i32] : vector<4xi8>
    // CHECK: [[c3:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK: [[resVal7:%.*]] = llvm.extractelement [[resVec1]][[[c3]] : i32] : vector<4xi8>

    // CHECK: llvm.insertvalue [[resVal3]], {{.*}}[3]
    // CHECK: llvm.insertvalue [[resVal7]], {{.*}}[7]

    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E4M3FNUZ, #mfma> -> tensor<128x32xf8E4M3FNUZ, #dotop0>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_dot_cvt_bf8_mfma16
  tt.func public @mfma_dot_cvt_bf8_mfma16(%arg0: tensor<128x32xf8E5M2, #mfma>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK: rocdl.ds_bpermute
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf8E5M2, #mfma> -> tensor<128x32xf8E5M2, #dotop0>
    tt.return
  }
}
