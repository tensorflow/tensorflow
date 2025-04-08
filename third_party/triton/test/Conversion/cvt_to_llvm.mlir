// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>

#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {

// CHECK-LABEL: convert_layout_blocked_blocked_vec
tt.func private @convert_layout_blocked_blocked_vec(%arg0: tensor<16x16xi32, #blocked0>) -> tensor<16x16xi32, #blocked2> {

  // CHECK-NEXT: [[SRC0:%.*]] = extractvalue {{.*}} %0, 0
  // CHECK-NEXT: [[SRC1:%.*]] = extractvalue {{.*}} %0, 1
  // CHECK-NEXT: [[SRC2:%.*]] = extractvalue {{.*}} %0, 2
  // CHECK-NEXT: [[SRC3:%.*]] = extractvalue {{.*}} %0, 3
  // CHECK-NEXT: [[SRC4:%.*]] = extractvalue {{.*}} %0, 4
  // CHECK-NEXT: [[SRC5:%.*]] = extractvalue {{.*}} %0, 5
  // CHECK-NEXT: [[SRC6:%.*]] = extractvalue {{.*}} %0, 6
  // CHECK-NEXT: [[SRC7:%.*]] = extractvalue {{.*}} %0, 7

  // CHECK-NEXT: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()

  // The layout conversion looks like
  //             dst_lane
  // dst_reg     0      1      2      3   ...  16     17     18     19  ...
  //  0          T0:0   T1:0   T4:0   T5:0     T0:4   T1:4   T4:4   T5:4
  //  1          T0:1   T1:1   T4:1   T5:1     T0:5   T1:5   T4:5   T5:5
  //  ...
  //  4          T2:0   T3:0   T6:0   T7:0     T2:4   T3:4   T6:4   T7:4
  //  5          T2:1   T3:1   T6:1   T7:1     T2:5   T3:5   T6:5   T7:5
  //  ...
  //
  // This subsection is tiled to fill the rest of the lanes and registers.
  //
  // There will need to be one select per shuffle input and one select per
  // shuffle output due to src registers (i%4, (i%4)+4) mapped to the same dst
  // register.

  // Lanes [2, 3, 6, 7, ...] will send register i+4 while the others send i+0.

  // CHECK-DAG: [[IS_UPPER_HALF:%.*]] = and i32 [[TID]], 2
  // CHECK-DAG: [[IS_LOWER_HALF:%.*]] = icmp eq i32 [[IS_UPPER_HALF]], 0

  // For register [0, 4), the lane shuffle idx is essentially computed as
  // `(x//2*4 + x%2)%16 + (x>=16)*2`

  // CHECK-DAG: [[X_MOD_2:%.*]] = and i32 [[TID]], 1
  // CHECK-DAG: [[X_2_4_LOWER:%.*]] = shl {{.*}} i32 [[IS_UPPER_HALF]], 1
  // CHECK-DAG: [[X_2_4_UPPER0:%.*]] = shl i32 [[TID]], 1
  // CHECK-DAG: [[X_2_4_UPPER1:%.*]] = and i32 [[X_2_4_UPPER0]], 24
  // CHECK-DAG: [[X_GE_16:%.*]] = and i32 [[TID]], 16
  // CHECK-DAG: [[X_GE_16_2:%.*]] = lshr exact i32 [[X_GE_16]], 3

  // CHECK-DAG: [[IDX0:%.*]] = or disjoint i32 [[X_2_4_LOWER]], [[X_MOD_2]]
  // CHECK-DAG: [[IDX1:%.*]] = or disjoint i32 [[IDX0]], [[X_2_4_UPPER1]]
  // CHECK-DAG: [[IDX2:%.*]] = or disjoint i32 [[IDX1]], [[X_GE_16_2]]

  // CHECK-DAG: [[SHFLSRC0:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC0]], i32 [[SRC4]]
  // CHECK-DAG: [[SHFLSRC1:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC1]], i32 [[SRC5]]
  // CHECK-DAG: [[SHFLSRC2:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC2]], i32 [[SRC6]]
  // CHECK-DAG: [[SHFLSRC3:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC3]], i32 [[SRC7]]
  // CHECK-DAG: [[SHFLSRC4:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC4]], i32 [[SRC0]]
  // CHECK-DAG: [[SHFLSRC5:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC5]], i32 [[SRC1]]
  // CHECK-DAG: [[SHFLSRC6:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC6]], i32 [[SRC2]]
  // CHECK-DAG: [[SHFLSRC7:%.*]] = select i1 [[IS_LOWER_HALF]], i32 [[SRC7]], i32 [[SRC3]]

  // CHECK-DAG: [[SHFLOUT0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC0]], i32 [[IDX2]], i32 31)
  // CHECK-DAG: [[SHFLOUT1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC1]], i32 [[IDX2]], i32 31)
  // CHECK-DAG: [[SHFLOUT2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC2]], i32 [[IDX2]], i32 31)
  // CHECK-DAG: [[SHFLOUT3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC3]], i32 [[IDX2]], i32 31)

  // For register [4, 8), the upper and lower halves swap.

  // CHECK-DAG: [[IDX3:%.*]] = or disjoint i32 [[IDX1]], 2
  // CHECK-DAG: [[IDX4:%.*]] = xor i32 [[IDX3]], [[X_GE_16_2]]

  // CHECK-DAG: [[SHFLOUT4:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC4]], i32 [[IDX4]], i32 31)
  // CHECK-DAG: [[SHFLOUT5:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC5]], i32 [[IDX4]], i32 31)
  // CHECK-DAG: [[SHFLOUT6:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC6]], i32 [[IDX4]], i32 31)
  // CHECK-DAG: [[SHFLOUT7:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[SHFLSRC7]], i32 [[IDX4]], i32 31)

  // For lanes [16, 32), swap the two results.

  // CHECK-DAG: [[SWAP_RESULTS:%.*]] = icmp eq i32 [[X_GE_16]], 0

  // CHECK: [[DST0:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT0]], i32 [[SHFLOUT4]]
  // CHECK: [[DST1:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT1]], i32 [[SHFLOUT5]]
  // CHECK: [[DST2:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT2]], i32 [[SHFLOUT6]]
  // CHECK: [[DST3:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT3]], i32 [[SHFLOUT7]]
  // CHECK: [[DST4:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT4]], i32 [[SHFLOUT0]]
  // CHECK: [[DST5:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT5]], i32 [[SHFLOUT1]]
  // CHECK: [[DST6:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT6]], i32 [[SHFLOUT2]]
  // CHECK: [[DST7:%.*]] = select i1 [[SWAP_RESULTS]], i32 [[SHFLOUT7]], i32 [[SHFLOUT3]]

  // CHECK: insertvalue {{.*}}, i32 [[DST0]], 0
  // CHECK: insertvalue {{.*}}, i32 [[DST1]], 1
  // CHECK: insertvalue {{.*}}, i32 [[DST2]], 2
  // CHECK: insertvalue {{.*}}, i32 [[DST3]], 3
  // CHECK: insertvalue {{.*}}, i32 [[DST4]], 4
  // CHECK: insertvalue {{.*}}, i32 [[DST5]], 5
  // CHECK: insertvalue {{.*}}, i32 [[DST6]], 6
  // CHECK: insertvalue {{.*}}, i32 [[DST7]], 7

  %0 = ttg.convert_layout %arg0 : tensor<16x16xi32, #blocked0> -> tensor<16x16xi32, #blocked2>
  tt.return %0 : tensor<16x16xi32, #blocked2>
}

// CHECK-LABEL: convert_layout_blocked_blocked
tt.func private @convert_layout_blocked_blocked(%arg0: tensor<16x16xi32, #blocked0>) -> tensor<16x16xi32, #blocked1> {
  // This conversion looks like:
  //             dst_lane
  // dst_reg     0      1  ... 16     17  ...
  // 0          T0:0  T16:0    T1:0  T17:0
  // 1          T4:0  T20:0    T5:0  T21:0
  // 2          T8:0  T24:0    T9:0  T25:0
  // 3         T12:0  T28:0   T13:0  T29:0
  // 4          T2:0  T18:0    T3:0  T19:0
  // 5          T6:0  T22:0    T7:0  T23:0
  // 6         T10:0  T26:0   T11:0  T27:0
  // 7         T14:0  T30:0   T15:0  T31:0
  //
  // Where the registers change every 2 lanes like [0, 4, 1, 5, 2, 6, 3, 7] and
  // wraps around at lane 16. Due to this, there needs to be 8 selects per
  // shuffle input and output. The lane mapping also changes every register. Due
  // to this, we choose to fall back to the shared memory implementation.

  // CHECK-NOT: shfl.sync.idx
  // CHECK: st.shared

  %0 = ttg.convert_layout %arg0 : tensor<16x16xi32, #blocked0> -> tensor<16x16xi32, #blocked1>
  tt.return %0 : tensor<16x16xi32, #blocked1>
}

tt.func private @cvt_mma_to_dot_fp8(%a: tensor<128x64xi32, #mma>) -> tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> {
  %opA = ttg.convert_layout %a : tensor<128x64xi32, #mma> -> tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
  tt.return %opA : tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
}

tt.func @anchor(%ptr: !llvm.ptr, %arg0: tensor<16x16xi32, #blocked0>, %arg1: tensor<128x64xi32, #mma>) {
  %0 = tt.call @convert_layout_blocked_blocked(%arg0) : (tensor<16x16xi32, #blocked0>) -> tensor<16x16xi32, #blocked1>
  %1 = builtin.unrealized_conversion_cast %0 : tensor<16x16xi32, #blocked1> to !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
  llvm.store volatile %1, %ptr : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr

  %2 = tt.call @convert_layout_blocked_blocked_vec(%arg0) : (tensor<16x16xi32, #blocked0>) -> tensor<16x16xi32, #blocked2>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<16x16xi32, #blocked2> to !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
  llvm.store volatile %3, %ptr : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr

  tt.return
}

}
