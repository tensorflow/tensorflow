// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tc_gen5_mma
  // CHECK: %[[WID:.+]] = nvgpu.warp_id
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[P0:.+]] = llvm.icmp "eq" %[[WID]], %[[C0]] : i32
  // CHECK: %[[P1:.+]] = llvm.and %{{.*}}, %[[P0]]  : i1
  // CHECK: llvm.cond_br %[[P1]]
  // CHECK: %[[E:.+]] = nvvm.elect.sync -> i1
  // CHECK-COUNT-8: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[E]]
  //         CHECK: @$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %[[E]]
  tt.func @tc_gen5_mma(%a: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier :
     (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       i1, i1,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_multi_m_n
  // CHECK-DAG: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK-DAG: %[[T0:.+]] = llvm.add %[[TMEM_BASE]], %[[C0]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T0]]
  // CHECK: %[[T1:.+]] = llvm.add %[[TMEM_BASE]], %[[C64]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T1]]
  // 1048576 = row << 16 + col = 16 << 16 + 0
  // CHECK: %[[C1048576:.+]] = llvm.mlir.constant(1048576 : i32) : i32
  // CHECK: %[[T2:.+]] = llvm.add %[[TMEM_BASE]], %[[C1048576]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T2]]
  // 1048640 = row << 16 + col = 16 << 16 + 64
  // CHECK: %[[C1048640:.+]] = llvm.mlir.constant(1048640 : i32) : i32
  // CHECK: %[[T3:.+]] = llvm.add %[[TMEM_BASE]], %[[C1048640]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T3]]

  tt.func @tc_gen5_mma_multi_m_n(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier :
     (!ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       i1, i1,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 2], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CTAsPerCGA = [1, 2], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CTAsPerCGA = [1, 2], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [2], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, unpacked = true, CTASplitN = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_multi_ctas
  // CHECK-DAG: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C32:.+]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG: %[[T0:.+]] = llvm.add %[[TMEM_BASE]], %[[C0]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T0]]
  // CHECK: %[[T1:.+]] = llvm.add %[[TMEM_BASE]], %[[C32]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T1]]
  // 1048576 = row << 16 + col = 16 << 16 + 0
  // CHECK: %[[C1048576:.+]] = llvm.mlir.constant(1048576 : i32) : i32
  // CHECK: %[[T2:.+]] = llvm.add %[[TMEM_BASE]], %[[C1048576]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T2]]
  // 1048640 = row << 16 + col = 16 << 16 + 32
  // CHECK: %[[C1048608:.+]] = llvm.mlir.constant(1048608 : i32) : i32
  // CHECK: %[[T3:.+]] = llvm.add %[[TMEM_BASE]], %[[C1048608]] : i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[T3]]

  tt.func @tc_gen5_mma_multi_ctas(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier :
     (!ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       i1, i1,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.32x32b.x128.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100, $101, $102, $103, $104, $105, $106, $107, $108, $109, $110, $111, $112, $113, $114, $115, $116, $117, $118, $119, $120, $121, $122, $123, $124, $125, $126, $127}, [$128];", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %{{.*}} : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_allocation
  // CHECK: llvm.mlir.constant(4194306 : i32) : i32
  tt.func public @tensor_memory_allocation() {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 2 : i32, tensor_memory_row_offset = 64 : i32} : () -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_m64
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.st.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: tcgen05.ld.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.ld.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld_m64(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_unpack_f16
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63}, [$64];", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %{{.*}} : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_unpack_f16() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16, #blocked1>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale
  // CHECK-SAME: (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[USE_ACC:.+]]: i1, %{{.*}}: i1, %{{.*}})
  // CHECK-DAG: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C32:.+]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[T0:.+]] = llvm.add %[[TMEM_BASE]], %[[C0]] : i32
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(144774144 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[T0]], %{{.+}}, %{{.+}}, %[[DESC0]], %{{.+}}, %{{.+}}, %[[USE_ACC]]
  // CHECK: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1
  // CHECK: %[[DESC1:.+]] = llvm.mlir.constant(681645072 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[T0]], %{{.+}}, %{{.+}}, %[[DESC1]], %{{.+}}, %{{.+}}, %[[TRUE]]
  tt.func @tc_gen5_mma_block_scale(%a: !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<32x128xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e4m3 rhs = e2m1, %barrier :
    (!ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<32x128xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    i1,
    i1,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_fp4_a
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(144769664 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC0]]
  // CHECK: %[[DESC1:.+]] = llvm.mlir.constant(681640592 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC1]]
  // CHECK: %[[DESC2:.+]] = llvm.mlir.constant(1218511520 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC2]]
  // CHECK: %[[DESC3:.+]] = llvm.mlir.constant(1755382448 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC3]]
  tt.func @tc_gen5_mma_block_scale_fp4_a(%a: !ttg.memdesc<128x64xi8, #shared1, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e4m3, %barrier :
    (!ttg.memdesc<128x64xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    i1,
    i1,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 2], CTASplitNum = [1, 2], CTAOrder = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [2], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true, CTASplitM = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @tc_gen5_mma_2ctas(%a: !ttg.memdesc<256x32xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) {
    // CHECK: tcgen05.mma.cta_group::2.kind::f16
    // CHECK: tcgen05.mma.cta_group::2.kind::f16
    // CHECK: tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier {two_ctas} :
     (!ttg.memdesc<256x32xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       i1, i1,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread=[1, 4], threadsPerWarp=[32, 1], warpsPerCTA=[4, 1], order=[0, 1]}>
#shared = #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=1, order=[1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_copy_2d(%src: !ttg.memdesc<256x16xi8, #shared, #ttg.shared_memory>,
                               %dst: !ttg.memdesc<128x32xi32, #tmem, #ttng.tensor_memory, mutable>,
			       %barrier: !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory>) {
    // CHECK-COUNT-8: tcgen05.cp.cta_group::1.warpx4.32x128b
    // CHECK: tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64
    ttng.tmem_copy %src, %dst, %barrier : (!ttg.memdesc<256x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<128x32xi32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory>) -> ()

    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_nvfp4
  // CHECK-DAG: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[T0:.+]] = llvm.add %[[TMEM_BASE]], %[[C0]] : i32
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(138413184 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[T0]], %{{.+}}, %{{.+}}, %[[DESC0]]
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[T0]], %{{.+}}, %{{.+}}, %[[DESC0]]
  tt.func @tc_gen5_mma_block_scale_nvfp4(%a: !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %barrier :
    (!ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
    i1,
    i1,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_mxfp4
  // CHECK-DAG: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[T0:.+]] = llvm.add %[[TMEM_BASE]], %[[C0]] : i32
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(146801792 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[T0]], %{{.+}}, %{{.+}}, %[[DESC0]]
  // CHECK: %[[DESC1:.+]] = llvm.mlir.constant(1220543648 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[T0]], %{{.+}}, %{{.+}}, %[[DESC1]]
  tt.func @tc_gen5_mma_block_scale_mxfp4(%a: !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %barrier :
    (!ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>,
    i1,
    i1,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_128x256
  // CHECK: tcgen05.st.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.st.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.st.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.st.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x64.b32
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld_128x256(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x256xf32, #blocked1>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_128x256_8_warps
  // CHECK: tcgen05.st.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld_128x256_8_warps(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x256xf32, #blocked1>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked1>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, unpacked = false>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @tc_gen5_mma_lhs_tmem(%arg0: !ttg.memdesc<128x32xf16, #tmem, #ttng.tensor_memory>, %arg1: !ttg.memdesc<32x128xf16, #shared, #smem>, %arg2: !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg3: i1, %arg4: i1, %arg5: !ttg.memdesc<1xi64, #shared1, #smem>) {
    // CHECK-LABEL: tc_gen5_mma_lhs_tmem
    //       CHECK: tcgen05.mma.cta_group::1.kind::f16
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : (
      !ttg.memdesc<128x32xf16, #tmem, #ttng.tensor_memory>,
      !ttg.memdesc<32x128xf16, #shared, #smem>,
      !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>,
      i1, i1, !ttg.memdesc<1xi64, #shared1, #smem>) -> ()
    tt.return
  }
}
