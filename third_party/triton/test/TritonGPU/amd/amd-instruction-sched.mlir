// RUN: triton-opt %s -triton-amdgpu-insert-instruction-sched-hints='variant=llvm_iglp_0' -triton-amdgpu-lower-insert-instruction-sched-hints -verify-diagnostics | FileCheck %s -check-prefix=INSERT_IGLP0
// RUN: triton-opt %s -triton-amdgpu-insert-instruction-sched-hints='variant=llvm_iglp_1' -triton-amdgpu-lower-insert-instruction-sched-hints -verify-diagnostics | FileCheck %s -check-prefix=INSERT_IGLP1
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=hip:gfx942 num-ctas=1 num-warps=4 threads-per-warp=64' -tritongpu-coalesce -tritonamdgpu-accelerate-matmul='arch-generation-name=gfx942 matrix-instruction-size=32 kPack=1' -tritongpu-remove-layout-conversions -tritonamdgpu-stream-pipeline='num_stages=1' -triton-amdgpu-insert-instruction-sched-hints='variant=local_prefetch' -tritongpu-reduce-data-duplication -optimize-amd-lds-usage='target-arch=gfx942' -convert-scf-to-cf -convert-index-to-llvm -allocate-shared-memory -convert-triton-amdgpu-to-llvm='arch=gfx942' -verify-diagnostics | FileCheck %s -check-prefix=INSTR_COUNT_NS1
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=hip:gfx942 num-ctas=1 num-warps=4 threads-per-warp=64' -tritongpu-coalesce -tritonamdgpu-accelerate-matmul='arch-generation-name=gfx942 matrix-instruction-size=32 kPack=1' -tritongpu-remove-layout-conversions -tritonamdgpu-stream-pipeline='num_stages=2' -triton-amdgpu-insert-instruction-sched-hints='variant=local_prefetch' -tritongpu-reduce-data-duplication -optimize-amd-lds-usage='target-arch=gfx942' -convert-scf-to-cf -convert-index-to-llvm -allocate-shared-memory -convert-triton-amdgpu-to-llvm='arch=gfx942' -verify-diagnostics | FileCheck %s -check-prefix=INSTR_COUNT_NS2
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=hip:gfx942 num-ctas=1 num-warps=4 threads-per-warp=64' -tritongpu-coalesce -tritonamdgpu-accelerate-matmul='arch-generation-name=gfx942 matrix-instruction-size=16 kPack=1' -tritongpu-remove-layout-conversions -tritonamdgpu-stream-pipeline='num_stages=2' -triton-amdgpu-insert-instruction-sched-hints='variant=local_prefetch' -tritongpu-reduce-data-duplication -optimize-amd-lds-usage='target-arch=gfx942' -convert-scf-to-cf -convert-index-to-llvm -allocate-shared-memory -convert-triton-amdgpu-to-llvm='arch=gfx942' -triton-amdgpu-lower-insert-instruction-sched-hints='arch=gfx942 num_stages=2' -debug-only='lower-insert-instruction-sched-hints' -verify-diagnostics 2>&1 | FileCheck %s -check-prefix=USE_LOCAL_PREFETCH_GLOBAL_LOAD
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=hip:gfx942 num-ctas=1 num-warps=4 threads-per-warp=64' -tritongpu-coalesce -tritongpu-remove-layout-conversions -tritonamdgpu-stream-pipeline='num_stages=1' | FileCheck %s -check-prefix=LABELING_PS_1
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=hip:gfx942 num-ctas=1 num-warps=4 threads-per-warp=64' -tritongpu-coalesce -tritongpu-remove-layout-conversions -tritonamdgpu-stream-pipeline='num_stages=2' | FileCheck %s -check-prefix=LABELING_PS_2

module {
  // INSERT_IGLP0-LABEL: @test_dot_op
  // INSERT_IGLP1-LABEL: @test_dot_op
  // INSTR_COUNT_NS1-LABEL: @test_dot_op
  // INSTR_COUNT_NS2-LABEL: @test_dot_op
  // USE_LOCAL_PREFETCH_GLOBAL_LOAD: @test_dot_op
  // LABELING_PS_1-LABEL: @test_dot_op
  // LABELING_PS_2-LABEL: @test_dot_op
  tt.func @test_dot_op(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %C : !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32> -> tensor<128x32xi32>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi32>
  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32> -> tensor<32x128xi32>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32>

  %a_off = arith.constant dense<4> : tensor<128x32xi32>
  %b_off = arith.constant dense<4> : tensor<32x128xi32>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>, tensor<128x128xf32>) {
    %a = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>>
    %b = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>>

    // INSERT_IGLP0: rocdl.iglp.opt 0
    // INSERT_IGLP1: rocdl.iglp.opt 1

    // INSTR_COUNT_NS1: amdgpu.instruction_sched_hint
    // INSTR_COUNT_NS1-SAME: isBufferLoadsAEnabled = false
    // INSTR_COUNT_NS1-SAME: isBufferLoadsBEnabled = false
    // INSTR_COUNT_NS1-SAME: numDsReadsA = #amdgpu.InstCounter<8, vector<4xf16>>
    // INSTR_COUNT_NS1-SAME: numDsReadsB = #amdgpu.InstCounter<32, vector<1xf16>>
    // INSTR_COUNT_NS1-SAME: numDsWritesA = #amdgpu.InstCounter<0, none>
    // INSTR_COUNT_NS1-SAME: numDsWritesB = #amdgpu.InstCounter<0, none>
    // INSTR_COUNT_NS1-SAME: numGlobalLoadsA = #amdgpu.InstCounter<4, vector<4xf16>>
    // INSTR_COUNT_NS1-SAME: numGlobalLoadsB = #amdgpu.InstCounter<4, vector<4xf16>>
    // INSTR_COUNT_NS1-SAME: numMMAs = #amdgpu.InstCounter<16, tensor<32x32x8xf16>>

    // INSTR_COUNT_NS2: amdgpu.instruction_sched_hint
    // INSTR_COUNT_NS2-SAME: isBufferLoadsAEnabled = false
    // INSTR_COUNT_NS2-SAME: isBufferLoadsBEnabled = false
    // INSTR_COUNT_NS2-SAME: numDsReadsA = #amdgpu.InstCounter<8, vector<4xf16>>
    // INSTR_COUNT_NS2-SAME: numDsReadsB = #amdgpu.InstCounter<32, vector<1xf16>>
    // INSTR_COUNT_NS2-SAME: numDsWritesA = #amdgpu.InstCounter<4, vector<4xf16>>
    // INSTR_COUNT_NS2-SAME: numDsWritesB = #amdgpu.InstCounter<4, vector<4xf16>>
    // INSTR_COUNT_NS2-SAME: numGlobalLoadsA = #amdgpu.InstCounter<4, vector<4xf16>>
    // INSTR_COUNT_NS2-SAME: numGlobalLoadsB = #amdgpu.InstCounter<4, vector<4xf16>>
    // INSTR_COUNT_NS2-SAME: numMMAs = #amdgpu.InstCounter<16, tensor<32x32x8xf16>>

    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.barrier [[SCHED_GUARD:.+]]
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE:512]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA:8]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ:32]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_WRITE]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[VMEM_READ]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ:256]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[DS_READ]], 2, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.group.barrier [[MFMA]], 1, 0
    // USE_LOCAL_PREFETCH_GLOBAL_LOAD: rocdl.sched.barrier [[SCHED_GUARD]]


    // LABELING_PS_1: scf.for
    // LABELING_PS_1: %[[REG0_OP0:.+]] = tt.load {{.*}} {OpIdx = #amdgpu.OpIdx<0>}
    // LABELING_PS_1: %[[REG0_OP1:.+]] = tt.load {{.*}} {OpIdx = #amdgpu.OpIdx<1>}
    // LABELING_PS_1: %[[REG1_OP0:.+]] = ttg.convert_layout %[[REG0_OP0]]
    // LABELING_PS_1: %[[REG1_OP1:.+]] = ttg.convert_layout %[[REG0_OP1]]
    // LABELING_PS_1: tt.dot %[[REG1_OP0]], %[[REG1_OP1]], {{.*}}

    // LABELING_PS_2: scf.for
    // LABELING_PS_2: %[[REG0_OP0:.+]] = tt.load {{.*}} {OpIdx = #amdgpu.OpIdx<0>}
    // LABELING_PS_2: %[[REG0_OP1:.+]] = tt.load {{.*}} {OpIdx = #amdgpu.OpIdx<1>}
    // LABELING_PS_2: ttg.local_store %[[REG0_OP0]], %{{.*}} {OpIdx = #amdgpu.OpIdx<0>}
    // LABELING_PS_2: ttg.local_store %[[REG0_OP1]], %{{.*}} {OpIdx = #amdgpu.OpIdx<1>}

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi32>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>, tensor<128x128xf32>
  }

  // C ptrs
  %c_ptr_splat = tt.splat %C : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  %c_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32>
  %c_tmp1 = tt.expand_dims %c_tmp0 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %c_offs = tt.broadcast %c_tmp1 : tensor<1x128xi32> -> tensor<128x128xi32>
  %c_ptr = tt.addptr %c_ptr_splat, %c_offs : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>

  tt.store %c_ptr, %loop#2 : tensor<128x128x!tt.ptr<f32>>
  tt.return
}
}
