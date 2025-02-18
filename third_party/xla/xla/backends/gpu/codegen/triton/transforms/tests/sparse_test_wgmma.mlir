// RUN: xla-opt %s --sparse-wgmma-to-llvm | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @wgmma_sp(%descA: i64, %metaA: i32, %descB: i64, %acc: !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>) {
    // CHECK: @wgmma_sp(%[[LHS:.*]]: i64, %[[META:.*]]: i32, %[[RHS:.*]]: i64,
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = []
    // CHECK-SAME: "wgmma.mma_async.sp.sync.aligned.m64n16k32.f32.bf16.bf16 {$0,$1,$2,$3,$4,$5,$6,$7}, $16, $17, $18, 0, 1, 1, 1, 0, 0;"
    // CHECK-SAME: "=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,l,l,r" %0, %1, %2, %3, %4, %5, %6, %7, %[[LHS]], %[[RHS]], %[[META]]
    %acc0 = nvgpu.wgmma_sp %descA meta %metaA, %descB, %acc
    {eltTypeA = 5 : i32, eltTypeB = 5 : i32, eltTypeC = 7 : i32, layoutA = 0 : i32, layoutB = 1 : i32, m = 64 : i32, n = 16 : i32, k = 32 : i32} :
    (i64, i32, i64, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    tt.return
  }
}
