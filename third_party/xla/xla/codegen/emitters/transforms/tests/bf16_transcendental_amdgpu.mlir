// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx1250\"}'" \
// RUN:   | FileCheck %s

// gfx1250 has a native bf16 exp2 instruction (v_exp_bf16), reached via the
// llvm.amdgcn.exp2 intrinsic, instead of upcasting to f32 and calling
// __ocml_exp2_f32.
module {
  func.func @exp2_bf16(%arg0: bf16) -> bf16 {
    %0 = math.exp2 %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @exp2_bf16
// CHECK: llvm.call_intrinsic "llvm.amdgcn.exp2"({{.*}}) : (bf16) -> bf16
// CHECK-NOT: __ocml_exp2_f32

// -----

// A plain bf16 exp is rewritten to exp2(x * log2(e)) on gfx1250 so it also uses
// the native instruction.
module {
  func.func @exp_bf16(%arg0: bf16) -> bf16 {
    %0 = math.exp %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @exp_bf16
// CHECK: llvm.fmul
// CHECK: llvm.call_intrinsic "llvm.amdgcn.exp2"({{.*}}) : (bf16) -> bf16
