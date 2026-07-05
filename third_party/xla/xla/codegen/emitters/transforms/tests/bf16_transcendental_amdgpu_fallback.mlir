// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx942\"}'" \
// RUN:   | FileCheck %s

// On architectures without native bf16 transcendentals (e.g. gfx942), bf16
// exp2 is upcast to f32 and lowered to __ocml_exp2_f32 rather than the gfx1250
// v_exp_bf16 / llvm.amdgcn.exp2 path.
module {
  func.func @exp2_bf16(%arg0: bf16) -> bf16 {
    %0 = math.exp2 %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @exp2_bf16
// CHECK: llvm.call @__ocml_exp2_f32
// CHECK-NOT: llvm.amdgcn.exp2
