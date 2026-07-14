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

// -----

// sqrt has a generic bf16 lowering (llvm.intr.sqrt) on gfx942, so it does not
// use the native gfx1250 llvm.amdgcn.sqrt path.
module {
  func.func @sqrt_bf16(%arg0: bf16) -> bf16 {
    %0 = math.sqrt %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @sqrt_bf16
// CHECK: llvm.intr.sqrt
// CHECK-NOT: llvm.amdgcn.sqrt

// -----

// rsqrt is upcast to f32 and lowered to __ocml_rsqrt_f32 on gfx942.
module {
  func.func @rsqrt_bf16(%arg0: bf16) -> bf16 {
    %0 = math.rsqrt %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @rsqrt_bf16
// CHECK: llvm.call @__ocml_rsqrt_f32
// CHECK-NOT: llvm.amdgcn.rsq

// -----

// tanh is upcast to f32 and lowered to __ocml_tanh_f32 on gfx942.
module {
  func.func @tanh_bf16(%arg0: bf16) -> bf16 {
    %0 = math.tanh %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @tanh_bf16
// CHECK: llvm.call @__ocml_tanh_f32
// CHECK-NOT: llvm.amdgcn.tanh

// -----

// log2 is upcast to f32 and lowered to __ocml_log2_f32 on gfx942.
module {
  func.func @log2_bf16(%arg0: bf16) -> bf16 {
    %0 = math.log2 %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: llvm.func @log2_bf16
// CHECK: llvm.call @__ocml_log2_f32
// CHECK-NOT: llvm.amdgcn.log
