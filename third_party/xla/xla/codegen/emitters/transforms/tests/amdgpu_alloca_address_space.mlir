// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx90a:sramecc+:xnack\"}'" \
// RUN:   | FileCheck %s --check-prefix=CHECK-GFX90A

module {
  // Provide explicit allocas in address space 0 so the pass-level utility can
  // rewrite them to the AMDGPU private address space (5).
  llvm.func @amdgpu_alloca_address_space() {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %alloc_struct = llvm.alloca %c1 x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr
    %alloc_i64 = llvm.alloca %c1 x i64 : (i32) -> !llvm.ptr
    llvm.return
  }
}

// CHECK-GFX90A-LABEL: llvm.func @amdgpu_alloca_address_space()
// CHECK-GFX90A: llvm.alloca %{{.*}} x !llvm.struct<(f32, f32)> : (i32) -> !llvm.ptr<5>
// CHECK-GFX90A: llvm.alloca %{{.*}} x i64 : (i32) -> !llvm.ptr<5>

// -----

