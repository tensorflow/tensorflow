// RUN: emitters_opt %s --split-input-file \
// RUN:   --xla-lower-to-llvm-gpu="gpu_device_info='oneapi_compute_capability { architecture: \"bmg\"}'" \
// RUN: | FileCheck %s

module {
  // CHECK-LABEL: func @test_log1p
  func.func @test_log1p(%arg0: f32) -> f32 {
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
    // CHECK: %[[ADD:.*]] = llvm.fadd %[[ONE]], %arg0 : f32
    // CHECK: %[[LOG:.*]] = llvm.intr.log(%[[ADD]]) : (f32) -> f32
    // CHECK: return %[[LOG]] : f32
    %0 = math.log1p %arg0 : f32
    return %0 : f32
  }
}
