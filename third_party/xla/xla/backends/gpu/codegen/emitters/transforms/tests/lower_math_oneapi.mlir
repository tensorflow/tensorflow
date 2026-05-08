// RUN: emitters_opt %s --split-input-file \
// RUN:   --xla-gpu-test-to-llvm="gpu_device_info='oneapi_compute_capability { architecture: \"bmg\"}'" \
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

// -----

module {
  func.func @test_tan(%arg0: complex<f32>) -> complex<f32> {
    %0 = complex.tan %arg0 : complex<f32>
    func.return %0 : complex<f32>
  }
}

// CHECK-LABEL: @test_tan
// CHECK-SAME: %[[ARG0:.*]]: !llvm.struct<(f32, f32)>
// CHECK: %[[V0:.*]] = llvm.extractvalue %[[ARG0]][0]
// CHECK: %[[V1:.*]] = llvm.extractvalue %[[ARG0]][1]
// CHECK: %[[E0:.*]] = llvm.intr.exp
// CHECK: %[[EM0:.*]] = llvm.fsub %[[E0]], %{{.*}}
// CHECK: %[[E1:.*]] = llvm.intr.exp
// CHECK: %[[EM1:.*]] = llvm.fsub %[[E1]], %{{.*}}
// CHECK: llvm.fsub %[[EM0]], %[[EM1]] : f32
// CHECK-DAG: llvm.intr.cos(%[[V0]])
// CHECK-DAG: llvm.intr.sin(%[[V0]])
// CHECK: llvm.return %{{.*}} : !llvm.struct<(f32, f32)>
