// RUN: emitters_opt %s --split-input-file \
// RUN:   --xla-gpu-test-to-llvm="gpu_device_info='oneapi_compute_capability { architecture: \"bmg\"}'" \
// RUN: | FileCheck %s

module {
  // CHECK-LABEL: func @test_log1p
  func.func @test_log1p(%arg0: f32) -> f32 {
    // CHECK-NOT: llvm.mlir.constant(1.000000e+00 : f32) : f32
    // CHECK-NOT: llvm.fadd %{{.*}}, %arg0 : f32
    // CHECK-NOT: llvm.intr.log
    // CHECK: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_log1pf(%arg0)
    // CHECK: return %{{.*}} : f32
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

// CHECK-DAG: spir_funccc @_Z{{.*}}__spirv_ocl_expm1f
// CHECK-DAG: spir_funccc @_Z{{.*}}__spirv_ocl_sinf
// CHECK-DAG: spir_funccc @_Z{{.*}}__spirv_ocl_cosf
// CHECK-LABEL: @test_tan
// CHECK-SAME: %[[ARG0:.*]]: !llvm.struct<(f32, f32)>
// CHECK: %[[V0:.*]] = llvm.extractvalue %[[ARG0]][0]
// CHECK: %[[V1:.*]] = llvm.extractvalue %[[ARG0]][1]
// CHECK-NOT: llvm.intr.exp
// CHECK-NOT: llvm.intr.cos
// CHECK-NOT: llvm.intr.sin
// CHECK: %[[E0:.*]] = llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1f
// CHECK: %[[E1:.*]] = llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_expm1f
// CHECK: llvm.fsub %[[E0]], %[[E1]] : f32
// CHECK-DAG: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_cosf(%[[V0]])
// CHECK-DAG: llvm.call spir_funccc @_Z{{.*}}__spirv_ocl_sinf(%[[V0]])
// CHECK: llvm.return %{{.*}} : !llvm.struct<(f32, f32)>
