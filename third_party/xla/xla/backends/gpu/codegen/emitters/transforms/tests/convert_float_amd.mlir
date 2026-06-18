// RUN: emitters_opt %s -split-input-file -xla-gpu-convert-float-amd="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx942:sramecc+:xnack\"}'" -canonicalize | FileCheck %s

module {
  func.func @intr_f16_to_f8(%arg0: f16) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : f16 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : f16 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_f16_to_f8
// CHECK: arith.extf %{{.+}} : f16 to f32
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_bf16_to_f8(%arg0: bf16) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : bf16 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : bf16 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_bf16_to_f8
// CHECK: arith.extf %{{.+}} : bf16 to f32
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f32_to_f8(%arg0: f32) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : f32 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : f32 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_f32_to_f8
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f64_to_f8(%arg0: f64) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : f64 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : f64 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_f64_to_f8
// CHECK: arith.truncf %{{.+}} : f64 to f32
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: arith.truncf %{{.+}} : f64 to f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f8_to_f16(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (f16, f16) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to f16
    %b = arith.extf %arg1 : f8E5M2FNUZ to f16
    return %a, %b : f16, f16
  }
}

// CHECK-LABEL: @intr_f8_to_f16
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: llvm.amdgcn.cvt.f32.bf8
// CHECK: arith.truncf %{{.+}} : f32 to f16

// -----

module {
  func.func @intr_f8_to_bf16(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (bf16, bf16) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to bf16
    %b = arith.extf %arg1 : f8E5M2FNUZ to bf16
    return %a, %b : bf16, bf16
  }
}

// CHECK-LABEL: @intr_f8_to_bf16
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: llvm.amdgcn.cvt.f32.bf8
// CHECK: llvm.bitcast %{{.+}} : f32 to i32
// CHECK: llvm.bitcast %{{.+}} : i16 to bf16
// CHECK-NOT: arith.truncf %{{.+}} : f32 to bf16

// -----

module {
  func.func @intr_f8_to_f32(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (f32, f32) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to f32
    %b = arith.extf %arg1 : f8E5M2FNUZ to f32
    return %a, %b : f32, f32
  }
}

// CHECK-LABEL: @intr_f8_to_f32
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: llvm.amdgcn.cvt.f32.bf8


// -----

module {
  func.func @intr_f8_to_f64(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (f64, f64) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to f64
    %b = arith.extf %arg1 : f8E5M2FNUZ to f64
    return %a, %b : f64, f64
  }
}

// CHECK-LABEL: @intr_f8_to_f64
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: arith.extf %{{.+}} : f32 to f64
// CHECK: llvm.amdgcn.cvt.f32.bf8
// CHECK: arith.extf %{{.+}} : f32 to f64

// -----

module {
  func.func @intr_f16_to_4f8(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : f16 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : f16 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : f16 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : f16 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : f16 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : f16 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : f16 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : f16 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_f16_to_4f8
// CHECK-COUNT-4: arith.extf %{{.+}} : f16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-4: arith.extf %{{.+}} : f16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_bf16_to_4f8(%arg0: bf16, %arg1: bf16, %arg2: bf16, %arg3: bf16) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : bf16 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : bf16 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : bf16 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : bf16 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : bf16 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : bf16 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : bf16 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : bf16 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_bf16_to_4f8
// CHECK-COUNT-4: arith.extf %{{.+}} : bf16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-4: arith.extf %{{.+}} : bf16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f32_to_4f8(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : f32 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : f32 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : f32 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : f32 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : f32 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : f32 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : f32 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : f32 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_f32_to_4f8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f64_to_4f8(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : f64 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : f64 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : f64 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : f64 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : f64 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : f64 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : f64 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : f64 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_f64_to_4f8
// CHECK-COUNT-4: arith.truncf %{{.+}} : f64 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-4: arith.truncf %{{.+}} : f64 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_4f8_to_f16(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (f16, f16, f16, f16, f16, f16, f16, f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to f16
    %1 = arith.extf %a1 : f8E4M3FNUZ to f16
    %2 = arith.extf %a2 : f8E4M3FNUZ to f16
    %3 = arith.extf %a3 : f8E4M3FNUZ to f16
    %4 = arith.extf %b0 : f8E5M2FNUZ to f16
    %5 = arith.extf %b1 : f8E5M2FNUZ to f16
    %6 = arith.extf %b2 : f8E5M2FNUZ to f16
    %7 = arith.extf %b3 : f8E5M2FNUZ to f16
    return %0, %1, %2, %3, %4, %5, %6, %7 : f16, f16, f16, f16, f16, f16, f16, f16
  }
}

// CHECK-LABEL: @intr_4f8_to_f16
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pkrtz
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pkrtz

// -----

module {
  func.func @intr_4f8_to_bf16(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to bf16
    %1 = arith.extf %a1 : f8E4M3FNUZ to bf16
    %2 = arith.extf %a2 : f8E4M3FNUZ to bf16
    %3 = arith.extf %a3 : f8E4M3FNUZ to bf16
    %4 = arith.extf %b0 : f8E5M2FNUZ to bf16
    %5 = arith.extf %b1 : f8E5M2FNUZ to bf16
    %6 = arith.extf %b2 : f8E5M2FNUZ to bf16
    %7 = arith.extf %b3 : f8E5M2FNUZ to bf16
    return %0, %1, %2, %3, %4, %5, %6, %7 : bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16
  }
}

// CHECK-LABEL: @intr_4f8_to_bf16
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-8: llvm.bitcast
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8
// CHECK-COUNT-8: llvm.bitcast
// CHECK-NOT: arith.truncf %{{.+}} : f32 to bf16

// -----

module {
  func.func @intr_4f8_to_f32(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to f32
    %1 = arith.extf %a1 : f8E4M3FNUZ to f32
    %2 = arith.extf %a2 : f8E4M3FNUZ to f32
    %3 = arith.extf %a3 : f8E4M3FNUZ to f32
    %4 = arith.extf %b0 : f8E5M2FNUZ to f32
    %5 = arith.extf %b1 : f8E5M2FNUZ to f32
    %6 = arith.extf %b2 : f8E5M2FNUZ to f32
    %7 = arith.extf %b3 : f8E5M2FNUZ to f32
    return %0, %1, %2, %3, %4, %5, %6, %7 : f32, f32, f32, f32, f32, f32, f32, f32
  }
}

// CHECK-LABEL: @intr_4f8_to_f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8

// -----

module {
  func.func @intr_4f8_to_f64(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to f64
    %1 = arith.extf %a1 : f8E4M3FNUZ to f64
    %2 = arith.extf %a2 : f8E4M3FNUZ to f64
    %3 = arith.extf %a3 : f8E4M3FNUZ to f64
    %4 = arith.extf %b0 : f8E5M2FNUZ to f64
    %5 = arith.extf %b1 : f8E5M2FNUZ to f64
    %6 = arith.extf %b2 : f8E5M2FNUZ to f64
    %7 = arith.extf %b3 : f8E5M2FNUZ to f64
    return %0, %1, %2, %3, %4, %5, %6, %7 : f64, f64, f64, f64, f64, f64, f64, f64
  }
}

// CHECK-LABEL: @intr_4f8_to_f64
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-4: arith.extf %{{.+}} : f32 to f64
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8
// CHECK-COUNT-4: arith.extf %{{.+}} : f32 to f64