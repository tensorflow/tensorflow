// RUN: emitters_opt %s -split-input-file -canonicalize -xla-gpu-recover-exp2 \
// RUN:    | FileCheck %s

///////////////////////////////////////////////////////////////////////////////
// tests for recovery in vanilla situations for different supported datatypes

module {
  func.func @expect_recover_f64_as_f64(%arg0: f64) -> f64 {
    %cst = arith.constant 0.6931471805599453 : f64
    %4 = arith.mulf %arg0, %cst : f64
    %5 = math.exp %4 : f64
    return %5 : f64
  }
}

// CHECK-LABEL: @expect_recover_f64_as_f64
// CHECK-NOT: arith.constant {{.+}} : f64
// CHECK-NOT: arith.mulf %{{.+}}, %{{.+}} : f64
// CHECK-NOT: math.exp %{{.+}} : f64
// CHECK: math.exp2 %arg0 : f64

// -----

module {
  func.func @expect_recover_f32_as_f32(%arg0: f32) -> f32 {
    %cst = arith.constant 0.6931472 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @expect_recover_f32_as_f32
// CHECK-NOT: arith.constant {{.+}} : f32
// CHECK-NOT: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK-NOT: math.exp %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32

// -----

module {
  func.func @expect_recover_bf16_as_f32(%arg0: f32) -> f32 {
    %cst = arith.constant 0.69140625 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @expect_recover_bf16_as_f32
// CHECK-NOT: arith.constant {{.+}} : f32
// CHECK-NOT: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK-NOT: math.exp %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32

// -----
// fp16 has the same log(2) representation as tf32, so no difference
module {
  func.func @expect_recover_f16tf32_as_f32(%arg0: f32) -> f32 {
    %cst = arith.constant 0.693359375 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @expect_recover_f16tf32_as_f32
// CHECK-NOT: arith.constant {{.+}} : f32
// CHECK-NOT: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK-NOT: math.exp %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32

///////////////////////////////////////////////////////////////////////////////
// tests for recovery when constant is +/- 1ulp from exact value.
// Doing that for fp32 only, as other types shouldn't be different (it'd be much
// simpler to cover everything if these test files could be generated)

module {
  func.func @expect_recover_f32_as_f32_p1ulp(%arg0: f32) -> f32 {
    %cst = arith.constant 0.69314724 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @expect_recover_f32_as_f32_p1ulp
// CHECK-NOT: arith.constant {{.+}} : f32
// CHECK-NOT: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK-NOT: math.exp %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32

// -----

module {
  func.func @expect_recover_f32_as_f32_m1ulp(%arg0: f32) -> f32 {
    %cst = arith.constant 0.6931471 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @expect_recover_f32_as_f32_m1ulp
// CHECK-NOT: arith.constant {{.+}} : f32
// CHECK-NOT: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK-NOT: math.exp %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32

// -----
// +/- 2 ulp shouldn't be recovered
module {
  func.func @no_recover_f32_as_f32_p2ulp(%arg0: f32) -> f32 {
    %cst = arith.constant 0.693147302 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @no_recover_f32_as_f32_p2ulp
// CHECK: arith.constant {{.+}} : f32
// CHECK: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK: math.exp %{{.+}} : f32
// CHECK-NOT: math.exp2 %{{.+}} : f32

// -----

module {
  func.func @no_recover_f32_as_f32_m2ulp(%arg0: f32) -> f32 {
    %cst = arith.constant 0.69314706 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @no_recover_f32_as_f32_m2ulp
// CHECK: arith.constant {{.+}} : f32
// CHECK: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK: math.exp %{{.+}} : f32
// CHECK-NOT: math.exp2 %{{.+}} : f32

///////////////////////////////////////////////////////////////////////////////
// more elaborate cases to check dependency handling

module {
  func.func @no_recover_f32_src_add(%arg0: f32) -> f32 {
    %cst = arith.constant 0.6931472 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %2 = arith.addf %4, %cst : f32
    %5 = math.exp %2 : f32
    return %5 : f32
  }
}

// CHECK-LABEL: @no_recover_f32_src_add
// CHECK: arith.constant {{.+}} : f32
// CHECK: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK: math.exp %{{.+}} : f32
// CHECK-NOT: math.exp2 %{{.+}} : f32

// -----
// recovery, but mul is used elsewhere
module {
  func.func @expect_recover_mul_used(%arg0: f32) -> f32 {
    %cst = arith.constant 0.6931472 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    %6 = arith.addf %5, %4 : f32
    return %6 : f32
  }
}

// CHECK-LABEL: @expect_recover_mul_used
// CHECK: arith.constant {{.+}} : f32
// CHECK: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32
// CHECK: arith.addf %{{.+}}, %{{.+}} : f32

// -----
// recovery, but const is used elsewhere
module {
  func.func @expect_recover_const_used(%arg0: f32) -> f32 {
    %cst = arith.constant 0.6931472 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %5 = math.exp %4 : f32
    %6 = arith.addf %5, %cst : f32
    return %6 : f32
  }
}

// CHECK-LABEL: @expect_recover_const_used
// CHECK: arith.constant {{.+}} : f32
// CHECK-NOT: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32
// CHECK: arith.addf %{{.+}}, %{{.+}} : f32

// -----
// recovery, but both ops are used elsewhere
module {
  func.func @expect_recover_both_used(%arg0: f32) -> f32 {
    %cst = arith.constant 0.6931472 : f32
    %4 = arith.mulf %arg0, %cst : f32
    %6 = arith.addf %4, %cst : f32
    %5 = math.exp %4 : f32
    %7 = arith.addf %5, %6 : f32
    return %7 : f32
  }
}

// CHECK-LABEL: @expect_recover_both_used
// CHECK: arith.constant {{.+}} : f32
// CHECK: arith.mulf %{{.+}}, %{{.+}} : f32
// CHECK: arith.addf %{{.+}}, %{{.+}} : f32
// CHECK: math.exp2 %arg0 : f32
// CHECK: arith.addf %{{.+}}, %{{.+}} : f32
