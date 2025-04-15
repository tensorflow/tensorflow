// RUN: emitters_opt %s -split-input-file -xla-expand-float-ops -canonicalize | FileCheck %s

module {
  func.func @tanh(%arg0: f32) -> f32 {
    %ret = math.tanh %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @tanh
// CHECK-NOT: tanh

// -----

module {
  func.func @erf(%arg0: f32) -> f32 {
    %ret = math.erf %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @erf
// CHECK-NOT: erf

// -----

module {
  func.func @maximumf(%arg0: f32, %arg1: f32) -> f32 {
    %ret = arith.maximumf %arg0, %arg1 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @maximumf
// CHECK: arith.maximumf

// -----

module {
  func.func @minimumf(%arg0: f32, %arg1: f32) -> f32 {
    %ret = arith.minimumf %arg0, %arg1 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @minimumf
// CHECK: arith.minimumf

// -----

module {
  func.func @minimumf64(%arg0: f64, %arg1: f64) -> f64 {
    %ret = arith.minimumf %arg0, %arg1 : f64
    return %ret : f64
  }
}

// CHECK-LABEL: @minimumf64
// CHECK: arith.minimumf

// -----

module {
  func.func @cmpif8(%arg0: f8E5M2, %arg1: f8E5M2) -> i1 {
    %ret = arith.cmpf une, %arg0, %arg1 : f8E5M2
    return %ret : i1
  }
}

// Just check that this lowers successfully. We have integration tests to verify
// correctness.
// CHECK-LABEL: @cmpif8
// CHECK-NOT: arith.cmpf une{{.*}}f8E5M2

// -----

module {
  func.func @fptoi8(%arg0: f8E5M2) -> i32 {
    %ret = arith.fptosi %arg0 : f8E5M2 to i32
    return %ret : i32
  }
}

// Just check that this lowers successfully. We have integration tests to verify
// correctness.
// CHECK-LABEL: @fptoi8
// CHECK-NOT: arith.fptosi {{.*}}f8E5M2

// -----

module {
  func.func @double_to_f8(%arg0: f64) -> f8E5M2 {
    %ret = arith.truncf %arg0 : f64 to f8E5M2
    return %ret : f8E5M2
  }
}

// Just check that this lowers successfully. We have integration tests to verify
// correctness.
// CHECK-LABEL: @double_to_f8
// CHECK-NOT: arith.truncf

// -----

module {
  func.func @bf16_to_f8(%arg0: bf16) -> f8E5M2 {
    %ret = arith.truncf %arg0 : bf16 to f8E5M2
    return %ret : f8E5M2
  }
}

// Verify that we go through f32/f16. We have integration tests to verify
// correctness.
// CHECK-LABEL: @bf16_to_f8
// CHECK: %[[EXT:.*]] = arith.extf {{.*}} : bf16 to f32
// CHECK: arith.truncf %[[EXT]] : f32 to f16
// CHECK-NOT: arith.truncf

// -----

module {
  func.func @f4_to_f16(%arg0: f4E2M1FN) -> f16 {
    %ret = arith.extf %arg0 : f4E2M1FN to f16
    return %ret : f16
  }
}

// CHECK-LABEL: @f4_to_f16
// CHECK-NOT: arith.extf

// -----

module {
  func.func @f16_to_f4(%arg0: f16) -> f4E2M1FN {
    %ret = arith.truncf %arg0 : f16 to f4E2M1FN
    return %ret : f4E2M1FN
  }
}

// CHECK-LABEL: @f16_to_f4
// CHECK-NOT: arith.truncf

// -----

module {
  func.func @f4_abs(%arg0: f4E2M1FN) -> f4E2M1FN {
    %ret = math.absf %arg0 : f4E2M1FN
    return %ret : f4E2M1FN
  }
}

// CHECK-LABEL: @f4_abs
// CHECK-NOT: math.absf
// CHECK: arith.constant 7 : i4

// -----

module {
  func.func @e8m0_abs(%arg0: f8E8M0FNU) -> f8E8M0FNU {
    %ret = math.absf %arg0 : f8E8M0FNU
    return %ret : f8E8M0FNU
  }
}

// CHECK-LABEL: @e8m0_abs
// CHECK-NOT: math.absf
// CHECK: return %arg0
