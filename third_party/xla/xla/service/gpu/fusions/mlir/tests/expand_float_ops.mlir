// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-float-ops="pre-ampere=true" -canonicalize | FileCheck %s -check-prefixes=CHECK,CHECK-PRE-AMPERE
// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-float-ops="pre-ampere=false" -canonicalize | FileCheck %s -check-prefixes=CHECK,CHECK-AMPERE

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
// CHECK-AMPERE: arith.maximumf
// CHECK-PRE-AMPERE: arith.cmpf
// CHECK-PRE-AMPERE: arith.select

// -----

module {
  func.func @minimumf(%arg0: f32, %arg1: f32) -> f32 {
    %ret = arith.minimumf %arg0, %arg1 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @minimumf
// CHECK-AMPERE: arith.minimumf
// CHECK-PRE-AMPERE: arith.cmpf
// CHECK-PRE-AMPERE: arith.select

// -----

module {
  func.func @minimumf64(%arg0: f64, %arg1: f64) -> f64 {
    %ret = arith.minimumf %arg0, %arg1 : f64
    return %ret : f64
  }
}

// CHECK-LABEL: @minimumf64
// CHECK-NOT: minimumf
// CHECK: arith.cmpf
// CHECK: arith.select