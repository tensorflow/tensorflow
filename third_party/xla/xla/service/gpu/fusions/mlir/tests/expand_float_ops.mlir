// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-float-ops="pre-ampere=true" -canonicalize | FileCheck %s -check-prefixes=CHECK,CHECK-PRE-AMPERE
// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-expand-float-ops="pre-ampere=false" -canonicalize | FileCheck %s -check-prefixes=CHECK,CHECK-AMPERE

module {
  func.func @f64_to_bf16(%arg0: f64) -> bf16 {
    %ret = arith.truncf %arg0 : f64 to bf16
    return %ret : bf16
  }
}

// CHECK-LABEL:    f64_to_bf16
// CHECK-SAME:         (%[[ARG:.*]]: f64)
// CHECK-PRE-AMPERE:       arith.truncf %[[ARG]] : f64 to f32
// CHECK-PRE-AMPERE-NOT:   arith.truncf

// CHECK-AMPERE:    %[[F32:.*]] = arith.truncf %[[ARG]] : f64 to f32
// CHECK-AMPERE:    arith.truncf %[[F32]] : f32 to bf16


module {
  func.func @bf16_to_f64(%arg0: bf16) -> f64 {
    %ret = arith.extf %arg0 : bf16 to f64
    return %ret : f64
  }
}

// CHECK-LABEL:    bf16_to_f64
// CHECK:            bitcast {{.*}} : i32 to f32
// CHECK:            arith.extf {{.*}} : f32 to f64

// -----

module {
  func.func @bf16_to_int(%arg0: bf16) -> i32 {
    %ret = arith.fptosi %arg0 : bf16 to i32
    return %ret : i32
  }
}

// CHECK-LABEL: bf16_to_int
// CHECK:         arith.fptosi {{.*}} : f32 to i32

// -----

module {
  func.func @int_to_bf16(%arg0: i16) -> bf16 {
    %ret = arith.sitofp %arg0 : i16 to bf16
    return %ret : bf16
  }
}

// CHECK-LABEL: int_to_bf16
// CHECK:         arith.sitofp {{.*}} : i16 to f32

// -----

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