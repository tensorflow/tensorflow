// RUN: emitters_opt %s -split-input-file -xla-cpu-expand-float-ops

module {
  func.func @trunc(%input: f32) -> bf16 {
    %truncated = arith.truncf %input : f32 to bf16
    func.return %truncated : bf16
  }
}
// CHECK-NOT: arith.truncf

// -----

module {
  func.func @extend(%input: bf16) -> f32 {
    %truncated = arith.extf %input : bf16 to f32
    func.return %truncated : f32
  }
}

// CHECK-NOT: arith.extf

// -----

module {
  func.func @erf64(%arg0: f64) -> f64 {
    %ret = math.erf %arg0 : f64
    return %ret : f64
  }
}

// CHECK-LABEL: @erf64
// CHECK-NOT: math.erf
// CHECK: %[[ERF_CALL:.*]] = call @erf
// CHECK: return %[[ERF_CALL]]

// -----

module {
  func.func @cbrt(%arg0: f64) -> f64 {
    %ret = math.cbrt %arg0 fastmath<reassoc> : f64 
    return %ret : f64
  }
}

// CHECK: @cbrt(%[[ARG:.*]]: f64) -> f64
// CHECK-NOT: math.cbrt
// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant 0.3333333
// CHECK: %[[ABS:.*]] = math.absf %[[ARG]] fastmath<reassoc> : f64
// CHECK: %[[CBRT_ABS:.*]] = math.powf %[[ABS]], %[[CONSTANT]] fastmath<reassoc> : f64
// CHECK: %[[CBRT_SIGNED:.*]] = math.copysign %[[CBRT_ABS]], %[[ARG]] fastmath<reassoc> : f64
// CHECK: return %[[CBRT_SIGNED]]
