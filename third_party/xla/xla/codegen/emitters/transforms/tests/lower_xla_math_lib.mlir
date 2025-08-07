// RUN: emitters_opt %s -split-input-file -xla-lower-xla-math-lib | FileCheck %s

module {
  func.func @exp_f64(%arg0: f64) -> f64 {
    %ret = math.exp %arg0 : f64
    return %ret : f64
  }
}

// CHECK: func @exp_f64
// CHECK: %[[RESULT:.*]] = call @local_xla.exp.f64(%arg0) : (f64) -> f64
// CHECK: return %[[RESULT]] : f64

// -----

module {
  func.func @exp_f32(%arg0: f32) -> f32 {
    %ret = math.exp %arg0 : f32
    return %ret : f32
  }
}

// CHECK: func @exp_f32
// CHECK-NOT: @local_xla.exp.f64
// CHECK: math.exp %arg0 : f32
// CHECK: return

// -----

module {
  func.func @exp_f64_vector(%arg0: vector<4xf64>) -> vector<4xf64> {
    %ret = math.exp %arg0 : vector<4xf64>
    return %ret : vector<4xf64>
  }
}

// CHECK: func @exp_f64_vector
// CHECK-NOT: math.exp %arg0 : vector<4xf64>
// CHECK: @local_xla.exp.v4f64

// -----

module {
  func.func @trunc(%input: f32) -> bf16 {
    %truncated = arith.truncf %input : f32 to bf16
    func.return %truncated : bf16
  }
}
// CHECK-LABEL: @trunc
// CHECK-SAME: (%[[ARG:.*]]: f32) -> bf16
// CHECK: %[[TRUNC_CALL:.*]] = call @local_xla.fptrunc.f32.to.bf16(%[[ARG]])
// CHECK: return %[[TRUNC_CALL]]

// -----

module {
  func.func @erf32(%arg0: f32) -> f32 {
    %ret = math.erf %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @erf32
// CHECK-NOT: math.erf
// CHECK: %[[ERF_CALL:.*]] = call @local_xla.erf.f32
// CHECK: return %[[ERF_CALL]]

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
  func.func @rsqrt(%arg0: f32) -> f32 {
    %ret = math.rsqrt %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @local_xla.rsqrt.f32
// CHECK-NOT: math.rsqrt
// CHECK: %[[RSQRT_CALL:.*]] = call @local_xla.rsqrt.f32
// CHECK: return %[[RSQRT_CALL]]

// -----

module {
  func.func @rsqrt(%arg0: f64) -> f64 {
    %ret = math.rsqrt %arg0 : f64
    return %ret : f64
  }
}

// CHECK-LABEL: @local_xla.rsqrt.f64
// CHECK-NOT: math.rsqrt
// CHECK: %[[RSQRT_CALL:.*]] = call @local_xla.rsqrt.f64
// CHECK: return %[[RSQRT_CALL]]