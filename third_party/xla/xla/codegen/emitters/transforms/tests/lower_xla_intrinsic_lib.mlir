// RUN: emitters_opt %s -split-input-file -xla-lower-xla-intrinsic-lib | FileCheck %s

module {
  func.func @exp_f64(%arg0: f64) -> f64 {
    %ret = math.exp %arg0 : f64
    return %ret : f64
  }
}

// CHECK: func @exp_f64
// CHECK: %[[RESULT:.*]] = call @xla.exp.f64(%arg0) : (f64) -> f64
// CHECK: return %[[RESULT]] : f64

// -----

module {
  func.func @exp_f32(%arg0: f32) -> f32 {
    %ret = math.exp %arg0 : f32
    return %ret : f32
  }
}

// CHECK: func @exp_f32
// CHECK-NOT: @xla.exp.f64
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
// CHECK: @xla.exp.v4f64

// -----

module {
  func.func @trunc(%input: f32) -> bf16 {
    %truncated = arith.truncf %input : f32 to bf16
    func.return %truncated : bf16
  }
}
// CHECK-LABEL: @trunc
// CHECK-SAME: (%[[ARG:.*]]: f32) -> bf16
// CHECK: %[[TRUNC_CALL:.*]] = call @xla.fptrunc.f32.to.bf16(%[[ARG]])
// CHECK: return %[[TRUNC_CALL]]

// -----

module {
  // CHECK-LABEL: @trunc
  func.func @trunc_vector(%input: vector<8xf32>) -> vector<8xbf16> {
    // CHECK-SAME: (%[[ARG:.*]]: vector<8xf32>) -> vector<8xbf16>
    // CHECK: %[[TRUNC_CALL:.*]] = call @xla.fptrunc.v8f32.to.v8bf16(%[[ARG]])
    %truncated = arith.truncf %input : vector<8xf32> to vector<8xbf16>
    // CHECK: return %[[TRUNC_CALL]]
    func.return %truncated : vector<8xbf16>
  }
}

// -----

module {
  func.func @erf32(%arg0: f32) -> f32 {
    %ret = math.erf %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @erf32
// CHECK-NOT: math.erf
// CHECK: %[[ERF_CALL:.*]] = call @xla.erf.f32
// CHECK: return %[[ERF_CALL]]

// -----

module {
  func.func @erf32_vector(%arg0: vector<4xf32>) -> vector<4xf32> {
    %ret = math.erf %arg0 : vector<4xf32>
    return %ret : vector<4xf32>
  }
}

// CHECK-LABEL: @erf32_vector
// CHECK-NOT: math.erf
// CHECK: %[[ERF_CALL:.*]] = call @xla.erf.v4f32
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
  func.func @erf64_vector(%arg0: vector<4xf64>) -> vector<4xf64> {
    %ret = math.erf %arg0 : vector<4xf64>
    return %ret : vector<4xf64>
  }
}

// CHECK-LABEL: @erf64_vector
// CHECK-NOT: math.erf
// CHECK-COUNT-4: call @erf

// -----

module {
  func.func @rsqrt(%arg0: f32) -> f32 {
    %ret = math.rsqrt %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @xla.rsqrt.f32
// CHECK-NOT: math.rsqrt
// CHECK: %[[RSQRT_CALL:.*]] = call @xla.rsqrt.f32
// CHECK: return %[[RSQRT_CALL]]

// -----

module {
  func.func @rsqrt(%arg0: f64) -> f64 {
    %ret = math.rsqrt %arg0 : f64
    return %ret : f64
  }
}

// CHECK-LABEL: @xla.rsqrt.f64
// CHECK-NOT: math.rsqrt
// CHECK: %[[RSQRT_CALL:.*]] = call @xla.rsqrt.f64
// CHECK: return %[[RSQRT_CALL]]

// -----

// Use a vector length of 3 as we know that will never be supported.
func.func @rsqrt_unsupported_vector_size(%arg0: vector<3xf32>) -> vector<3xf32> {
  // CHECK: %[[IN0:.*]] = vector.extract %arg0[0]
  // CHECK: %[[RSQRT0:.*]] = call @xla.rsqrt.f32(%[[IN0]])
  // CHECK: %[[IN1:.*]] = vector.extract %arg0[1]
  // CHECK: %[[RSQRT1:.*]] = call @xla.rsqrt.f32(%[[IN1]])
  // CHECK: %[[IN2:.*]] = vector.extract %arg0[2]
  // CHECK: %[[RSQRT2:.*]] = call @xla.rsqrt.f32(%[[IN2]])
  // CHECK: %[[RESULT:.*]] = vector.from_elements %[[RSQRT0]], %[[RSQRT1]], %[[RSQRT2]]
  %ret = math.rsqrt %arg0 : vector<3xf32>
  // CHECK: return %[[RESULT]]
  return %ret : vector<3xf32>
}
