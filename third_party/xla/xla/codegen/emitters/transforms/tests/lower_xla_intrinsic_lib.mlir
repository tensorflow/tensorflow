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

// -----

func.func @exp_f16(%arg0: f16) -> f16 {
  %ret = math.exp %arg0 : f16
  return %ret : f16
}
// CHECK-LABEL: @exp_f16
// CHECK: math.exp %arg0 : f16
// CHECK: return


// -----

func.func @tanh_bf16_vector(%arg0: vector<4xbf16>) -> vector<4xbf16> {
  %ret = math.tanh %arg0 : vector<4xbf16>
  return %ret : vector<4xbf16>
}
// CHECK-LABEL: @tanh_bf16_vector
// CHECK: %[[EXT:.*]] = arith.extf %arg0 : vector<4xbf16> to vector<4xf32>
// CHECK: %[[CALL:.*]] = call @xla.tanh.v4f32(%[[EXT]])
// CHECK: %[[TRUNC:.*]] = call @xla.fptrunc.v4f32.to.v4bf16(%[[CALL]])
// CHECK: return %[[TRUNC]]

// -----

func.func @rsqrt_unsupported_f16_vector(%arg0: vector<3xf16>) -> vector<3xf16> {
  %ret = math.rsqrt %arg0 : vector<3xf16>
  return %ret : vector<3xf16>
}
// CHECK-LABEL: @rsqrt_unsupported_f16_vector
// CHECK: %[[EXT:.*]] = arith.extf %arg0 : vector<3xf16> to vector<3xf32>
// CHECK: %[[EX0:.*]] = vector.extract %[[EXT]][0]
// CHECK: %[[C0:.*]] = call @xla.rsqrt.f32(%[[EX0]])
// CHECK: %[[EX1:.*]] = vector.extract %[[EXT]][1]
// CHECK: %[[C1:.*]] = call @xla.rsqrt.f32(%[[EX1]])
// CHECK: %[[EX2:.*]] = vector.extract %[[EXT]][2]
// CHECK: %[[C2:.*]] = call @xla.rsqrt.f32(%[[EX2]])
// CHECK: %[[FROM:.*]] = vector.from_elements %[[C0]], %[[C1]], %[[C2]]
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[FROM]] : vector<3xf32> to vector<3xf16>
// CHECK: return %[[TRUNC]]

// -----

module {
  func.func @atan2_simplify(%arg0: f32) -> f32 {
    %cst = arith.constant 1.0 : f32
    %ret = math.atan2 %arg0, %cst : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @atan2_simplify
// CHECK-NOT: math.atan2
// CHECK: %[[ATAN_CALL:.*]] = call @xla.atan.f32(%arg0) : (f32) -> f32
// CHECK: return %[[ATAN_CALL]] : f32

// -----

module {
  func.func @atan2_no_simplify(%arg0: f32) -> f32 {
    %cst = arith.constant 2.0 : f32
    %ret = math.atan2 %arg0, %cst : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @atan2_no_simplify
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32
// CHECK-NOT: call @xla.atan
// CHECK: %[[CST:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[RESULT:.*]] = call @xla.atan2.f32.f32(%[[ARG0]], %[[CST]]) : (f32, f32) -> f32
// CHECK: return %[[RESULT]] : f32

// -----

module {
  func.func @atan_vector(%arg0: vector<4xf32>) -> vector<4xf32> {
    %ret = math.atan %arg0 : vector<4xf32>
    return %ret : vector<4xf32>
  }
}

// CHECK-LABEL: @atan_vector
// CHECK-NOT: math.atan
// CHECK: %[[ATAN_CALL:.*]] = call @xla.atan.v4f32(%arg0) : (vector<4xf32>) -> vector<4xf32>
// CHECK: return %[[ATAN_CALL]]

// -----

module {
  func.func @atan_vector_unsupported(%arg0: vector<3xf32>) -> vector<3xf32> {
    %ret = math.atan %arg0 : vector<3xf32>
    return %ret : vector<3xf32>
  }
}

// CHECK-LABEL: @atan_vector_unsupported
// CHECK: %[[IN0:.*]] = vector.extract %arg0[0]
// CHECK: %[[ATAN0:.*]] = call @xla.atan.f32(%[[IN0]])
// CHECK: %[[IN1:.*]] = vector.extract %arg0[1]
// CHECK: %[[ATAN1:.*]] = call @xla.atan.f32(%[[IN1]])
// CHECK: %[[IN2:.*]] = vector.extract %arg0[2]
// CHECK: %[[ATAN2:.*]] = call @xla.atan.f32(%[[IN2]])
// CHECK: %[[RESULT:.*]] = vector.from_elements %[[ATAN0]], %[[ATAN1]], %[[ATAN2]]
// CHECK: return %[[RESULT]]

// -----

module {
  func.func @atan_f32(%arg0: f32) -> f32 {
    %ret = math.atan %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @atan_f32
// CHECK-NOT: math.atan
// CHECK: %[[RESULT:.*]] = call @xla.atan.f32(%arg0) : (f32) -> f32
// CHECK: return %[[RESULT]] : f32

// -----

module {
  func.func @atan2_vector_simplify(%arg0: vector<8xf32>) -> vector<8xf32> {
    %cst = arith.constant dense<1.000000e+00> : vector<8xf32>
    %ret = math.atan2 %arg0, %cst : vector<8xf32>
    return %ret : vector<8xf32>
  }
}

// CHECK-LABEL: @atan2_vector_simplify
// CHECK-NOT: math.atan2
// CHECK: %[[RESULT:.*]] = call @xla.atan.v8f32(%arg0) : (vector<8xf32>) -> vector<8xf32>
// CHECK: return %[[RESULT]] : vector<8xf32>


// -----

module {
  func.func @atan2_f64(%arg0: f64, %arg1: f64) -> f64 {
    %ret = math.atan2 %arg0, %arg1 : f64
    return %ret : f64
  }
}
// CHECK-LABEL: @atan2_f64
// CHECK-SAME: (%[[ARG0:.*]]: f64, %[[ARG1:.*]]: f64) -> f64
// CHECK: %[[RESULT:.*]] = call @xla.atan2.f64.f64(%[[ARG0]], %[[ARG1]]) : (f64, f64) -> f64
// CHECK: return %[[RESULT]] : f64

// -----

module {
  func.func @atan2_f32_vector(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
    %ret = math.atan2 %arg0, %arg1 : vector<4xf32>
    return %ret : vector<4xf32>
  }
}
// CHECK-LABEL: @atan2_f32_vector
// CHECK-SAME: (%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: vector<4xf32>) -> vector<4xf32>
// CHECK: %[[RESULT:.*]] = call @xla.atan2.v4f32.v4f32(%[[ARG0]], %[[ARG1]]) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK: return %[[RESULT]] : vector<4xf32>

// -----

module {
  // Use a vector length of 3 as we know that will never be supported.
  func.func @atan2_unsupported_vector_size(%arg0: vector<3xf32>, %arg1: vector<3xf32>) -> vector<3xf32> {
    // CHECK-LABEL: @atan2_unsupported_vector_size
    // CHECK-SAME: (%[[ARG0:.*]]: vector<3xf32>, %[[ARG1:.*]]: vector<3xf32>) -> vector<3xf32>
    // CHECK: %[[L0:.*]] = vector.extract %[[ARG0]][0]
    // CHECK: %[[R0:.*]] = vector.extract %[[ARG1]][0]
    // CHECK: %[[ATAN2_0:.*]] = call @xla.atan2.f32.f32(%[[L0]], %[[R0]])
    // CHECK: %[[L1:.*]] = vector.extract %[[ARG0]][1]
    // CHECK: %[[R1:.*]] = vector.extract %[[ARG1]][1]
    // CHECK: %[[ATAN2_1:.*]] = call @xla.atan2.f32.f32(%[[L1]], %[[R1]])
    // CHECK: %[[L2:.*]] = vector.extract %[[ARG0]][2]
    // CHECK: %[[R2:.*]] = vector.extract %[[ARG1]][2]
    // CHECK: %[[ATAN2_2:.*]] = call @xla.atan2.f32.f32(%[[L2]], %[[R2]])
    // CHECK: %[[RESULT:.*]] = vector.from_elements %[[ATAN2_0]], %[[ATAN2_1]], %[[ATAN2_2]]
    %ret = math.atan2 %arg0, %arg1 : vector<3xf32>
    // CHECK: return %[[RESULT]]
    return %ret : vector<3xf32>
  }
}

