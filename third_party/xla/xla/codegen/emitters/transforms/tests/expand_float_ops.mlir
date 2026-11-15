// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: emitters_opt %s -split-input-file -xla-expand-float-ops -canonicalize | FileCheck %s
// RUN: emitters_opt %s -split-input-file -xla-expand-float-ops="approximate_tanh=false" -canonicalize
// RUN | FileCheck %s -check-prefixes=CHECK-NO-APPROX-TANH

module {
  func.func @tanh(%arg0: f32) -> f32 {
    %ret = math.tanh %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @tanh
// CHECK-NOT: tanh

// CHECK-NO-APPROX-TANH-LABEL: @tanh
// CHECK-NO-APPROX-TANH-SAME: (%[[ARG:.*]]: f32) -> f32
// CHECK-NO-APPROX-TANH: %[[RES:.*]] = math.tanh %[[ARG]] : f32
// CHECK-NO-APPROX-TANH: return %[[RES]] : f32

// -----

module {
  func.func @erf(%arg0: f32) -> f32 {
    %ret = math.erf %arg0 : f32
    return %ret : f32
  }
}

// CHECK-LABEL: @erf
// CHECK-NOT: erf
// The lowering must include a saturation select so that erf(x) returns
// exactly ±1 for |x| >= kErfInvOneMinusHalfULP, instead of evaluating the
// polynomial on the clamped input (which gives ~0.9999998 for large x).
// CHECK: math.absf
// CHECK: arith.cmpf oge
// CHECK: math.copysign
// CHECK: arith.select

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

// -----

module {
  func.func @e8m0_lowest_binade_constants() -> (i8, i8, i8, i8, i8, i8, i8, i8, i8, i8) {
    %f32_sub = arith.constant 0x003FFFFF : f32
    %f32_below = arith.constant 0x005FFFFF : f32
    %f32_mid = arith.constant 0x00600000 : f32
    %f32_zero = arith.constant 0.0 : f32
    %f32_neg_zero = arith.constant -0.0 : f32
    %f16_sub_mid = arith.constant 0x0003 : f16
    %f64_sub = arith.constant 0x37F0000000000000 : f64
    %f64_below = arith.constant 0x3807FFFFFFFFFFFF : f64
    %f64_mid = arith.constant 0x3808000000000000 : f64
    %f64_zero = arith.constant 0.0 : f64
    %t0 = arith.truncf %f32_sub : f32 to f8E8M0FNU
    %t1 = arith.truncf %f32_below : f32 to f8E8M0FNU
    %t2 = arith.truncf %f32_mid : f32 to f8E8M0FNU
    %t3 = arith.truncf %f32_zero : f32 to f8E8M0FNU
    %t4 = arith.truncf %f32_neg_zero : f32 to f8E8M0FNU
    %t5 = arith.truncf %f16_sub_mid : f16 to f8E8M0FNU
    %t6 = arith.truncf %f64_sub : f64 to f8E8M0FNU
    %t7 = arith.truncf %f64_below : f64 to f8E8M0FNU
    %t8 = arith.truncf %f64_mid : f64 to f8E8M0FNU
    %t9 = arith.truncf %f64_zero : f64 to f8E8M0FNU
    %b0 = arith.bitcast %t0 : f8E8M0FNU to i8
    %b1 = arith.bitcast %t1 : f8E8M0FNU to i8
    %b2 = arith.bitcast %t2 : f8E8M0FNU to i8
    %b3 = arith.bitcast %t3 : f8E8M0FNU to i8
    %b4 = arith.bitcast %t4 : f8E8M0FNU to i8
    %b5 = arith.bitcast %t5 : f8E8M0FNU to i8
    %b6 = arith.bitcast %t6 : f8E8M0FNU to i8
    %b7 = arith.bitcast %t7 : f8E8M0FNU to i8
    %b8 = arith.bitcast %t8 : f8E8M0FNU to i8
    %b9 = arith.bitcast %t9 : f8E8M0FNU to i8
    return %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8, %b9 : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
  }
}

// CHECK-LABEL: @e8m0_lowest_binade_constants
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i8
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i8
// CHECK-DAG: %[[F16_SUB:.*]] = arith.constant 105 : i8
// CHECK-DAG: %[[NAN:.*]] = arith.constant -1 : i8
// CHECK: return %[[ZERO]], %[[ZERO]], %[[ONE]], %[[NAN]], %[[NAN]], %[[F16_SUB]], %[[ZERO]], %[[ZERO]], %[[ONE]], %[[NAN]] : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8

// -----

module {
  func.func @maximumf_f8(%arg0: f8E4M3FN, %arg1: f8E4M3FN) -> f8E4M3FN {
    %ret = arith.maximumf %arg0, %arg1 : f8E4M3FN
    return %ret : f8E4M3FN
  }
}

// CHECK-LABEL: @maximumf_f8
// CHECK-NOT: arith.maximumf {{.*}}f8E4M3FN
// CHECK: arith.maximumf {{.*}} : f32

// -----

module {
  func.func @minimumf_f8(%arg0: f8E4M3FN, %arg1: f8E4M3FN) -> f8E4M3FN {
    %ret = arith.minimumf %arg0, %arg1 : f8E4M3FN
    return %ret : f8E4M3FN
  }
}

// CHECK-LABEL: @minimumf_f8
// CHECK-NOT: arith.minimumf {{.*}}f8E4M3FN
// CHECK: arith.minimumf {{.*}} : f32
