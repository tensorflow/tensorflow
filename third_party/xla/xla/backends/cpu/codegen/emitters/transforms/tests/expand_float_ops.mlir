// RUN: emitters_opt %s -split-input-file -xla-cpu-expand-float-ops | FileCheck %s

func.func @extend(%input: bf16) -> f32 {
  // CHECK-NOT: arith.extf
  %truncated = arith.extf %input : bf16 to f32
  func.return %truncated : f32
}

// -----

func.func @extend_vector(%input: vector<8xbf16>) -> vector<8xf32> {
  // CHECK-NOT: arith.extf
  %truncated = arith.extf %input : vector<8xbf16> to vector<8xf32>
  func.return %truncated : vector<8xf32>
}

// -----

func.func @cbrt(%arg0: f64) -> f64 {
  %ret = math.cbrt %arg0 fastmath<reassoc> : f64
  return %ret : f64
}

// CHECK: @cbrt(%[[ARG:.*]]: f64) -> f64
// CHECK-NOT: math.cbrt
// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant 0.3333333
// CHECK: %[[ABS:.*]] = math.absf %[[ARG]] fastmath<reassoc> : f64
// CHECK: %[[CBRT_ABS:.*]] = math.powf %[[ABS]], %[[CONSTANT]] fastmath<reassoc> : f64
// CHECK: %[[CBRT_SIGNED:.*]] = math.copysign %[[CBRT_ABS]], %[[ARG]] fastmath<reassoc> : f64
// CHECK: return %[[CBRT_SIGNED]]

// -----

func.func @expm1(%arg0: f64) -> f64 {
  %ret = math.expm1 %arg0 : f64
  return %ret : f64
}

// CHECK-LABEL: @expm1
// CHECK-NOT: math.expm1
