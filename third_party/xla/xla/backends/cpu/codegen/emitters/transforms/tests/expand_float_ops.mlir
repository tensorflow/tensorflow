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

func.func @cbrt_vector(%arg0: vector<8xf64>) -> vector<8xf64> {
  %ret = math.cbrt %arg0 fastmath<reassoc> : vector<8xf64>
  return %ret : vector<8xf64>
}

// CHECK: @cbrt_vector(%[[ARG:.*]]: vector<8xf64>) -> vector<8xf64>
// CHECK-NOT: math.cbrt
// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant dense<0.33333333333333331> : vector<8xf64>
// CHECK: %[[ABS:.*]] = math.absf %[[ARG]] fastmath<reassoc> : vector<8xf64>
// CHECK: %[[CBRT_ABS:.*]] = math.powf %[[ABS]], %[[CONSTANT]] fastmath<reassoc> : vector<8xf64>
// CHECK: %[[CBRT_SIGNED:.*]] = math.copysign %[[CBRT_ABS]], %[[ARG]] fastmath<reassoc> : vector<8xf64>
// CHECK: return %[[CBRT_SIGNED]]

// -----

func.func @expm1(%arg0: f64) -> f64 {
  %ret = math.expm1 %arg0 : f64
  return %ret : f64
}

// CHECK-LABEL: @expm1
// CHECK-NOT: math.expm1

// -----

func.func @expm1_vector(%arg0: vector<4xf64>) -> vector<4xf64> {
  %ret = math.expm1 %arg0 : vector<4xf64>
  return %ret : vector<4xf64>
}

// CHECK-LABEL: @expm1_vector
// CHECK-NOT: math.expm1
