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
// CHECK: @local_xla.exp.f64