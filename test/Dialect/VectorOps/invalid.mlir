// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

// CHECK-LABEL: position_empty
func @position_empty(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected non-empty position attribute}}
  %1 = vector.extractelement %arg0[] : vector<4x8x16xf32>
  return
}

// -----

// CHECK-LABEL: position_rank_overflow
func @position_rank_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than vector}}
  %1 = vector.extractelement %arg0[0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<4x8x16xf32>
  return
}

// -----

// CHECK-LABEL: position_overflow
func @position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #2 to be a positive integer smaller than the corresponding vector dimension}}
  %1 = vector.extractelement %arg0[0 : i32, 43 : i32, 0 : i32] : vector<4x8x16xf32>
  return
}

// -----

// CHECK-LABEL: position_underflow
func @position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #3 to be a positive integer smaller than the corresponding vector dimension}}
  %1 = vector.extractelement %arg0[0 : i32, 0 : i32, -1 : i32] : vector<4x8x16xf32>
  return
}
