// RUN: emitters_opt %s -split-input-file -xla-simplify-arith="fast_min_max=true" -cse -canonicalize | FileCheck %s


module {
  func.func @maximumf(%arg0: f32, %arg1: f32) -> f32 {
    %max = arith.maximumf %arg0, %arg1 : f32
    return %max : f32
  }
}

// CHECK-LABEL: @maximumf
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf uge, %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: return %[[SELECT]] : f32

// -----

module {
  func.func @minimumf(%arg0: f32, %arg1: f32) -> f32 {
    %min = arith.minimumf %arg0, %arg1 : f32
    return %min : f32
  }
}

// CHECK-LABEL: @minimumf
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpf ule, %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
// CHECK-NEXT: return %[[SELECT]] : f32
