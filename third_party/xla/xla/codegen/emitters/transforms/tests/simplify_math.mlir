// RUN: emitters_opt %s -split-input-file -xla-simplify-arith | FileCheck %s

module {
  func.func @atan2_simplify(%arg0: f32) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    %ret = math.atan2 %arg0, %cst : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_simplify
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] : f32
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @atan2_no_simplify_not_one(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %ret = math.atan2 %arg0, %cst : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_no_simplify_not_one
// CHECK: math.atan2

// -----

module {
  func.func @atan2_no_simplify_not_constant(%arg0: f32, %arg1: f32) -> f32 {
    %ret = math.atan2 %arg0, %arg1 : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_no_simplify_not_constant
// CHECK: math.atan2

// -----

module {
  func.func @atan2_simplify_f64(%arg0: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %ret = math.atan2 %arg0, %cst : f64
    return %ret : f64
  }
}
// CHECK-LABEL: @atan2_simplify_f64
// CHECK-SAME: (%[[ARG0:.*]]: f64)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] : f64
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @atan2_simplify_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %ret = math.atan2 %arg0, %cst : tensor<4xf32>
    return %ret : tensor<4xf32>
  }
}
// CHECK-LABEL: @atan2_simplify_tensor
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] : tensor<4xf32>
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @atan2_simplify_fastmath(%arg0: f32) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    %ret = math.atan2 %arg0, %cst fastmath<fast> : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_simplify_fastmath
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] fastmath<fast> : f32
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @atan2_no_simplify_lhs_one(%arg0: f32) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    %ret = math.atan2 %cst, %arg0 : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_no_simplify_lhs_one
// CHECK: math.atan2

// -----

module {
  func.func @atan2_simplify_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
    %cst = arith.constant dense<1.000000e+00> : vector<8xf32>
    %ret = math.atan2 %arg0, %cst : vector<8xf32>
    return %ret : vector<8xf32>
  }
}
// CHECK-LABEL: @atan2_simplify_vector
// CHECK-SAME: (%[[ARG0:.*]]: vector<8xf32>)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] : vector<8xf32>
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @atan2_simplify_f16(%arg0: f16) -> f16 {
    %cst = arith.constant 1.000000e+00 : f16
    %ret = math.atan2 %arg0, %cst : f16
    return %ret : f16
  }
}
// CHECK-LABEL: @atan2_simplify_f16
// CHECK-SAME: (%[[ARG0:.*]]: f16)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] : f16
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @atan2_simplify_bf16(%arg0: bf16) -> bf16 {
    %cst = arith.constant 1.000000e+00 : bf16
    %ret = math.atan2 %arg0, %cst : bf16
    return %ret : bf16
  }
}
// CHECK-LABEL: @atan2_simplify_bf16
// CHECK-SAME: (%[[ARG0:.*]]: bf16)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] : bf16
// CHECK-NEXT:  return %[[RET]]

