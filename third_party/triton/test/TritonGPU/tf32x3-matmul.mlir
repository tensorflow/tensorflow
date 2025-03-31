// RUN: triton-opt %s -tritongpu-F32DotTC -canonicalize  | FileCheck %s --check-prefixes=CHECK

// CHECK:     %[[DOT1:.*]] = tt.dot %[[LHS_LOW:.*]], %[[RHS_HIGH:.*]], %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK:     %[[DOT2:.*]] = tt.dot %[[LHS_HIGH:.*]], %[[RHS_LOW:.*]], %[[DOT1]], inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK:     %[[CMP:.*]] = arith.cmpf uno, %[[DOT2]], %[[DOT2]] : tensor<16x16xf32>
// CHECK:     %[[MASKED:.*]] = arith.select %[[CMP]], %cst, %[[DOT2]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK:     %[[RESULT:.*]] = tt.dot %[[LHS_HIGH]], %[[RHS_HIGH]], %[[MASKED]], inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

module {
  tt.func @dot_test(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %4 = tt.dot %arg0, %arg1, %arg2, inputPrecision = tf32x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return %4 : tensor<16x16xf32>
  }
}
