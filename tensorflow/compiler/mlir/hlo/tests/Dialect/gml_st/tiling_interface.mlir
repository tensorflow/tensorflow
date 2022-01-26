// RUN: mlir-hlo-opt %s -gml-greedy-tiling -split-input-file | FileCheck %s

// CHECK-LABEL: @add
// CHECK-SAME: %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[POINT:.*]]: !gml_st.point
func @add(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: !gml_st.point) -> f32 {
  // CHECK: %[[LHSP:.*]] = gml_st.materialize %[[LHS]] at %[[POINT]]
  // CHECK: %[[RHSP:.*]] = gml_st.materialize %[[RHS]] at %[[POINT]]
  // CHECK: %[[RESULT:.*]] = arith.addf %[[LHSP]], %[[RHSP]]
  // CHECK: return %[[RESULT]]
  %0 = mhlo.add %arg0, %arg1 : tensor<32x32xf32>
  %1 = gml_st.materialize %0 at %arg2 : tensor<32x32xf32> at !gml_st.point
  return %1 : f32
}
