// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: ENTRY %main
func @main() -> tensor<2x2x1x1xf32> {
  // CHECK: constant.{{.*}} = s64[] constant(1)
  %cst = constant dense<1> : tensor<i64>
  // CHECK: constant.{{.*}} = f32[2,2,1,1]
  // CHECK-SAME: { { /*i0=0*/ { /*i1=0*/ {1} }, { /*i1=1*/ {2} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {4} } } }
  %cst_0 = constant dense<
    [[[[1.000000e+00]], [[2.000000e+00]]], [[[3.000000e+00]], [[4.000000e+00]]]]
  > : tensor<2x2x1x1xf32>

  // CHECK: s32[1] constant({1})
  %cst_1 = constant dense<1> : tensor<1xi32>

  // CHECK: %[[C:.*]] = s32[] constant(1)
  // CHECK: s32[10] broadcast(s32[] %[[C]])
  %cst_2 = constant dense<1> : tensor<10xi32>

  // CHECK: s32[4] constant({1, 2, 3, 4})
  %cst_3 = constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  // CHECK: s32[2,2] constant({ { 1, 2 }, { 3, 4 } })
  %cst_4 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>

  // CHECK: s32[2,2] constant({ { 3, 2 }, { 1, 4 } })
  %cst_5 = constant dense<[[3, 2], [1, 4]]> : tensor<2x2xi32>

  return %cst_0 : tensor<2x2x1x1xf32>
}
