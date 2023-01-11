// RUN: mlir-hlo-opt %s -gml-st-to-scf -split-input-file | FileCheck %s

func.func @for(%A: memref<192x192xf32>) {
   %c24 = arith.constant 24 : index
   %c16 = arith.constant 16 : index
   %c0 = arith.constant 0 : index
   %c192 = arith.constant 192 : index
   %cst = arith.constant 0.000000e+00 : f32

  gml_st.for (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16) {
    linalg.fill ins(%cst : f32) outs(%A : memref<192x192xf32>)
    gml_st.set_yield
  }
  func.return
}

// CHECK-LABEL: @for
// CHECK-DAG:   %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C192:.*]] = arith.constant 192 : index
// CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C24]]
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C16]]
// CHECK:           linalg.fill

// -----

func.func @for_with_result(%arg: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %result = gml_st.for (%i) = (%c0) to (%c10) step (%c1)
      outs (%out = %arg: vector<4xf32>) {
    %sum = arith.addf %out, %out : vector<4xf32>
    %tile = gml_st.tile [0] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield %sum into %out[%tile]
        : vector<4xf32> into vector<4xf32>[!gml_st.tile<4>]
  } : vector<4xf32>

  func.return %result : vector<4xf32>
}

// CHECK-LABEL: @for_with_result(
// CHECK-SAME:      %[[ARG:.*]]: vector<4xf32>)

// CHECK:      %[[RESULT:.*]] = scf.for
// CHECK-SAME:     iter_args(%[[OUT:.*]] = %[[ARG]])
// CHECK-NEXT:   %[[SUM:.*]] = arith.addf %[[OUT]], %[[OUT]]
// CHECK-NEXT:   scf.yield %[[SUM]]
// CHECK:      return %[[RESULT]] : vector<4xf32>
