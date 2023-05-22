// RUN: mlir-hlo-opt %s --split-input-file --gml-fusion-outlining | \
// RUN: FileCheck %s

func.func @map_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = gml_st.fusion ins(%arg2 = %arg0: tensor<?x?xf32>)
                     inits(%arg3 = %arg1: tensor<?x?xf32>) {
    %mapped = linalg.map { math.exp } ins(%arg2 : tensor<?x?xf32>)
        outs(%arg3 : tensor<?x?xf32>)
    %mapped_0 = linalg.map { arith.mulf }
        ins(%mapped, %mapped : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%arg3 : tensor<?x?xf32>)
    %mapped_1 = linalg.map { math.absf } ins(%mapped_0 : tensor<?x?xf32>)
        outs(%arg3 : tensor<?x?xf32>)
    gml_st.yield %mapped_1 : tensor<?x?xf32>
  } : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @map_fusion_fusion_0
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>
// CHECK-SAME:       attributes {fusion}
// CHECK:         %[[FUSION:.*]] = gml_st.fusion
// CHECK-SAME:        ins(%[[ARG2:.*]] = %[[ARG0]]: tensor<?x?xf32>)
// CHECK-SAME:        inits(%[[ARG3:.*]] = %[[ARG1]]: tensor<?x?xf32>)
// CHECK:           %[[MAPPED:.*]] = linalg.map { math.exp } ins(%[[ARG2]] : tensor<?x?xf32>) outs(%[[ARG3]] : tensor<?x?xf32>)
// CHECK:           %[[MAPPED_0:.*]] = linalg.map { arith.mulf } ins(%[[MAPPED]], %[[MAPPED]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG3]] : tensor<?x?xf32>)
// CHECK:           %[[MAPPED_1:.*]] = linalg.map { math.absf } ins(%[[MAPPED_0]] : tensor<?x?xf32>) outs(%[[ARG3]] : tensor<?x?xf32>)
// CHECK:           gml_st.yield %[[MAPPED_1]]
// CHECK:         return %[[FUSION]]
// CHECK:       @map_fusion(%[[ARG0_0:.*]]: tensor<?x?xf32>, %[[ARG1_0:.*]]: tensor<?x?xf32>)
// CHECK:         %[[VAL:.*]] = call @map_fusion_fusion_0(%[[ARG0_0]], %[[ARG1_0]])
// CHECK:         return %[[VAL]]

// -----

func.func @multiple_fusions(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = gml_st.fusion ins(%arg3 = %arg0: tensor<?x?xf32>)
                     inits(%arg4 = %arg1: tensor<?x?xf32>) {
    %sorted0 = thlo.sort ins(%arg3 : tensor<?x?xf32>)
        outs(%arg4 : tensor<?x?xf32>) dimension = 0 is_stable = false
        (%lhs0: f32, %rhs0: f32) {
      %2 = arith.cmpf ogt, %lhs0, %rhs0 : f32
      thlo.yield %2 : i1
    }
    gml_st.yield %sorted0 : tensor<?x?xf32>
  } : tensor<?x?xf32>
  %1 = gml_st.fusion ins(%arg3 = %0: tensor<?x?xf32>)
                     inits(%arg4 = %arg2: tensor<?xf32>) {
    %reduced = linalg.reduce { arith.addf } ins(%arg3 : tensor<?x?xf32>)
        outs(%arg4 : tensor<?xf32>) dimensions = [0]
    %mapped = linalg.map { math.exp } ins(%reduced : tensor<?xf32>)
        outs(%arg4 : tensor<?xf32>)
    gml_st.yield %mapped : tensor<?xf32>
  } : tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: @multiple_fusions_fusion_0
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>
// CHECK-SAME:      attributes {fusion}
// CHECK:         %[[FUSION:.*]] = gml_st.fusion
// CHECK-SAME:        ins(%[[ARG2:.*]] = %[[ARG0]]: tensor<?x?xf32>)
// CHECK-SAME:        inits(%[[ARG3:.*]] = %[[ARG1]]: tensor<?x?xf32>)
// CHECK:           %[[SORTED0:.*]] = thlo.sort ins(%[[ARG2]] : tensor<?x?xf32>) outs(%[[ARG3]] : tensor<?x?xf32>) dimension = 0 is_stable = false
// CHECK:             (%[[LHS0:.*]]: f32, %[[RHS0:.*]]: f32)
// CHECK:               %[[CMPF:.*]] = arith.cmpf ogt, %[[LHS0]], %[[RHS0]] : f32
// CHECK:               thlo.yield %[[CMPF]] : i1
// CHECK:           gml_st.yield %[[SORTED0]]
// CHECK:         return %[[FUSION]]
// CHECK:       @multiple_fusions_fusion_1
// CHECK-SAME:      %[[ARG0_0:.*]]: tensor<?x?xf32>, %[[ARG1_0:.*]]: tensor<?xf32>
// CHECK-SAME:      attributes {fusion}
// CHECK:         %[[FUSION_0:.*]] = gml_st.fusion
// CHECK-SAME:        ins(%[[ARG2_0:.*]] = %[[ARG0_0]]: tensor<?x?xf32>)
// CHECK-SAME:        inits(%[[ARG3_0:.*]] = %[[ARG1_0]]: tensor<?xf32>)
// CHECK:           %[[REDUCED:.*]] = linalg.reduce { arith.addf } ins(%[[ARG2_0]] : tensor<?x?xf32>) outs(%[[ARG3_0]] : tensor<?xf32>) dimensions = [0]
// CHECK:           %[[MAPPED:.*]] = linalg.map { math.exp } ins(%[[REDUCED]] : tensor<?xf32>) outs(%[[ARG3_0]] : tensor<?xf32>)
// CHECK:           gml_st.yield %[[MAPPED]]
// CHECK:         return %[[FUSION_0]]
// CHECK:       @multiple_fusions
// CHECK-SAME:      %[[ARG0_1:.*]]: tensor<?x?xf32>, %[[ARG1_1:.*]]: tensor<?x?xf32>, %[[ARG2_1:.*]]: tensor<?xf32>
// CHECK:         %[[VAL:.*]] = call @multiple_fusions_fusion_0(%[[ARG0_1]], %[[ARG1_1]])
// CHECK:         %[[VAL_0:.*]] = call @multiple_fusions_fusion_1(%[[VAL]], %[[ARG2_1]])
// CHECK:         return %[[VAL_0]]

// -----

func.func @cst_defined_above() -> tensor<1x10xf32> {
  %0 = tensor.empty() : tensor<1x10xf32>
  %1 = gml_st.fusion inits(%arg3 = %0 : tensor<1x10xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%arg3 : tensor<1x10xf32>) -> tensor<1x10xf32>
    gml_st.yield %2 : tensor<1x10xf32>
  } { some_attr = 123 } : tensor<1x10xf32>
  return %1 : tensor<1x10xf32>
}

// CHECK-LABEL: @cst_defined_above_fusion_0
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x10xf32>
// CHECK-SAME:      attributes {fusion}
// CHECK:         %[[FUSION:.*]] = gml_st.fusion inits(%[[ARG1:.*]] = %[[ARG0]]: tensor<1x10xf32>) {
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG1]] : tensor<1x10xf32>)
// CHECK:           gml_st.yield %[[FILL]]
// CHECK:         } {some_attr = 123 : i64}
// CHECK:         return %[[FUSION]]
// CHECK:       @cst_defined_above
// CHECK:         %[[EMPTY:.*]] = tensor.empty()
// CHECK:         %[[VAL:.*]] = call @cst_defined_above_fusion_0(%[[EMPTY]])
// CHECK:         return %[[VAL]]
