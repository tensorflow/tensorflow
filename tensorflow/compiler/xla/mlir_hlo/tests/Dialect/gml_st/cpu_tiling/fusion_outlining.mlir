// RUN: mlir-hlo-opt %s --split-input-file --gml-fusion-outlining | \
// RUN: FileCheck %s

func.func @map_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = gml_st.fusion (%arg2 = %arg1: tensor<?x?xf32>,
      %arg3 = %arg0: tensor<?x?xf32>) {
    %mapped = linalg.map { math.exp } ins(%arg3 : tensor<?x?xf32>)
        outs(%arg2 : tensor<?x?xf32>)
    %mapped_0 = linalg.map { arith.mulf }
        ins(%mapped, %mapped : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%arg2 : tensor<?x?xf32>)
    %mapped_1 = linalg.map { math.absf } ins(%mapped_0 : tensor<?x?xf32>)
        outs(%arg2 : tensor<?x?xf32>)
    gml_st.yield %mapped_1 : tensor<?x?xf32>
  } : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK:      @map_fusion_fusion_0
// CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>
// CHECK:        %[[MAPPED:.*]] = linalg.map { math.exp } ins(%[[ARG1]] : tensor<?x?xf32>) outs(%[[ARG0]] : tensor<?x?xf32>)
// CHECK:        %[[MAPPED_0:.*]] = linalg.map { arith.mulf } ins(%[[MAPPED]], %[[MAPPED]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG0]] : tensor<?x?xf32>)
// CHECK:        %[[MAPPED_1:.*]] = linalg.map { math.absf } ins(%[[MAPPED_0]] : tensor<?x?xf32>) outs(%[[ARG0]] : tensor<?x?xf32>)
// CHECK:        return %[[MAPPED_1]]
// CHECK:      @map_fusion
// CHECK-SAME:     %[[ARG0_0:.*]]: tensor<?x?xf32>, %[[ARG1_0:.*]]: tensor<?x?xf32>
// CHECK:        %[[VAL:.*]] = call @map_fusion_fusion_0(%[[ARG1_0]], %[[ARG0_0]])
// CHECK:        return %[[VAL]]

// -----

func.func @multiple_fusions(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = gml_st.fusion (%arg3 = %arg0: tensor<?x?xf32>,
      %arg4 = %arg1: tensor<?x?xf32>) {
    %sorted0 = thlo.sort ins(%arg3 : tensor<?x?xf32>)
        outs(%arg4 : tensor<?x?xf32>) dimension = 0 is_stable = false
      (%lhs0: f32, %rhs0: f32) {
        %2 = arith.cmpf ogt, %lhs0, %rhs0 : f32
        thlo.yield %2 : i1
      }
    gml_st.yield %sorted0 : tensor<?x?xf32>
  } : tensor<?x?xf32>
  %1 = gml_st.fusion (%arg3 = %arg2: tensor<?xf32>,
      %arg4 = %0: tensor<?x?xf32>) {
    %reduced = linalg.reduce { arith.addf } ins(%arg4 : tensor<?x?xf32>)
        outs(%arg3 : tensor<?xf32>) dimensions = [0]
    %mapped = linalg.map { math.exp } ins(%reduced : tensor<?xf32>)
        outs(%arg3 : tensor<?xf32>)
    gml_st.yield %mapped : tensor<?xf32>
  } : tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK:      @multiple_fusions_fusion_0
// CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>
// CHECK:        %[[SORTED0:.*]] = thlo.sort ins(%[[ARG0]] : tensor<?x?xf32>) outs(%[[ARG1]] : tensor<?x?xf32>)
// CHECK:        return %[[SORTED0]]
// CHECK:      @multiple_fusions_fusion_1
// CHECK-SAME:     %[[ARG0_0:.*]]: tensor<?xf32>, %[[ARG1_0:.*]]: tensor<?x?xf32>
// CHECK:        %[[REDUCED:.*]] = linalg.reduce { arith.addf } ins(%[[ARG1_0]] : tensor<?x?xf32>) outs(%[[ARG0_0]] : tensor<?xf32>)
// CHECK:        %[[MAPPED:.*]] = linalg.map { math.exp } ins(%[[REDUCED]] : tensor<?xf32>) outs(%[[ARG0_0]] : tensor<?xf32>)
// CHECK:        return %[[MAPPED]]
// CHECK:      @multiple_fusions
// CHECK-SAME:     %[[ARG0_1:.*]]: tensor<?x?xf32>, %[[ARG1_1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?xf32>
// CHECK:        %[[VAL:.*]] = call @multiple_fusions_fusion_0(%[[ARG0_1]], %[[ARG1_1]])
// CHECK:        %[[VAL_0:.*]] = call @multiple_fusions_fusion_1(%[[ARG2]], %[[VAL]])
// CHECK:        return %[[VAL_0]]

// -----

func.func @cst_defined_above(%arg0: tensor<1x32xf32>, %arg1: tensor<32x10xf32>, %arg2: tensor<10xf32>) -> tensor<1x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x10xf32>
  %1 = gml_st.fusion (%arg3 = %0: tensor<1x10xf32>, %arg4 = %arg2: tensor<10xf32>, %arg5 = %arg0: tensor<1x32xf32>, %arg6 = %arg1: tensor<32x10xf32>) {
    %2 = linalg.fill ins(%cst : f32) outs(%arg3 : tensor<1x10xf32>) -> tensor<1x10xf32>
    gml_st.yield %2 : tensor<1x10xf32>
  } : tensor<1x10xf32>
  return %1 : tensor<1x10xf32>
}

// CHECK:      @cst_defined_above_fusion_0
// CHECK-SAME:     %[[ARG0:.*]]: tensor<1x10xf32>, %[[ARG1:.*]]: tensor<10xf32>, %[[ARG2:.*]]: tensor<1x32xf32>, %[[ARG3:.*]]: tensor<32x10xf32>
// CHECK-DAG:    %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG0]] : tensor<1x10xf32>)
// CHECK:        return %[[FILL]]
// CHECK:      @cst_defined_above
// CHECK-SAME:     %[[ARG0_0:.*]]: tensor<1x32xf32>, %[[ARG1_0:.*]]: tensor<32x10xf32>, %[[ARG2_0:.*]]: tensor<10xf32>
// CHECK:        %[[EMPTY:.*]] = tensor.empty()
// CHECK:        %[[VAL:.*]] = call @cst_defined_above_fusion_0(%[[EMPTY]], %[[ARG2_0]], %[[ARG0_0]], %[[ARG1_0]])
// CHECK:        return %[[VAL]]
