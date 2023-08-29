// RUN: mlir-hlo-opt %s -split-input-file -verify-diagnostics

func.func @fusion_cluster_not_isolated(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %map0 = linalg.map { math.exp }
            ins(%arg0 : tensor<?x?xf32>)
            outs(%init : tensor<?x?xf32>)
  // expected-note@+1 {{required by region isolation constraints}}
  %0 = gml_st.fusion ins(%a1 = %arg1 : tensor<?x?xf32>)
                     inits(%in = %init : tensor<?x?xf32>) {
    // expected-error@+1 {{op using value defined outside the region}}
    %map1 = linalg.map { arith.mulf }
      ins(%map0, %a1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%in : tensor<?x?xf32>)
    gml_st.yield %map1 : tensor<?x?xf32>
  } : tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
