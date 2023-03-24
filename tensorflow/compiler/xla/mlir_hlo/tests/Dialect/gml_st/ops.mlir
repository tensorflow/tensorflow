// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect | \
// RUN: mlir-hlo-opt --verify-diagnostics --split-input-file \
// RUN:     --allow-unregistered-dialect | \
// RUN: FileCheck %s

func.func @fusion_cluster(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = gml_st.fusion (%a0 = %arg0 : tensor<?x?xf32>,
                      %a1 = %arg1 : tensor<?x?xf32>,
                      %in = %init : tensor<?x?xf32>) {
    %map0 = linalg.map { math.exp }
      ins(%a0 : tensor<?x?xf32>)
      outs(%in : tensor<?x?xf32>)
    %map1 = linalg.map { arith.mulf }
      ins(%map0, %a1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%in : tensor<?x?xf32>)
    gml_st.yield %map1 : tensor<?x?xf32>
  } { "some_attr" = 1 } : tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @fusion_cluster
// CHECK:       gml_st.fusion
// CHECK:         linalg.map
// CHECK:         linalg.map
// CHECK:         gml_st.yield
