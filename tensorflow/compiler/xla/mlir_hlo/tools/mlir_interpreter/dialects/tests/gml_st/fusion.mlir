// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @gml_st_fusion() -> tensor<4xf32> {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %b = tensor.empty() : tensor<4xf32>
  %0 = gml_st.fusion ins(%a0 = %a : tensor<4xf32>)
                     inits(%in = %b : tensor<4xf32>) {
    %res = linalg.map { math.exp }
      ins(%a0 : tensor<4xf32>)
      outs(%in : tensor<4xf32>)
    gml_st.yield %res : tensor<4xf32>
  } : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @gml_st_fusion
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01]

func.func @bufferized() -> memref<4xf32> {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  gml_st.fusion ins(%a0 = %a : memref<4xf32>)
                inits(%in = %b : memref<4xf32>) {
    linalg.map { math.exp }
      ins(%a0 : memref<4xf32>)
      outs(%in : memref<4xf32>)
    gml_st.yield %in : memref<4xf32>
  }
  func.return %b : memref<4xf32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01]
