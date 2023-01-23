// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @yield_scalar() -> tensor<8xindex> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  %init = tensor.empty() : tensor<8xindex>
  %iota = gml_st.for (%i) = (%c0) to (%c8) step (%c1)
            outs(%init_ = %init : tensor<8xindex>) {
    %tile = gml_st.tile [%i] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %i into %init_[%tile]
      : index into tensor<8xindex>[!gml_st.tile<1>]
  } : tensor<8xindex>
  func.return %iota : tensor<8xindex>
}

// CHECK-LABEL: @yield_scalar
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 1, 2, 3, 4, 5, 6, 7]
