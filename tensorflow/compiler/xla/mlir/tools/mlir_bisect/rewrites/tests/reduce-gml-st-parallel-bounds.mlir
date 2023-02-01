// RUN: mlir-bisect %s --debug-strategy=ReduceGmlStParallelBounds | FileCheck %s

func.func @main() -> tensor<8xindex> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %init = tensor.empty() : tensor<8xindex>
  %iota = gml_st.parallel (%i) = (%c0) to (%c8) step (%c1)
      outs (%init_ = %init: tensor<8xindex>) {
    %tile = gml_st.tile [%i] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %i into %init_[%tile]
      : index into tensor<8xindex>[!gml_st.tile<1>]
  } : tensor<8xindex>
  func.return %iota : tensor<8xindex>
}

// CHECK: func @main()
// CHECK:   %[[C7:.*]] = arith.constant 7
// CHECK:   gml_st.parallel {{.*}} to (%[[C7]])
