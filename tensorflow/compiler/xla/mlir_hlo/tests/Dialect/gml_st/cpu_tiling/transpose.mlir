// RUN: mlir-hlo-opt %s --gml-st-cpu-tiling-pipeline \
// RUN: | FileCheck %s

func.func @transpose(%input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  %transpose = linalg.transpose
    ins(%input:tensor<16x32x64xf32>)
    outs(%init:tensor<32x64x16xf32>)
    permutation = [1, 2, 0]
  func.return %transpose : tensor<32x64x16xf32>
}
// CHECK-LABEL: func.func @transpose

// CHECK:      gml_st.parallel
// CHECK:        vector.transpose
// CHECK-SAME:     [1, 2, 0] : vector<8x1x8xf32> to vector<1x8x8xf32>
// CHECK:        gml_st.set_yield

// -----

func.func @peel_transpose(%input: tensor<16x32x65xf32>,
    %init: tensor<32x65x16xf32>) -> tensor<32x65x16xf32> {
  %transpose = linalg.transpose
    ins(%input:tensor<16x32x65xf32>)
    outs(%init:tensor<32x65x16xf32>)
    permutation = [1, 2, 0]
  func.return %transpose : tensor<32x65x16xf32>
}

// CHECK-LABEL: @peel_transpose

// CHECK:      gml_st.parallel
// CHECK:        vector.transpose
// CHECK-SAME:     [1, 2, 0] : vector<8x1x8xf32> to vector<1x8x8xf32>
// CHECK:        gml_st.set_yield

// CHECK:      gml_st.parallel
// CHECK:        gml_st.parallel
// CHECK:          tensor.extract
// CHECK:          gml_st.set_yield
// CHECK:       gml_st.set_yield
