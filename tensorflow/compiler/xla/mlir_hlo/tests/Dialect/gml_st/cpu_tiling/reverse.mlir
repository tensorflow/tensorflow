// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline \
// RUN: | FileCheck %s

func.func @reverse_static_perfect_tiles(
  %input: tensor<64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
  %res = thlo.reverse
    ins(%input: tensor<64xf32>)
    outs(%init: tensor<64xf32>)
    reverse_dimensions = [0]
  func.return %res : tensor<64xf32>
}

// CHECK-LABEL: @reverse_static_perfect_tiles

// CHECK: gml_st.parallel
// CHECK:   vector.shuffle
// CHECK:   gml_st.set_yield

// -----

func.func @reverse_dynamic(
  %input: tensor<?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %res = thlo.reverse
     ins(%input: tensor<?x?xf32>)
     outs(%init: tensor<?x?xf32>)
     reverse_dimensions = [0, 1]
  func.return %res : tensor<?x?xf32>
}

// CHECK-LABEL: @reverse_dynamic

// CHECK: gml_st.parallel
// CHECK:   vector.shuffle
// CHECK:   gml_st.set_yield

// CHECK: gml_st.parallel
// CHECK:   gml_st.parallel
// CHECK:     tensor.extract_slice
// CHECK:     gml_st.set_yield
// CHECK:   gml_st.set_yield
