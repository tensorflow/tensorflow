// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-greedy-tiling-and-fusion="tile-sizes=8,16 distribute=true distribution-label=test" \
// RUN:     --canonicalize --cse | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @softmax(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant -0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<64x128xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = arith.maxf %arg2, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64xf32>
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<64x128xf32>
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %11 = arith.subf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  %6 = linalg.generic {indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = math.exp %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %8 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<64x128xf32>)
      outs(%7 : tensor<64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = arith.addf %arg2, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64xf32>
  %9 = linalg.generic {indexing_maps = [#map1, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<64x128xf32>
  %10 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%6, %9 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %11 = arith.divf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  return %10 : tensor<64x128xf32>
}

// CHECK-LABEL:  @softmax
// CHECK-NOT:      linalg.generic
// CHECK:          %[[PARALLEL:.*]] = gml_st.parallel
// CHECK:            gml_st.set_yield
// CHECK-NOT:      linalg.generic
// CHECK:          return %[[PARALLEL]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @softmax_fuse_tensor_extract(%arg0: tensor<64x128xf32>,
                                       %cst_tensor: tensor<f32>) -> tensor<64x128xf32> {
  %cst = tensor.extract %cst_tensor[] : tensor<f32>
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<64x128xf32>) outs(%1 : tensor<64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = arith.maxf %arg2, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64xf32>
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<64x128xf32>
  %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %4 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %11 = arith.subf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  %6 = linalg.generic {indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = math.exp %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %8 = linalg.generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<64x128xf32>)
      outs(%7 : tensor<64xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %11 = arith.addf %arg2, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<64xf32>
  %9 = linalg.generic {indexing_maps = [#map1, #map0],
      iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<64xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<64x128xf32>
  %10 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
      ins(%6, %9 : tensor<64x128xf32>, tensor<64x128xf32>)
      outs(%3 : tensor<64x128xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %11 = arith.divf %arg1, %arg2 : f32
    linalg.yield %11 : f32
  } -> tensor<64x128xf32>
  return %10 : tensor<64x128xf32>
}

// CHECK-LABEL:  @softmax
// CHECK-NOT:      tensor.extract
// CHECK:          %[[PARALLEL_BLOCK:.*]] = gml_st.parallel
// CHECK:            tensor.extract
