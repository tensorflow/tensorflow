// RUN: mlir-hlo-opt %s --split-input-file --gml-collapse-shape | FileCheck %s

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-collapse-shape="retain-trailing-dims=1" | \
// RUN: FileCheck %s --check-prefix=CHECK-1

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-collapse-shape="retain-trailing-dims=2" | \
// RUN: FileCheck %s --check-prefix=CHECK-2

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-collapse-shape="retain-trailing-dims=3" | \
// RUN: FileCheck %s --check-prefix=CHECK-3

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @bcast(%arg0: tensor<2x4x2048xf32>) -> tensor<2x4x2048x4096xf32> {
  %0 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<2x4x2048xf32>) outs(%0 : tensor<2x4x2048x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x2048x4096xf32>
  return %1 : tensor<2x4x2048x4096xf32>
}

// CHECK:        func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-NOT:    collapse_shape
// CHECK-NOT:    expand_shape

// CHECK-1:      #map = affine_map<(d0, d1) -> (d0)>
// CHECK-1:      #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-1:      func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[GENERIC:.*]] = linalg.generic
// CHECK-1-SAME:       indexing_maps = [#map, #map1]
// CHECK-1-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-1-SAME:       ins(%[[COLLAPSED]] : tensor<16384xf32>)
// CHECK-1-SAME:       outs(%[[EMPTY]] : tensor<16384x4096xf32>)
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      #map = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-2:      #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-2:      func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-2:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2]]
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-2:        %[[GENERIC:.*]] = linalg.generic
// CHECK-2-SAME:       indexing_maps = [#map, #map1]
// CHECK-2-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-2-SAME:       ins(%[[COLLAPSED]] : tensor<8x2048xf32>)
// CHECK-2-SAME:       outs(%[[EMPTY]] : tensor<8x2048x4096xf32>)
// CHECK-2:        ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-2:          linalg.yield %[[IN]] : f32
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]]
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----

#map = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @bcast_from_scalar() -> tensor<2x4x2048x4096xf32> {
  %0 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %cst = arith.constant 0xFF800000 : f32
  %1 = tensor.empty() : tensor<f32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<f32>) -> tensor<f32>
  %3 = linalg.generic {indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%2 : tensor<f32>) outs(%0 : tensor<2x4x2048x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x2048x4096xf32>
  return %3 : tensor<2x4x2048x4096xf32>
}

// CHECK:      #map = affine_map<(d0) -> ()>
// CHECK:      #map1 = affine_map<(d0) -> (d0)>
// CHECK:      func.func @bcast_from_scalar()
// CHECK:        %[[EMPTY:.*]] = tensor.empty() : tensor<67108864xf32>
// CHECK:        %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#map, #map1]
// CHECK-SAME:       iterator_types = ["parallel"]
// CHECK-SAME:       ins(%{{.*}} : tensor<f32>)
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<67108864xf32>)
// CHECK:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-SAME:     0, 1, 2, 3]]
// CHECK:        return %[[EXPANDED]]

// CHECK-1:      #map = affine_map<(d0, d1) -> ()>
// CHECK-1:      #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-1:      func.func @bcast_from_scalar()
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty() : tensor<16384x4096xf32>
// CHECK-1:        %[[GENERIC:.*]] = linalg.generic
// CHECK-1-SAME:       indexing_maps = [#map, #map1]
// CHECK-1-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-1-SAME:       ins(%{{.*}} : tensor<f32>)
// CHECK-1-SAME:       outs(%[[EMPTY]] : tensor<16384x4096xf32>)
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      #map = affine_map<(d0, d1, d2) -> ()>
// CHECK-2:      #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-2:      func.func @bcast_from_scalar()
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty() : tensor<8x2048x4096xf32>
// CHECK-2:        %[[GENERIC:.*]] = linalg.generic
// CHECK-2-SAME:       indexing_maps = [#map, #map1]
// CHECK-2-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-2-SAME:       ins(%{{.*}} : tensor<f32>)
// CHECK-2-SAME:       outs(%[[EMPTY]] : tensor<8x2048x4096xf32>)
// CHECK-2:        ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-2:          linalg.yield %[[IN]] : f32
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @bcast_from_scalar()
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @reduction(%arg0: tensor<2x4x2048x4096xf32>) -> tensor<2x4x2048xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2x4x2048xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x2048xf32>)
      -> tensor<2x4x2048xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0 : tensor<2x4x2048x4096xf32>) outs(%1 : tensor<2x4x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.maxf %out, %in : f32
    linalg.yield %3 : f32
  } -> tensor<2x4x2048xf32>
  return %2 : tensor<2x4x2048xf32>
}

// CHECK:        func.func @reduction(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-NOT:    collapse_shape
// CHECK-NOT:    expand_shape

// CHECK-1:      #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-1:      #map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-1:      func.func @reduction(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-1-DAG:    %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<16384xf32>)
// CHECK-1:        %[[GENERIC:.*]] = linalg.generic
// CHECK-1-SAME:       indexing_maps = [#map, #map1]
// CHECK-1-SAME:       iterator_types = ["parallel", "reduction"]
// CHECK-1-SAME:       ins(%[[COLLAPSED]] : tensor<16384x4096xf32>)
// CHECK-1-SAME:       outs(%[[FILL]] : tensor<16384xf32>)
// CHECK-1:        ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-1:          %[[MAXF:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
// CHECK-1:          linalg.yield %[[MAXF]] : f32
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-1-SAME:     [0, 1, 2]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-2:      #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-2:      func.func @reduction(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-2-DAG:    %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-2:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-2:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<8x2048xf32>)
// CHECK-2:        %[[GENERIC:.*]] = linalg.generic
// CHECK-2-SAME:       indexing_maps = [#map, #map1]
// CHECK-2-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-2-SAME:       ins(%[[COLLAPSED]] : tensor<8x2048x4096xf32>)
// CHECK-2-SAME:       outs(%[[FILL]] : tensor<8x2048xf32>)
// CHECK-2:        ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-2:          %[[MAXF:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
// CHECK-2:          linalg.yield %[[MAXF]] : f32
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-2-SAME:     [0, 1], [2]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @reduction(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @cwise(%arg0: tensor<2x4x2048x4096xf32>,
    %arg1: tensor<2x4x2048x4096xf32>) -> tensor<2x4x2048x4096xf32> {
  %0 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2x4x2048x4096xf32>, tensor<2x4x2048x4096xf32>)
      outs(%0 : tensor<2x4x2048x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.subf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<2x4x2048x4096xf32>
  return %1 : tensor<2x4x2048x4096xf32>
}

// CHECK:        #map = affine_map<(d0) -> (d0)>
// CHECK:        func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK:          %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-SAME:       [0, 1, 2, 3]]
// CHECK:          %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG1]] [
// CHECK-SAME:       [0, 1, 2, 3]]
// CHECK:          %[[EMPTY:.*]] = tensor.empty()
// CHECK:          %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#map, #map, #map]
// CHECK-SAME:         iterator_types = ["parallel"]
// CHECK-SAME:         ins(%[[COLLAPSED]], %[[COLLAPSED_0]] : tensor<67108864xf32>, tensor<67108864xf32>)
// CHECK-SAME:         outs(%[[EMPTY]] : tensor<67108864xf32>)
// CHECK:          ^bb0(%[[IN:.*]]: f32, %[[IN_1:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:            %[[SUBF:.*]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:            linalg.yield %[[SUBF]] : f32
// CHECK:          %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-SAME:       [0, 1, 2, 3]]
// CHECK:          return %[[EXPANDED]]

// CHECK-1:      #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-1:      func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG1]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[GENERIC:.*]] = linalg.generic
// CHECK-1-SAME:       indexing_maps = [#map, #map, #map]
// CHECK-1-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-1-SAME:       ins(%[[COLLAPSED]], %[[COLLAPSED_0]] : tensor<16384x4096xf32>, tensor<16384x4096xf32>)
// CHECK-1-SAME:       outs(%[[EMPTY]] : tensor<16384x4096xf32>)
// CHECK-1:        ^bb0(%[[IN:.*]]: f32, %[[IN_1:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-1:          %[[SUBF:.*]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-1:          linalg.yield %[[SUBF]] : f32
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-2:      func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-2:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG1]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-2:        %[[GENERIC:.*]] = linalg.generic
// CHECK-2-SAME:       indexing_maps = [#map, #map, #map]
// CHECK-2-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-2-SAME:       ins(%[[COLLAPSED]], %[[COLLAPSED_0]] : tensor<8x2048x4096xf32>, tensor<8x2048x4096xf32>)
// CHECK-2-SAME:       outs(%[[EMPTY]] : tensor<8x2048x4096xf32>)
// CHECK-2:        ^bb0(%[[IN:.*]]: f32, %[[IN_1:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-2:          %[[SUBF:.*]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-2:          linalg.yield %[[SUBF]] : f32
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @partial_softmax(%arg0: tensor<2x4x2048x4096xf32>)
    -> tensor<2x4x2048x4096xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2x4x2048xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x2048xf32>)
      -> tensor<2x4x2048xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0 : tensor<2x4x2048x4096xf32>) outs(%1 : tensor<2x4x2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.maxf %out, %in : f32
    linalg.yield %6 : f32
  } -> tensor<2x4x2048xf32>
  %3 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%2 : tensor<2x4x2048xf32>) outs(%3 : tensor<2x4x2048x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x2048x4096xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %4 : tensor<2x4x2048x4096xf32>, tensor<2x4x2048x4096xf32>)
      outs(%3 : tensor<2x4x2048x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.subf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<2x4x2048x4096xf32>
  return %5 : tensor<2x4x2048x4096xf32>
}

// CHECK-1:      #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-1:      #map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-1:      func.func @partial_softmax(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-1-DAG:    %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<16384xf32>)
// CHECK-1:        %[[GENERIC:.*]] = linalg.generic
// CHECK-1-SAME:       indexing_maps = [#map, #map1]
// CHECK-1-SAME:       iterator_types = ["parallel", "reduction"]
// CHECK-1-SAME:       ins(%[[COLLAPSED]] : tensor<16384x4096xf32>)
// CHECK-1-SAME:       outs(%[[FILL]] : tensor<16384xf32>)
// CHECK-1:        ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-1:          %[[MAXF:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
// CHECK-1:          linalg.yield %[[MAXF]] : f32
// CHECK-1:        %[[EMPTY_0:.*]] = tensor.empty()
// CHECK-1:        %[[GENERIC_0:.*]] = linalg.generic
// CHECK-1-SAME:       indexing_maps = [#map1, #map]
// CHECK-1-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-1-SAME:       ins(%[[GENERIC]] : tensor<16384xf32>)
// CHECK-1-SAME:       outs(%[[EMPTY_0]] : tensor<16384x4096xf32>)
// CHECK-1:        ^bb0(%[[IN_0:.*]]: f32, %[[OUT_0:.*]]: f32):
// CHECK-1:          linalg.yield %[[IN_0]] : f32
// CHECK-1:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY_1:.*]] = tensor.empty()
// CHECK-1:        %[[GENERIC_1:.*]] = linalg.generic
// CHECK-1-SAME:       indexing_maps = [#map, #map, #map]
// CHECK-1-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-1-SAME:       ins(%[[COLLAPSED_0]], %[[GENERIC_0]] : tensor<16384x4096xf32>, tensor<16384x4096xf32>)
// CHECK-1-SAME:       outs(%[[EMPTY_1]] : tensor<16384x4096xf32>)
// CHECK-1:        ^bb0(%[[IN_1:.*]]: f32, %[[IN_1_0:.*]]: f32, %[[OUT_1:.*]]: f32):
// CHECK-1:          %[[SUBF:.*]] = arith.subf %[[IN_1]], %[[IN_1_0]] : f32
// CHECK-1:          linalg.yield %[[SUBF]] : f32
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC_1]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-2:      #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-2:      func.func @partial_softmax(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-2-DAG:    %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-2:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-2:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<8x2048xf32>)
// CHECK-2:        %[[GENERIC:.*]] = linalg.generic
// CHECK-2-SAME:       indexing_maps = [#map, #map1]
// CHECK-2-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-2-SAME:       ins(%[[COLLAPSED]] : tensor<8x2048x4096xf32>)
// CHECK-2-SAME:       outs(%[[FILL]] : tensor<8x2048xf32>)
// CHECK-2:        ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-2:          %[[MAXF:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
// CHECK-2:          linalg.yield %[[MAXF]] : f32
// CHECK-2:        %[[EMPTY_0:.*]] = tensor.empty()
// CHECK-2:        %[[GENERIC_0:.*]] = linalg.generic
// CHECK-2-SAME:       indexing_maps = [#map1, #map]
// CHECK-2-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-2-SAME:       ins(%[[GENERIC]] : tensor<8x2048xf32>)
// CHECK-2-SAME:       outs(%[[EMPTY_0]] : tensor<8x2048x4096xf32>)
// CHECK-2:        ^bb0(%[[IN_0:.*]]: f32, %[[OUT_0:.*]]: f32):
// CHECK-2:          linalg.yield %[[IN_0]] : f32
// CHECK-2:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[EMPTY_1:.*]] = tensor.empty()
// CHECK-2:        %[[GENERIC_1:.*]] = linalg.generic
// CHECK-2-SAME:       indexing_maps = [#map, #map, #map]
// CHECK-2-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-2-SAME:       ins(%[[COLLAPSED_0]], %[[GENERIC_0]] : tensor<8x2048x4096xf32>, tensor<8x2048x4096xf32>)
// CHECK-2-SAME:       outs(%[[EMPTY_1]] : tensor<8x2048x4096xf32>)
// CHECK-2:        ^bb0(%[[IN_1:.*]]: f32, %[[IN_1_0:.*]]: f32, %[[OUT_1:.*]]: f32):
// CHECK-2:          %[[SUBF:.*]] = arith.subf %[[IN_1]], %[[IN_1_0]] : f32
// CHECK-2:          linalg.yield %[[SUBF]] : f32
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[GENERIC_1]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @partial_softmax(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape
