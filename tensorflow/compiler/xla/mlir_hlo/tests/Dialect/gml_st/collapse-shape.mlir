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

func.func @bcast(%arg0: tensor<2x4x2048xf32>) -> tensor<2x4x2048x4096xf32> {
  %0 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %1 = linalg.broadcast
        ins(%arg0 : tensor<2x4x2048xf32>)
        outs(%0 : tensor<2x4x2048x4096xf32>)
        dimensions = [3]
  return %1 : tensor<2x4x2048x4096xf32>
}

// CHECK:        func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-NOT:    collapse_shape
// CHECK-NOT:    expand_shape

// CHECK-1:      func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[BROADCAST:.*]] = linalg.broadcast
// CHECK-1:      ins(%[[COLLAPSED]] : tensor<16384xf32>)
// CHECK-1:      outs(%[[EMPTY]] : tensor<16384x4096xf32>)
// CHECK-1:      dimensions = [1]
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCAST]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-2:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2]]
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-2:        %[[BROADCASTED:.*]] = linalg.broadcast
// CHECK-2:      ins(%[[COLLAPSED]] : tensor<8x2048xf32>)
// CHECK-2:      outs(%[[EMPTY]] : tensor<8x2048x4096xf32>)
// CHECK-2:      dimensions = [2]
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCASTED]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @bcast(%[[ARG0:.*]]: tensor<2x4x2048xf32>)
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----

func.func @bcast_from_scalar() -> tensor<2x4x2048x4096xf32> {
  %0 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %cst = arith.constant 0xFF800000 : f32
  %1 = tensor.empty() : tensor<f32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<f32>) -> tensor<f32>
  %3 = linalg.broadcast
      ins(%2 : tensor<f32>)
      outs(%0 : tensor<2x4x2048x4096xf32>)
      dimensions = [0, 1, 2, 3]
  return %3 : tensor<2x4x2048x4096xf32>
}

// CHECK:      func.func @bcast_from_scalar()
// CHECK:        %[[EMPTY:.*]] = tensor.empty() : tensor<67108864xf32>
// CHECK:        %[[BROADCAST:.*]] = linalg.broadcast
// CHECK:           ins(%{{.*}} : tensor<f32>)
// CHECK:           outs(%[[EMPTY]] : tensor<67108864xf32>)
// CHECK:           dimensions = [0]
// CHECK:        %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCAST]] [
// CHECK-SAME:     0, 1, 2, 3]]
// CHECK:        return %[[EXPANDED]]

// CHECK-1:      func.func @bcast_from_scalar()
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty() : tensor<16384x4096xf32>
// CHECK-1:        %[[BROADCAST:.*]] = linalg.broadcast
// CHECK-1-NEXT:       ins(%{{.*}} : tensor<f32>)
// CHECK-1-NEXT:       outs(%[[EMPTY]] : tensor<16384x4096xf32>)
// CHECK-1-NEXT:       dimensions = [1, 0]  
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCAST]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      func.func @bcast_from_scalar()
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty() : tensor<8x2048x4096xf32>
// CHECK-2:        %[[BROADCAST:.*]] = linalg.broadcast
// CHECK-2-NEXT:       ins(%{{.*}} : tensor<f32>
// CHECK-2-NEXT:       outs(%[[EMPTY]] : tensor<8x2048x4096xf32>)
// CHECK-2-NEXT:       dimensions = [1, 2, 0]
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[BROADCAST]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @bcast_from_scalar()
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----

func.func @reduction(%arg0: tensor<2x4x2048x4096xf32>) -> tensor<2x4x2048xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2x4x2048xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x2048xf32>)
      -> tensor<2x4x2048xf32>
  %2 = linalg.reduce
      ins(%arg0 : tensor<2x4x2048x4096xf32>)
      outs(%1 : tensor<2x4x2048xf32>)
      dimensions = [3]
      (%in: f32, %out: f32) {
        %3 = arith.maxf %out, %in : f32
        linalg.yield %3 : f32
      }
  return %2 : tensor<2x4x2048xf32>
}

// CHECK:        func.func @reduction(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-NOT:    collapse_shape
// CHECK-NOT:    expand_shape

// CHECK-1:      func.func @reduction(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-1-DAG:    %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<16384xf32>)
// CHECK-1:        %[[REDUCED:.*]] = linalg.reduce
// CHECK-1-NEXT:       ins(%[[COLLAPSED]] : tensor<16384x4096xf32>)
// CHECK-1-NEXT:       outs(%[[FILL]] : tensor<16384xf32>)
// CHECK-1:        (%[[IN:.*]]: f32, %[[OUT:.*]]: f32) {
// CHECK-1:          %[[MAXF:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
// CHECK-1:          linalg.yield %[[MAXF]] : f32
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[REDUCED]] [
// CHECK-1-SAME:     [0, 1, 2]]
// CHECK-1:        return %[[EXPANDED]]


// -----

func.func @cwise(%arg0: tensor<2x4x2048x4096xf32>,
    %arg1: tensor<2x4x2048x4096xf32>) -> tensor<2x4x2048x4096xf32> {
  %0 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %1 = linalg.map
      ins(%arg0, %arg1 : tensor<2x4x2048x4096xf32>, tensor<2x4x2048x4096xf32>)
      outs(%0 : tensor<2x4x2048x4096xf32>)
    (%in: f32, %in_0: f32) {
       %2 = arith.subf %in, %in_0 : f32
       linalg.yield %2 : f32
    }
  return %1 : tensor<2x4x2048x4096xf32>
}

// CHECK:        func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK:          %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-SAME:       [0, 1, 2, 3]]
// CHECK:          %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG1]] [
// CHECK-SAME:       [0, 1, 2, 3]]
// CHECK:          %[[EMPTY:.*]] = tensor.empty()
// CHECK:          %[[MAP:.*]] = linalg.map
// CHECK:           ins(%[[COLLAPSED]], %[[COLLAPSED_0]] : tensor<67108864xf32>, tensor<67108864xf32>)
// CHECK:           outs(%[[EMPTY]] : tensor<67108864xf32>)
// CHECK:           (%[[IN:.*]]: f32, %[[IN_1:.*]]: f32) {
// CHECK:              %[[SUBF:.*]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:              linalg.yield %[[SUBF]] : f32
// CHECK:          %[[EXPANDED:.*]] = tensor.expand_shape %[[MAP]] [
// CHECK-SAME:       [0, 1, 2, 3]]
// CHECK:          return %[[EXPANDED]]

// CHECK-1:      func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG1]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[MAP:.*]] = linalg.map
// CHECK-1-NEXT:       ins(%[[COLLAPSED]], %[[COLLAPSED_0]] : tensor<16384x4096xf32>, tensor<16384x4096xf32>)
// CHECK-1-NEXT:       outs(%[[EMPTY]] : tensor<16384x4096xf32>)
// CHECK-1:        (%[[IN:.*]]: f32, %[[IN_1:.*]]: f32) {
// CHECK-1:          %[[SUBF:.*]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-1:          linalg.yield %[[SUBF]] : f32
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[MAP]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-2:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG1]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-2:        %[[MAP:.*]] = linalg.map
// CHECK-2-NEXT:       ins(%[[COLLAPSED]], %[[COLLAPSED_0]] : tensor<8x2048x4096xf32>, tensor<8x2048x4096xf32>)
// CHECK-2-NEXT:       outs(%[[EMPTY]] : tensor<8x2048x4096xf32>)
// CHECK-2:        (%[[IN:.*]]: f32, %[[IN_1:.*]]: f32) {
// CHECK-2:          %[[SUBF:.*]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-2:          linalg.yield %[[SUBF]] : f32
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[MAP]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @cwise(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>, %[[ARG1:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----

func.func @partial_softmax(%arg0: tensor<2x4x2048x4096xf32>)
    -> tensor<2x4x2048x4096xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<2x4x2048xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4x2048xf32>)
      -> tensor<2x4x2048xf32>
  %2 = linalg.reduce
      ins(%arg0 : tensor<2x4x2048x4096xf32>)
      outs(%1 : tensor<2x4x2048xf32>)
      dimensions = [3]
  (%in: f32, %out: f32) {
    %6 = arith.maxf %out, %in : f32
    linalg.yield %6 : f32
  }
  %3 = tensor.empty() : tensor<2x4x2048x4096xf32>
  %4 = linalg.broadcast
      ins(%2 : tensor<2x4x2048xf32>)
      outs(%3 : tensor<2x4x2048x4096xf32>)
      dimensions = [3]
  %5 = linalg.map
      ins(%arg0, %4 : tensor<2x4x2048x4096xf32>, tensor<2x4x2048x4096xf32>)
      outs(%3 : tensor<2x4x2048x4096xf32>)
  (%in: f32, %in_0: f32) {
    %6 = arith.subf %in, %in_0 : f32
    linalg.yield %6 : f32
  }
  return %5 : tensor<2x4x2048x4096xf32>
}

// CHECK-1:      func.func @partial_softmax(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-1-DAG:    %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-1:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-1:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<16384xf32>)
// CHECK-1:        %[[REDUCE:.*]] = linalg.reduce
// CHECK-1-NEXT:       ins(%[[COLLAPSED]] : tensor<16384x4096xf32>)
// CHECK-1-NEXT:       outs(%[[FILL]] : tensor<16384xf32>)
// CHECK-1-NEXT:       dimensions = [1]
// CHECK-1:        (%[[IN:.*]]: f32, %[[OUT:.*]]: f32) {
// CHECK-1:          %[[MAXF:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
// CHECK-1:          linalg.yield %[[MAXF]] : f32
// CHECK-1:        %[[EMPTY_0:.*]] = tensor.empty()
// CHECK-1:        %[[BROADCAST:.*]] = linalg.broadcast
// CHECK-1-NEXT:       ins(%[[REDUCE]] : tensor<16384xf32>)
// CHECK-1-NEXT:       outs(%[[EMPTY_0]] : tensor<16384x4096xf32>)
// CHECK-1-NEXT:       dimensions = [1]
// CHECK-1:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        %[[EMPTY_1:.*]] = tensor.empty()
// CHECK-1:        %[[MAP:.*]] = linalg.map
// CHECK-1-NEXT:       ins(%[[COLLAPSED_0]], %[[BROADCAST]] : tensor<16384x4096xf32>, tensor<16384x4096xf32>)
// CHECK-1-NEXT:       outs(%[[EMPTY_1]] : tensor<16384x4096xf32>)
// CHECK-1:        (%[[IN_1:.*]]: f32, %[[IN_1_0:.*]]: f32) {
// CHECK-1:          %[[SUBF:.*]] = arith.subf %[[IN_1]], %[[IN_1_0]] : f32
// CHECK-1:          linalg.yield %[[SUBF]] : f32
// CHECK-1:        %[[EXPANDED:.*]] = tensor.expand_shape %[[MAP]] [
// CHECK-1-SAME:     [0, 1, 2], [3]]
// CHECK-1:        return %[[EXPANDED]]

// CHECK-2:      func.func @partial_softmax(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-2-DAG:    %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-2:        %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[EMPTY:.*]] = tensor.empty()
// CHECK-2:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<8x2048xf32>)
// CHECK-2:        %[[REDUCE:.*]] = linalg.reduce
// CHECK-2-NEXT:       ins(%[[COLLAPSED]] : tensor<8x2048x4096xf32>)
// CHECK-2-NEXT:       outs(%[[FILL]] : tensor<8x2048xf32>)
// CHECK-2-NEXT:       dimensions = [2]
// CHECK-2:        (%[[IN:.*]]: f32, %[[OUT:.*]]: f32) {
// CHECK-2:          %[[MAXF:.*]] = arith.maxf %[[OUT]], %[[IN]] : f32
// CHECK-2:          linalg.yield %[[MAXF]] : f32
// CHECK-2:        %[[EMPTY_0:.*]] = tensor.empty()
// CHECK-2:        %[[BROADCAST:.*]] = linalg.broadcast
// CHECK-2-NEXT:       ins(%[[REDUCE]] : tensor<8x2048xf32>)
// CHECK-2-NEXT:       outs(%[[EMPTY_0]] : tensor<8x2048x4096xf32>)
// CHECK-2-NEXT:       dimensions = [2]
// CHECK-2:        %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[ARG0]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        %[[EMPTY_1:.*]] = tensor.empty()
// CHECK-2:        %[[MAP:.*]] = linalg.map
// CHECK-2-NEXT:       ins(%[[COLLAPSED_0]], %[[BROADCAST]] : tensor<8x2048x4096xf32>, tensor<8x2048x4096xf32>)
// CHECK-2-NEXT:       outs(%[[EMPTY_1]] : tensor<8x2048x4096xf32>)
// CHECK-2:        (%[[IN_1:.*]]: f32, %[[IN_1_0:.*]]: f32) {
// CHECK-2:          %[[SUBF:.*]] = arith.subf %[[IN_1]], %[[IN_1_0]] : f32
// CHECK-2:          linalg.yield %[[SUBF]] : f32
// CHECK-2:        %[[EXPANDED:.*]] = tensor.expand_shape %[[MAP]] [
// CHECK-2-SAME:     [0, 1], [2], [3]]
// CHECK-2:        return %[[EXPANDED]]

// CHECK-3:        func.func @partial_softmax(%[[ARG0:.*]]: tensor<2x4x2048x4096xf32>)
// CHECK-3-NOT:    collapse_shape
// CHECK-3-NOT:    expand_shape

// -----


func.func @collapse_shape_of_cwise(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = linalg.map
      ins(%arg0 : tensor<2x4xf32>)
      outs(%0 : tensor<2x4xf32>)
    (%in: f32) {
       %2 = arith.negf %in : f32
       linalg.yield %2 : f32
    }
  %3 = tensor.collapse_shape %1 [[0, 1]] : tensor<2x4xf32> into tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK:   func.func @collapse_shape_of_cwise
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape {{.*}} [
// CHECK-SAME: [0, 1]] : tensor<2x4xf32> into tensor<8xf32>
// CHECK: %[[MAPPED:.*]] = linalg.map
// CHECK: ins(%[[COLLAPSED]] : tensor<8xf32>)

// CHECK-1: func.func @collapse_shape_of_cwise
// CHECK-2: func.func @collapse_shape_of_cwise
// CHECK-3: func.func @collapse_shape_of_cwise

