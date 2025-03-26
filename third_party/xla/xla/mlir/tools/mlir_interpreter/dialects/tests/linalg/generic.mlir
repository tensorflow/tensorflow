// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul() -> (tensor<2x2xi32>, tensor<2x2xi32>) {
  %lhs = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %rhs = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %init = tensor.empty() : tensor<2x2xi32>
  %ret = linalg.generic #matmul_trait
    ins(%lhs, %rhs : tensor<2x3xi32>, tensor<3x2xi32>)
    outs(%init : tensor<2x2xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32):
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
    } -> tensor<2x2xi32>
  return %ret, %init : tensor<2x2xi32>, tensor<2x2xi32>
}

// CHECK-LABEL: @matmul
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]
// CHECK-NEXT{LITERAL}: [[0, 0], [0, 0]]

func.func @bufferized() -> memref<2x2xi32> {
  %lhs = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : memref<2x3xi32>
  %rhs = arith.constant dense<[[2, 1], [3, 4], [5, 6]]> : memref<3x2xi32>
  %alloc = memref.alloc() : memref<2x2xi32>
  linalg.generic #matmul_trait
    ins(%lhs, %rhs : memref<2x3xi32>, memref<3x2xi32>)
    outs(%alloc : memref<2x2xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32):
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
    }
  return %alloc : memref<2x2xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[23, 27], [53, 60]]

#map = affine_map<(d0) -> (d0)>

func.func @vector() -> tensor<4xvector<2xi32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %lhs = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %rhs = arith.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
  %init = tensor.empty() : tensor<4xvector<2xi32>>
  %ret = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : tensor<4xi32>, tensor<4xi32>)
    outs(%init : tensor<4xvector<2xi32>>) {
    ^bb(%a: i32, %b: i32, %c: vector<2xi32>):
      %d = vector.insertelement %a, %c[%c0 : index] : vector<2xi32>
      %e = vector.insertelement %b, %d[%c1 : index] : vector<2xi32>
      linalg.yield %e : vector<2xi32>
    } -> tensor<4xvector<2xi32>>
  return %ret : tensor<4xvector<2xi32>>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<4xvector<2xi32>>: [[1, 5], [2, 6], [3, 7], [4, 8]]

func.func @matmul_dynamic() -> tensor<2x2xi32> {
  %lhs = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %rhs = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %lhs_cast = tensor.cast %lhs : tensor<2x3xi32> to tensor<2x?xi32>
  %rhs_cast = tensor.cast %rhs : tensor<3x2xi32> to tensor<?x2xi32>
  %init = tensor.empty() : tensor<2x2xi32>
  %ret = linalg.generic #matmul_trait
    ins(%lhs_cast, %rhs_cast : tensor<2x?xi32>, tensor<?x2xi32>)
    outs(%init : tensor<2x2xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32):
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
    } -> tensor<2x2xi32>
  return %ret : tensor<2x2xi32>
}

// CHECK-LABEL: @matmul_dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]

func.func @dynamic_generic_w_cst() -> tensor<4x?xf64> {
  %cst_1 =  arith.constant 123.456 : f64
  %extracted_slice_ = arith.constant dense<[[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8]]> : tensor<4x2xf64>
  %extracted_slice = tensor.cast %extracted_slice_ : tensor<4x2xf64> to tensor<4x?xf64>
  %6 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> ()>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel",
      "parallel"]} ins(%cst_1 : f64) outs(%extracted_slice : tensor<4x?xf64>) {
  ^bb0(%in: f64, %out: f64):
    linalg.yield %in : f64
  } -> tensor<4x?xf64>
  return %6 : tensor<4x?xf64>
}

// CHECK-LABEL: @dynamic_generic_w_cst
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<4x2xf64>: [[1.234560e+02, 1.234560e+02], [1.234560e+02, 1.234560e+02], [1.234560e+02, 1.234560e+02], [1.234560e+02, 1.234560e+02]]
