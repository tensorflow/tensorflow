// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

// Adapted from
// mlir/test/Integration/Dialect/Vector/CPU/test-contraction.mlir

#dotp_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dotp_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"]
}

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

#mattransvec_accesses = [
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#mattransvec_trait = {
  indexing_maps = #mattransvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#column_major_matmat_accesses = [
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (j, i)>
]
#column_major_matmat_trait = {
  indexing_maps = #column_major_matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @dot_products() -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %a = arith.constant dense<[1, 2]> : vector<2xi32>
  %b = arith.constant dense<[3, 4]> : vector<2xi32>
  // Contraction: dot-product a x b
  %dp1 = vector.contract #dotp_trait %a, %b, %c0
    : vector<2xi32>, vector<2xi32> into i32
  %dp2 = vector.contract #dotp_trait %a, %b, %c1
    : vector<2xi32>, vector<2xi32> into i32
  return %dp1, %dp2 : i32, i32
}

// CHECK-LABEL: @dot_product
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 11
// CHECK-NEXT: i32: 12

func.func @matrix_vector() -> (vector<2xi32>, vector<2xi32>) {
  %z1 = arith.constant dense<0> : vector<2xi32>
  %a = arith.constant dense<[1, 2]> : vector<2xi32>
  %c = arith.constant dense<[5, 6]> : vector<2xi32>
  %A = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %mv1 = vector.contract #matvec_trait %A, %c, %z1
    : vector<2x2xi32>, vector<2xi32> into vector<2xi32>
  %mv2 = vector.contract #matvec_trait %A, %c, %a
    : vector<2x2xi32>, vector<2xi32> into vector<2xi32>
  return %mv1, %mv2 : vector<2xi32>, vector<2xi32>
}

// CHECK-LABEL: @matrix_vector
// CHECK-NEXT: Results
// CHECK-NEXT: [17, 39]
// CHECK-NEXT: [18, 41]

func.func @matrix_trans_vector() -> (vector<2xi32>, vector<2xi32>) {
  %z1 = arith.constant dense<0> : vector<2xi32>
  %a = arith.constant dense<[1, 2]> : vector<2xi32>
  %c = arith.constant dense<[5, 6]> : vector<2xi32>
  %A = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %mv1 = vector.contract #mattransvec_trait %A, %c, %z1
    : vector<2x2xi32>, vector<2xi32> into vector<2xi32>
  %mv2 = vector.contract #mattransvec_trait %A, %c, %a
    : vector<2x2xi32>, vector<2xi32> into vector<2xi32>
  return %mv1, %mv2 : vector<2xi32>, vector<2xi32>
}

// CHECK-LABEL: @matrix_trans_vector
// CHECK-NEXT: Results
// CHECK-NEXT: [23, 34]
// CHECK-NEXT: [24, 36]

func.func @matrix_matrix() -> (vector<2x2xi32>, vector<2x2xi32>) {
  %z2 = arith.constant dense<0> : vector<2x2xi32>
  %A = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %B = arith.constant dense<[[5, 6], [7, 8]]> : vector<2x2xi32>
  %mm1 = vector.contract #matmat_trait %A, %B, %z2
    : vector<2x2xi32>, vector<2x2xi32> into vector<2x2xi32>
  %mm2 = vector.contract #matmat_trait %A, %B, %A
    : vector<2x2xi32>, vector<2x2xi32> into vector<2x2xi32>
  return %mm1, %mm2 : vector<2x2xi32>, vector<2x2xi32>
}

// CHECK-LABEL: @matrix_matrix
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[19, 22], [43, 50]]
// CHECK-NEXT{LITERAL}: [[20, 24], [46, 54]]

func.func @matrix_matrix_column_major() -> (vector<2x2xi32>, vector<2x2xi32>) {
  %z2 = arith.constant dense<0> : vector<2x2xi32>
  %A = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %B = arith.constant dense<[[5, 6], [7, 8]]> : vector<2x2xi32>
  %llvm_matrix_column_major_mm0 =
    vector.contract #column_major_matmat_trait %A, %B, %z2
      : vector<2x2xi32>, vector<2x2xi32> into vector<2x2xi32>
  %llvm_matrix_column_major_mm1 =
    vector.contract #column_major_matmat_trait %A, %B, %A
      : vector<2x2xi32>, vector<2x2xi32> into vector<2x2xi32>
  return %llvm_matrix_column_major_mm0, %llvm_matrix_column_major_mm1 :
    vector<2x2xi32>, vector<2x2xi32>
}

// CHECK-LABEL: @matrix_matrix_column_major
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[23, 31], [34, 46]]
// CHECK-NEXT{LITERAL}: [[24, 33], [37, 50]]
