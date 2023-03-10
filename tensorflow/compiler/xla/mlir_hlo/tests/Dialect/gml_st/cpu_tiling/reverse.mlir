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

// CHECK: scf.forall
// CHECK:   vector.shuffle
// CHECK:   tensor.parallel_insert_slice

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

// CHECK: scf.forall
// CHECK:   vector.shuffle
// CHECK:   tensor.parallel_insert_slice

// CHECK: scf.forall
// CHECK:   scf.forall
// CHECK:     tensor.extract_slice
// CHECK:     tensor.parallel_insert_slice
// CHECK:   tensor.parallel_insert_slice

// -----

func.func @reverse_dynamic_not_last_dim(
  %input: tensor<?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %res = thlo.reverse
     ins(%input: tensor<?x?xf32>)
     outs(%init: tensor<?x?xf32>)
     reverse_dimensions = [0]
  func.return %res : tensor<?x?xf32>
}

// CHECK-LABEL: @reverse_dynamic

// CHECK: scf.forall
// CHECK:   tensor.extract_slice {{.*}} [1, 8] [1, 1]

// CHECK: scf.forall
// CHECK:   %[[REM_SIZE:.*]] = affine.apply
// CHECK:   tensor.extract_slice {{.*}} [1, %[[REM_SIZE]]] [1, 1]
// CHECK-NOT:   scf.forall
// CHECK: scf.forall.in_parallel
