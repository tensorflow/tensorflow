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

// CHECK: scf.for
// CHECK:   vector.transfer_read
// CHECK:   vector.shuffle
// CHECK:   vector.transfer_write

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

// CHECK: scf.for
// CHECK:   vector.shuffle
// CHECK:   vector.transfer_write

// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     tensor.extract_slice
// CHECK:     tensor.insert_slice

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

// CHECK: scf.for
// CHECK:   tensor.extract_slice {{.*}} [1, 8] [1, 1]

// CHECK: scf.for
// CHECK:   %[[REM_SIZE:.*]] = affine.apply
// CHECK:   tensor.extract_slice {{.*}} [1, %[[REM_SIZE]]] [1, 1]
// CHECK:   tensor.insert_slice {{.*}} tensor<1x?xf32> into tensor<?x?xf32>