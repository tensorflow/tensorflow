// RUN: mlir-hlo-opt %s --gml-tile-by-one | FileCheck %s

func.func @reverse_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %reversed = thlo.reverse ins(%arg0 : tensor<?x?xf32>)
      outs(%arg1 : tensor<?x?xf32>) reverse_dimensions = [0, 1]
  return %reversed : tensor<?x?xf32>
}

// CHECK:      @reverse_dynamic
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            tensor.extract_slice
// CHECK-SAME:           <?x?xf32> to tensor<1x1xf32>

// -----

func.func @map(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %mapped = linalg.map { math.absf } ins(%arg0 : tensor<?x?xf32>)
      outs(%arg1 : tensor<?x?xf32>)
  return %mapped : tensor<?x?xf32>
}

// CHECK:      @map
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            tensor.extract_slice
// CHECK-SAME:           <?x?xf32> to tensor<1x1xf32>
// CHECK:            linalg.map { math.absf }
// CHECK-SAME:           tensor<1x1xf32>

// -----

func.func @dont_tile_scalarlike_map(%arg0: tensor<1x1xf32>,
    %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %mapped = linalg.map { math.absf } ins(%arg0 : tensor<1x1xf32>)
      outs(%arg1 : tensor<1x1xf32>)
  return %mapped : tensor<1x1xf32>
}

// CHECK:      @dont_tile_scalarlike_map
// CHECK-NOT:    scf.for
// CHECK-NOT:    scf.parallel
// CHECK:        linalg.map
// CHECK-SAME:       tensor<1x1xf32>
// CHECK-NOT:    scf.for
// CHECK-NOT:    scf.parallel
