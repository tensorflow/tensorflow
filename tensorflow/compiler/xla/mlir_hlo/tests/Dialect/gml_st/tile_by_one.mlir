// RUN: mlir-hlo-opt %s --gml-tile-by-one | FileCheck %s

func.func @concat(%init : tensor<?x?xi32>, %a: tensor<?x?xi32>,
    %b: tensor<?x?xi32>, %c: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>) dimension = 1
  func.return %concat : tensor<?x?xi32>
}

// CHECK-LABEL:  @concat
// CHECK-SAME:       %[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: tensor<?x?xi32>, %[[ARG2:.*]]: tensor<?x?xi32>, %[[ARG3:.*]]: tensor<?x?xi32>
// CHECK:          scf.for
// CHECK:            scf.for
// CHECK:              scf.if
// CHECK:                tensor.extract_slice %[[ARG1]]
// CHECK:                scf.yield
// CHECK:              else
// CHECK:                scf.if
// CHECK:                  tensor.extract_slice %[[ARG2]]
// CHECK:                  scf.yield
// CHECK:                else
// CHECK:                  tensor.extract_slice %[[ARG3]]
// CHECK:                  scf.yield
// CHECK:                scf.yield
// CHECK:              scf.yield
// CHECK:            scf.yield
// CHECK:          return
