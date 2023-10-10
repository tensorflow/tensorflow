// RUN: mlir-hlo-opt %s --gml-st-rewrite-forall-ops --split-input-file \
// RUN: | FileCheck %s

func.func @add(%in: tensor<3x3xi32>, %out: tensor<3x3xi32>) -> tensor<3x3xi32> {
  %c3 = arith.constant 3 : index

  %result = scf.forall (%i, %j) in (%c3, %c3)
      shared_outs(%o = %out) -> tensor<3x3xi32> {
    %addend = tensor.extract_slice %in[%i, %j][1, 1][1, 1]
        : tensor<3x3xi32> to tensor<i32>
    %augend = tensor.extract_slice %out[%i, %j][1, 1][1, 1]
        : tensor<3x3xi32> to tensor<i32>
    %sum = mhlo.add %augend, %addend : tensor<i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %sum into %o[%i, %j][1, 1][1, 1]
        : tensor<i32> into tensor<3x3xi32>
    }
  } {some_attr = "attr_value"}

  return %result : tensor<3x3xi32>
}

// CHECK-LABEL: @add
//      CHECK: %[[RESULT:.*]] = scf.for
// CHECK-NEXT:   %[[INNER:.*]] = scf.for
// CHECK-NEXT:     tensor.extract_slice
// CHECK-NEXT:     tensor.extract_slice
// CHECK-NEXT:     mhlo.add
// CHECK-NEXT:     %[[INSERTED:.*]] = tensor.insert_slice
// CHECK-NEXT:     scf.yield %[[INSERTED]]
// CHECK-NEXT:   } {some_attr = "attr_value"}
// CHECK-NEXT:   scf.yield %[[INNER]]
// CHECK-NEXT: } {some_attr = "attr_value"}
// CHECK-NEXT: return %[[RESULT]]

// -----

func.func @bufferized_add() -> memref<3xi32> {
  %c3 = arith.constant 3 : index
  %in = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %out = arith.constant dense<[4, 5, 6]> : memref<3xi32>

  scf.forall (%i) in (%c3) {
    %addend = tensor.extract %in[%i] : tensor<3xi32>
    %augend = memref.load %out[%i] : memref<3xi32>
    %sum = arith.addi %augend, %addend : i32
    memref.store %sum, %out[%i] : memref<3xi32>
  }

  return %out : memref<3xi32>
}

// CHECK-LABEL: @bufferized_add
//  CHECK-DAG: %[[C0:.*]] = arith.constant 0
//  CHECK-DAG: %[[C1:.*]] = arith.constant 1
//  CHECK-DAG: %[[C3:.*]] = arith.constant 3
//      CHECK: scf.for {{.*}} = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-NEXT:   tensor.extract
// CHECK-NEXT:   memref.load
// CHECK-NEXT:   arith.addi
// CHECK-NEXT:   memref.store
// CHECK-NEXT: }
