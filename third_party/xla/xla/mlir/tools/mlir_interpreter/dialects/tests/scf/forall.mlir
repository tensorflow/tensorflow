
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @add() -> (tensor<3xi32>, tensor<3xi32>) {
  %c3 = arith.constant 3 : index
  %in = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %out = arith.constant dense<[4, 5, 6]> : tensor<3xi32>

  %result = scf.forall (%i) in (%c3) shared_outs(%o = %out) -> tensor<3xi32> {
    %addend = tensor.extract_slice %in[%i][1][1] : tensor<3xi32> to tensor<1xi32>
    %augend = tensor.extract_slice %out[%i][1][1] : tensor<3xi32> to tensor<1xi32>
    %sum = mhlo.add %augend, %addend : tensor<1xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %sum into %o[%i][1][1]
        : tensor<1xi32> into tensor<3xi32>
    }
  }

  return %out, %result : tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: @add
// CHECK-NEXT: Results
// CHECK-NEXT: [4, 5, 6]
// CHECK-NEXT: [5, 7, 9]

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
// CHECK-NEXT: Results
// CHECK-NEXT: [5, 7, 9]

func.func @step() -> memref<4xi32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  %c42 = arith.constant 42 : i32
  %out = arith.constant dense<[1, 1, 1, 1]> : memref<4xi32>

  scf.forall (%i) = (0) to (%c4) step (%c2) {
    memref.store %c42, %out[%i] : memref<4xi32>
  }

  return %out : memref<4xi32>
}

// CHECK-LABEL: @step
// CHECK-NEXT: Results
// CHECK-NEXT: [42, 1, 42, 1]
