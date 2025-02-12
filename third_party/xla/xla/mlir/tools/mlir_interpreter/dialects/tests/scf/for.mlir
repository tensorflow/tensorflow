
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @for() -> memref<4xi64> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<4xi64>
  scf.for %i = %c0 to %c4 step %c2 {
    %1 = arith.index_cast %i: index to i64
    memref.store %1, %alloc[%i]: memref<4xi64>
  }
  return %alloc : memref<4xi64>
}

// CHECK-LABEL: @for
// CHECK: Results
// CHECK-NEXT{LITERAL}: [0, 0, 2, 0]

func.func @nested() -> memref<2x2xindex> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %alloc = memref.alloc() : memref<2x2xindex>
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      memref.store %c1, %alloc[%i, %j]: memref<2x2xindex>
    }
  }
  return %alloc : memref<2x2xindex>
}

// CHECK-LABEL: @nested
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 1], [1, 1]]

func.func @iter_arg() -> index {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%x = %c1) -> index {
    %sum = arith.addi %i, %x : index
    scf.yield %sum : index
  }
  return %sum : index
}

// CHECK-LABEL: @iter_arg
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: i64: 7

func.func @int32() -> memref<4xi32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %alloc = memref.alloc() : memref<4xi32>
  scf.for %i = %c0 to %c4 step %c1 : i32 {
    %index = arith.index_cast %i : i32 to index
    memref.store %i, %alloc[%index]: memref<4xi32>
  }
  return %alloc : memref<4xi32>
}

// CHECK-LABEL: int32
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <4xi32>: [0, 1, 2, 3]

func.func @int16() -> memref<4xi16> {
  %c0 = arith.constant 0 : i16
  %c1 = arith.constant 1 : i16
  %c4 = arith.constant 4 : i16
  %alloc = memref.alloc() : memref<4xi16>
  scf.for %i = %c0 to %c4 step %c1 : i16 {
    %index = arith.index_cast %i : i16 to index
    memref.store %i, %alloc[%index]: memref<4xi16>
  }
  return %alloc : memref<4xi16>
}

// CHECK-LABEL: int16
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: <4xi16>: [0, 1, 2, 3]
