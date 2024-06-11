// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

#map = affine_map<(d0)[] -> (1000, d0 + 512, d0*100)>

func.func @min() -> (index, index, index) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c500 = arith.constant 500 : index

  %0 = affine.min #map (%c1)[]
  %1 = affine.min #map (%c8)[]
  %2 = affine.min #map (%c500)[]

  return %0, %1, %2 : index, index, index
}

// CHECK-LABEL: @min
// CHECK-NEXT: Results
// CHECK-NEXT: 100
// CHECK-NEXT: 520
// CHECK-NEXT: 1000

func.func @max() -> (index, index) {
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index

  %0 = affine.max #map (%c1)[]
  %1 = affine.max #map (%c11)[]

  return %0, %1 : index, index
}

// CHECK-LABEL: @max
// CHECK-NEXT: Results
// CHECK-NEXT: 1000
// CHECK-NEXT: 1100
