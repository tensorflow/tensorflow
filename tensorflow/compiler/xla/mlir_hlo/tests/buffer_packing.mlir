// RUN: mlir-hlo-opt -buffer-packing -split-input-file %s | FileCheck %s

// CHECK-LABEL: @noPackingSameLiveRange
func.func @noPackingSameLiveRange() -> (f32, f32) {
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<42xf32>
  memref.store %c2, %0[%c1] : memref<42xf32>
  memref.store %c2, %1[%c1] : memref<42xf32>
  %2 = memref.load %0[%c1] : memref<42xf32>
  %3 = memref.load %1[%c1] : memref<42xf32>
  return %2, %3 : f32, f32
}

// -----

// CHECK-LABEL: @packingScfIfSameSize
func.func @packingScfIfSameSize(%pred : i1) -> (f32) {
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<192xi8>
  // CHECK: %[[VIEW1:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<42xf32>
  // CHECK: %[[VIEW2:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<42xf32>
  // CHECK: scf.if
  // CHECK: memref.load %[[VIEW1]]
  // CHECK: else
  // CHECK: memref.load %[[VIEW2]]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<42xf32>
  %2 = scf.if %pred -> f32 {
    memref.store %c2, %0[%c1] : memref<42xf32>
    %2 = memref.load %0[%c1] : memref<42xf32>
    scf.yield %2 : f32
  } else {
    memref.store %c2, %1[%c1] : memref<42xf32>
    %2 = memref.load %1[%c1] : memref<42xf32>
    scf.yield %2 : f32
  }
  return %2 : f32
}

// -----

// CHECK-LABEL: @packingScfIfDifferentSize
func.func @packingScfIfDifferentSize(%pred : i1) -> (f32) {
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<192xi8>
  // CHECK: scf.if
  // CHECK: %[[VIEW1:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<42xf32>
  // CHECK: memref.load %[[VIEW1]]
  // CHECK: else
  // CHECK: %[[VIEW2:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<16xf32>
  // CHECK: memref.load %[[VIEW2]]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32
  %0 = scf.if %pred -> f32 {
    %0 = memref.alloc() : memref<42xf32>
    memref.store %c2, %0[%c1] : memref<42xf32>
    %1 = memref.load %0[%c1] : memref<42xf32>
    scf.yield %1 : f32
  } else {
    %0 = memref.alloc() : memref<16xf32>
    memref.store %c2, %0[%c1] : memref<16xf32>
    %1 = memref.load %0[%c1] : memref<16xf32>
    scf.yield %1 : f32
  }
  return %0 : f32
}

// -----

// CHECK-LABEL: @packingScfIfDifferentElementType
func.func @packingScfIfDifferentElementType(%pred : i1) -> (f32) {
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<128xi8>
  // CHECK: scf.if
  // CHECK: %[[VIEW1:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<128xi8> to memref<42xf16>
  // CHECK: memref.load %[[VIEW1]]
  // CHECK: else
  // CHECK: %[[VIEW2:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<128xi8> to memref<16xf32>
  // CHECK: memref.load %[[VIEW2]]
  %c1 = arith.constant 1 : index
  %0 = scf.if %pred -> f32 {
    %c2 = arith.constant 2.0 : f16
    %0 = memref.alloc() : memref<42xf16>
    memref.store %c2, %0[%c1] : memref<42xf16>
    %1 = memref.load %0[%c1] : memref<42xf16>
    %2 = arith.extf %1 : f16 to f32
    scf.yield %2 : f32
  } else {
    %c2 = arith.constant 2.0 : f32
    %0 = memref.alloc() : memref<16xf32>
    memref.store %c2, %0[%c1] : memref<16xf32>
    %1 = memref.load %0[%c1] : memref<16xf32>
    scf.yield %1 : f32
  }
  return %0 : f32
}

// -----

// CHECK-LABEL: @packWithOutsideControlFlow
func.func @packWithOutsideControlFlow(%pred : i1) -> (f32, f32) {
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<192xi8>
  // CHECK: %[[VIEW0:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<42xf32>
  // CHECK: memref.load %[[VIEW0]]
  // CHECK: scf.if
  // CHECK: %[[VIEW1:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<42xf32>
  // CHECK: memref.load %[[VIEW1]]
  // CHECK: else
  // CHECK: %[[VIEW2:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<42xf32>
  // CHECK: memref.load %[[VIEW2]]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32
  %0 = memref.alloc() : memref<42xf32>
  memref.store %c2, %0[%c1] : memref<42xf32>
  %1 = memref.load %0[%c1] : memref<42xf32>
  %2 = scf.if %pred -> f32 {
    %3 = memref.alloc() : memref<42xf32>
    memref.store %c2, %3[%c1] : memref<42xf32>
    %4 = memref.load %3[%c1] : memref<42xf32>
    scf.yield %4 : f32
  } else {
    %3 = memref.alloc() : memref<42xf32>
    memref.store %c2, %3[%c1] : memref<42xf32>
    %4 = memref.load %3[%c1] : memref<42xf32>
    scf.yield %4 : f32
  }
  return %1, %2 : f32, f32
}

// -----

// CHECK-LABEL: @packTwoInOne
func.func @packTwoInOne(%pred : i1) -> (f32) {
  // CHECK: %[[MEM:.*]] = memref.alloc() : memref<192xi8>
  // CHECK: scf.if
  // CHECK: %[[VIEW1:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<42xf32>
  // CHECK: memref.load %[[VIEW1]]
  // CHECK: else
  // CHECK: %[[VIEW2:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<16xf32>
  // CHECK: %[[VIEW3:.*]] = memref.view %[[MEM]][%{{.*}}][] : memref<192xi8> to memref<8xf32>
  // CHECK: memref.load %[[VIEW2]]
  // CHECK: memref.load %[[VIEW3]]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32
  %0 = scf.if %pred -> f32 {
    %0 = memref.alloc() : memref<42xf32>
    memref.store %c2, %0[%c1] : memref<42xf32>
    %1 = memref.load %0[%c1] : memref<42xf32>
    scf.yield %1 : f32
  } else {
    %0 = memref.alloc() : memref<16xf32>
    %1 = memref.alloc() : memref<8xf32>
    memref.store %c2, %0[%c1] : memref<16xf32>
    %2 = memref.load %0[%c1] : memref<16xf32>
    memref.store %c2, %1[%c1] : memref<8xf32>
    %3 = memref.load %1[%c1] : memref<8xf32>
    %4 = arith.addf %2, %3 : f32
    scf.yield %4 : f32
  }
  return %0 : f32
}
