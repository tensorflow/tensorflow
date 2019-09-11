// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.CompositeExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: extract_vector
func @extract_vector() -> (i32, i32, i32) {
  // CHECK: spv.constant 42 : i32
  // CHECK: spv.constant -33 : i32
  // CHECK: spv.constant 6 : i32
  %0 = spv.constant dense<[42, -33, 6]> : vector<3xi32>
  %1 = spv.CompositeExtract %0[0 : i32] : vector<3xi32>
  %2 = spv.CompositeExtract %0[1 : i32] : vector<3xi32>
  %3 = spv.CompositeExtract %0[2 : i32] : vector<3xi32>
  return %1, %2, %3 : i32, i32, i32
}

// -----

// CHECK-LABEL: extract_array_final
func @extract_array_final() -> (i32, i32) {
  // CHECK: spv.constant 4 : i32
  // CHECK: spv.constant -5 : i32
  %0 = spv.constant [dense<[4, -5]> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.CompositeExtract %0[0 : i32, 0 : i32] : !spv.array<1 x vector<2 x i32>>
  %2 = spv.CompositeExtract %0[0 : i32, 1 : i32] : !spv.array<1 x vector<2 x i32>>
  return %1, %2 : i32, i32
}

// -----

// CHECK-LABEL: extract_array_interm
func @extract_array_interm() -> (vector<2xi32>) {
  // CHECK: spv.constant dense<[4, -5]> : vector<2xi32>
  %0 = spv.constant [dense<[4, -5]> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<1 x vector<2 x i32>>
  return %1 : vector<2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

// TODO(antiagainst): test constants in different blocks

func @deduplicate_scalar_constant() -> (i32, i32) {
  // CHECK: %[[CST:.*]] = spv.constant 42 : i32
  %0 = spv.constant 42 : i32
  %1 = spv.constant 42 : i32
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : i32, i32
}

// -----

func @deduplicate_vector_constant() -> (vector<3xi32>, vector<3xi32>) {
  // CHECK: %[[CST:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %0 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : vector<3xi32>, vector<3xi32>
}

// -----

func @deduplicate_composite_constant() -> (!spv.array<1 x vector<2xi32>>, !spv.array<1 x vector<2xi32>>) {
  // CHECK: %[[CST:.*]] = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %0 = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : !spv.array<1 x vector<2xi32>>, !spv.array<1 x vector<2xi32>>
}

