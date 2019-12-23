// RUN: tf-opt %s -split-input-file -xla-legalize-to-std | FileCheck %s

// -----

// CHECK-LABEL: func @iota.const.1() -> tensor<4xi32> {
func @iota.const.1() -> tensor<4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %0 = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @iota.const.2() -> tensor<2x4xi32> {
func @iota.const.2() -> tensor<2x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[}}0, 0, 0, 0], [1, 1, 1, 1]]> : tensor<2x4xi32>
  %0 = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: func @iota.const.3() -> tensor<2x4xi32> {
func @iota.const.3() -> tensor<2x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[}}0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<2x4xi32>
  %0 = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<2x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: func @iota.const.4() -> tensor<2x3x4xi32> {
func @iota.const.4() -> tensor<2x3x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0{{\]\]}}, {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]> : tensor<2x3x4xi32>
  %0 = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2x3x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// -----

// CHECK-LABEL: func @iota.const.5() -> tensor<2x3x4xi32> {
func @iota.const.5() -> tensor<2x3x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[\[}}0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2{{\]\]}}, {{\[\[}}0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]]> : tensor<2x3x4xi32>
  %0 = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<2x3x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// -----

// CHECK-LABEL: func @iota.const.6() -> tensor<2x3x4xi32> {
func @iota.const.6() -> tensor<2x3x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[\[}}0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3{{\]\]}}, {{\[\[}}0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x3x4xi32>
  %0 = "xla_hlo.iota"() {iota_dimension = 2 : i64} : () -> tensor<2x3x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}
