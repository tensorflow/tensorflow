// RUN: xla-opt -split-input-file -xla-hlo-to-lhlo-with-xla %s | FILECHECK_OPTS="" FileCheck --enable-var-scope %s

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.abs
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %abs = "mhlo.abs"(%value) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %abs : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.add
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.add"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.and
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.and"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.atan2
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.atan2"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value: tensor<2x2xf32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.bitcast_convert
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.bitcast_convert"(%value) : (tensor<2x2xf32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.ceil
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.ceil"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.cbrt
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.cbrt"(%value) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<2x2xf32> {lmhlo.params = 2
// CHECK-SAME: %[[ARG3:.*]]: memref<16xi8>
func @main(%pred: tensor<2x2xf32>, %lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.clamp
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %0 = "mhlo.clamp"(%pred, %lhs, %rhs) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.count_leading_zeros
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.count_leading_zeros"(%value) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.compare
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.compare"(%value0, %value1) {comparison_direction="GT"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<1x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<1x2xf32>, %value1: tensor<1x2xf32>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lmhlo.complex
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.complex"(%value0, %value1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf16> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<2x2xf16>
// CHECK: lmhlo.convert
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.convert"(%value) : (tensor<2x2xf32>) -> tensor<2x2xf16>
  return %res : tensor<2x2xf16>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lmhlo.cosine
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.cosine"(%value0) : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>>
  return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.divide
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.divide"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.exponential
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.exponential"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.exponential_minus_one
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.exponential_minus_one"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.floor
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.floor"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<4xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.is_finite
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.is_finite"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.log
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.log"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.log_plus_one
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.log_plus_one"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.map
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK: return
  %res = "mhlo.map"(%value0, %value1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %c = "mhlo.add"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %ret = "mhlo.add"(%a, %c) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%ret) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.maximum
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.maximum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.minimum
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.minimum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.multiply
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.multiply"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.negate
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.negate"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi1> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<4xi8>
func @main(%value0: tensor<2x2xi1>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.not
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.not"(%value0) : (tensor<2x2xi1>) -> tensor<2x2xi1>
  return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.not
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.not"(%value0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi1> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi1> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
func @main(%value0: tensor<2x2xi1>, %value1: tensor<2x2xi1>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.or
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.or"(%value0, %value1) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.or
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.or"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.popcnt
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.popcnt"(%value0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.power
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.power"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<1x2xf32>
// CHECK: lmhlo.real
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.real"(%value0) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  return %res : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<1x2xf32>
// CHECK: lmhlo.imag
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.imag"(%value0) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  return %res : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.reduce_precision
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.reduce_precision"(%value0) {exponent_bits=5 : i32, mantissa_bits=12 : i32}: (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.remainder
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.remainder"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.round_nearest_afz
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.round_nearest_afz"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.rsqrt
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.rsqrt"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi1> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<2x2xf32> {lmhlo.params = 2
// CHECK-SAME: %[[ARG3:.*]]: memref<16xi8>
func @main(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.select
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %0 = "mhlo.select"(%pred, %lhs, %rhs) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.shift_left
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.shift_left"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.shift_right_arithmetic
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.shift_right_arithmetic"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.shift_right_logical
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.shift_right_logical"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.sign
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.sign"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.sine
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.sine"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.sqrt
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.sqrt"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.subtract
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.subtract"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.tanh
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "mhlo.tanh"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi1> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi1> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
func @main(%value0: tensor<2x2xi1>, %value1: tensor<2x2xi1>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.xor
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.xor"(%value0, %value1) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.xor
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: lmhlo.terminator
  %res = "mhlo.xor"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<5x5xi32>
// CHECK-SAME: %[[ARG1:.*]]: memref<5x5xf32>
// CHECK-SAME: %[[ARG2:.*]]: memref<100xi8> {lmhlo.output_index = dense<0>
// CHECK-SAME: %[[ARG3:.*]]: memref<100xi8> {lmhlo.output_index = dense<1>
// CHECK: %[[VIEW0:.*]] = memref.view %[[ARG2]]{{.*}} : memref<100xi8> to memref<5x5xi32>
// CHECK: %[[VIEW1:.*]] = memref.view %[[ARG3]]{{.*}} : memref<100xi8> to memref<5x5xf32>
// CHECK: "lmhlo.sort"(%[[ARG0]], %[[ARG1]], %[[VIEW0]], %[[VIEW1]])
func @main(%key: tensor<5x5xi32>, %value: tensor<5x5xf32>) -> (tensor<5x5xi32>, tensor<5x5xf32>) {
  %res:2 = "mhlo.sort"(%key, %value) ({
  ^bb0(%a: tensor<i32>, %b: tensor<i32>, %c: tensor<f32>, %d: tensor<f32>):
    %ret = "mhlo.compare"(%c, %d) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%ret) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true}: (tensor<5x5xi32>, tensor<5x5xf32>) -> (tensor<5x5xi32>, tensor<5x5xf32>)

  return %res#0, %res#1 : tensor<5x5xi32>, tensor<5x5xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<f32> {{.*}}lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<f32> {{.*}}lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
// CHECK: "lmhlo.fusion"() ( {
// CHECK:   %[[VAR0:.*]] = memref.tensor_load %[[ARG0]] : memref<f32>
// CHECK:   %[[VAR1:.*]] = memref.tensor_load %[[ARG1]] : memref<f32>
// CHECK:   %[[VAR2:.*]] = mhlo.add %[[VAR0]], %[[VAR1]] : tensor<f32>
// CHECK:   tensor_store %[[VAR2]], %[[MEMREF:.*]] : memref<f32>
// CHECK:   "lmhlo.terminator"() : () -> ()
// CHECK: }) : () -> ()
func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %result = "mhlo.fusion"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %result = "mhlo.add"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%result) : (tensor<f32>) -> ()
    }) { fusion_kind = "kLoop" } : (tensor<f32>, tensor<f32>) -> tensor<f32>

  return %result : tensor<f32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.fusion"() ( {
// CHECK:   %[[VAL0:.*]] = memref.tensor_load %{{.*}} : memref<f32>
// CHECK:   %[[VAL1:.*]] = memref.tensor_load %{{.*}} : memref<f32>
// CHECK:   %[[VAL2:.*]] = memref.tensor_load %{{.*}} : memref<f32>
// CHECK:   tensor_store %[[VAL0]], %{{.*}} : memref<f32>
// CHECK:   tensor_store %[[VAL1]], %{{.*}} : memref<f32>
// CHECK:   tensor_store %[[VAL2]], %{{.*}} : memref<f32>
// CHECK:   "lmhlo.terminator"() : () -> ()
// CHECK: }) : () -> ()
func @main(%arg0: tuple<tuple<tensor<f32>>, tensor<f32>>, %arg1: tuple<tensor<f32>>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>> {
  %result = "mhlo.fusion"(%arg0, %arg1) ( {
    ^bb0(%arg2: tuple<tuple<tensor<f32>>, tensor<f32>>, %arg3: tuple<tensor<f32>>):
      %0 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tuple<tensor<f32>>, tensor<f32>>) -> tuple<tensor<f32>>
      %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>
      %2 = "mhlo.get_tuple_element"(%arg2) {index = 1 : i32} : (tuple<tuple<tensor<f32>>, tensor<f32>>) -> tensor<f32>
      %3 = "mhlo.get_tuple_element"(%arg3) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>
      %4 = "mhlo.tuple"(%1, %2, %3) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>>
      "mhlo.return"(%4) : (tuple<tensor<f32>, tensor<f32>, tensor<f32>>) -> ()
    }) { fusion_kind = "kLoop" } : (tuple<tuple<tensor<f32>>, tensor<f32>>, tuple<tensor<f32>>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>>

  return %result : tuple<tensor<f32>, tensor<f32>, tensor<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK:   "lmhlo.reduce"({{.*}}) ( {
// CHECK:   ^bb0(%[[VAL1:.*]]: tensor<f32>, %[[VAL2:.*]]: tensor<i32>, %[[VAL3:.*]]: tensor<f32>, %[[VAL4:.*]]: tensor<i32>):  // no predecessors
// CHECK:     %[[VAL5:.*]] = mhlo.maximum %[[VAL1]], %[[VAL3]] : tensor<f32>
// CHECK:     %[[VAL6:.*]] = mhlo.maximum %[[VAL2]], %[[VAL4:.*]] : tensor<i32>
// CHECK:     %[[VAL7:.*]] = "mhlo.tuple"(%[[VAL5]], %[[VAL6:.*]]) : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
// CHECK:     "mhlo.return"(%[[VAL7:.*]]) : (tuple<tensor<f32>, tensor<i32>>) -> ()
// CHECK:   })
func @main(%arg0 : tensor<1x10xf32>, %arg1 : tensor<1x10xi32>, %arg2 : tensor<f32>, %arg3 : tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %result0, %result1 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ( {
    ^bb0(%fa: tensor<f32>, %ia : tensor<i32>, %fb: tensor<f32>, %ib: tensor<i32>):   // no predecessors
      %fmax = "mhlo.maximum"(%fa, %fb) {} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %imax = "mhlo.maximum"(%ia, %ib) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "mhlo.return"(%fmax, %imax) : (tensor<f32>, tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
  return %result0, %result1 : tensor<1xf32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.concatenate"(%arg0, %arg1, %arg2, %{{.*}}) {dimension = 1 : i64} : (memref<5x2xf32>, memref<5x5xf32>, memref<5x7xf32>, memref<5x14xf32>) -> ()
func @main(%arg0 : tensor<5x2xf32>,
           %arg1 : tensor<5x5xf32>,
           %arg2 : tensor<5x7xf32>) -> tensor<5x14xf32> {
  %result = "mhlo.concatenate"(%arg0, %arg1, %arg2) {
    dimension = 1 : i64
  } : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  return %result : tensor<5x14xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.iota"(%{{.*}}) {iota_dimension = 1 : i64} : (memref<1x10xf32>) -> ()
func @main() -> tensor<1x10xf32> {
  %result = "mhlo.iota"() {
    iota_dimension = 1 : i64
  } : () -> tensor<1x10xf32>
  return %result : tensor<1x10xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.reverse"(%arg0, %{{.*}}) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (memref<10x11x12x13xf32>, memref<10x11x12x13xf32>) -> ()
func @main(%arg0 : tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %result = "mhlo.reverse"(%arg0) {
    dimensions = dense<[1,2]> : tensor<2xi64>
  } : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  return %result : tensor<10x11x12x13xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.slice"(%arg0, %{{.*}}) {
// CHECK-SAME: limit_indices = dense<[2, 4]> : tensor<2xi64>,
// CHECK-SAME: start_indices = dense<[1, 0]> : tensor<2xi64>,
// CHECK-SAME: strides = dense<[1, 2]> : tensor<2xi64>}
// CHECK-SAME: : (memref<3x4xf32>, memref<1x2xf32>) -> ()
func @main(%arg: tensor<3x4xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.gather"(%arg0, %arg1, %{{.*}}) {
// CHECK-SAME: dimension_numbers = {collapsed_slice_dims = dense<[0, 1]> : tensor<2xi64>,
// CHECK-SAME: index_vector_dim = 1 : i64, offset_dims = dense<1> : tensor<1xi64>,
// CHECK-SAME: start_index_map = dense<[0, 1]> : tensor<2xi64>},
// CHECK-SAME: slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>}
// CHECK-SAME: : (memref<200x100x300xf32>, memref<10x2xi32>, memref<10x300xf32>) -> ()
func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<10x300xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = {collapsed_slice_dims = dense<[0, 1]> : tensor<2xi64>, index_vector_dim = 1 : i64, offset_dims = dense<1> : tensor<1xi64>, start_index_map = dense<[0, 1]> : tensor<2xi64>}, indices_are_sorted = true, slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>} : (tensor<200x100x300xf32>, tensor<10x2xi32>) -> tensor<10x300xf32>
  return %0 : tensor<10x300xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.dynamic_slice"(%arg0, %arg1, %arg2, %{{.*}}) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (memref<3x4xf32>, memref<i64>, memref<i64>, memref<1x4xf32>) -> ()
func @main(%arg: tensor<3x4xf32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xf32> {
  %0 = "mhlo.dynamic-slice"(%arg, %start1, %start2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.dynamic-update-slice"(%{{.*}}, %arg1, %arg2, %arg3, %{{.*}}) : (memref<4x4xf32>, memref<1x4xf32>, memref<i32>, memref<i32>, memref<4x4xf32>) -> ()
func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<1x4xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<4x4xf32> {
  %0 = "mhlo.dynamic-update-slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<4x4xf32>, tensor<1x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
