// RUN: xla-translate-opt -split-input-file -xla-hlo-to-lhlo-with-xla %s | FILECHECK_OPTS="" FileCheck --enable-var-scope %s

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.abs
  %abs = "mhlo.abs"(%value) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %abs : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.add
// CHECK: lmhlo.terminator
  %res = "mhlo.add"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.and
// CHECK: lmhlo.terminator
  %res = "mhlo.and"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.atan2
// CHECK: lmhlo.terminator
  %res = "mhlo.atan2"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value: tensor<2x2xf32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.bitcast_convert
  %res = "mhlo.bitcast_convert"(%value) : (tensor<2x2xf32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.ceil
  %res = "mhlo.ceil"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.cbrt
  %res = "mhlo.cbrt"(%value) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8> {lmhlo.params = 2
// CHECK-SAME: %[[ARG3:.*]]: memref<16xi8>
func.func @main(%pred: tensor<2x2xf32>, %lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.clamp
// CHECK: lmhlo.terminator
  %0 = "mhlo.clamp"(%pred, %lhs, %rhs) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.count_leading_zeros
  %res = "mhlo.count_leading_zeros"(%value) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.fusion
// CHECK: mhlo.compare
// CHECK: lmhlo.terminator
  %res = "mhlo.compare"(%value0, %value1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<8xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<1x2xf32>, %value1: tensor<1x2xf32>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lmhlo.fusion
// CHECK: mhlo.complex
// CHECK: lmhlo.terminator
  %res = "mhlo.complex"(%value0, %value1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  func.return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func.func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf16> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<2x2xf16>
// CHECK: lmhlo.fusion
// CHECK: mhlo.convert
  %res = "mhlo.convert"(%value) : (tensor<2x2xf32>) -> tensor<2x2xf16>
  func.return %res : tensor<2x2xf16>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lmhlo.fusion
// CHECK: mhlo.cosine
// CHECK: lmhlo.terminator
  %res = "mhlo.cosine"(%value0) : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>>
  func.return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.divide
// CHECK: lmhlo.terminator
  %res = "mhlo.divide"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.exponential
  %res = "mhlo.exponential"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.exponential_minus_one
  %res = "mhlo.exponential_minus_one"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.floor
  %res = "mhlo.floor"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<4xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.fusion
// CHECK: mhlo.is_finite
  %res = "mhlo.is_finite"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.log
  %res = "mhlo.log"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.log_plus_one
  %res = "mhlo.log_plus_one"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.map
// CHECK: return
  %res = "mhlo.map"(%value0, %value1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %c = "mhlo.add"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %ret = "mhlo.add"(%a, %c) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%ret) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.maximum
// CHECK: lmhlo.terminator
  %res = "mhlo.maximum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.minimum
// CHECK: lmhlo.terminator
  %res = "mhlo.minimum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.multiply
// CHECK: lmhlo.terminator
  %res = "mhlo.multiply"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.negate
  %res = "mhlo.negate"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<4xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<4xi8>
func.func @main(%value0: tensor<2x2xi1>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.fusion
// CHECK: mhlo.not
  %res = "mhlo.not"(%value0) : (tensor<2x2xi1>) -> tensor<2x2xi1>
  func.return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.not
  %res = "mhlo.not"(%value0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<4xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<4xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
func.func @main(%value0: tensor<2x2xi1>, %value1: tensor<2x2xi1>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.fusion
// CHECK: mhlo.or
// CHECK: lmhlo.terminator
  %res = "mhlo.or"(%value0, %value1) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  func.return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.or
// CHECK: lmhlo.terminator
  %res = "mhlo.or"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.popcnt
  %res = "mhlo.popcnt"(%value0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.power
// CHECK: lmhlo.terminator
  %res = "mhlo.power"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func.func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<1x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.real
  %res = "mhlo.real"(%value0) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  func.return %res : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func.func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<1x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.imag
  %res = "mhlo.imag"(%value0) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  func.return %res : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.reduce_precision
  %res = "mhlo.reduce_precision"(%value0) {exponent_bits=5 : i32, mantissa_bits=12 : i32}: (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.remainder
// CHECK: lmhlo.terminator
  %res = "mhlo.remainder"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.round_nearest_afz
  %res = "mhlo.round_nearest_afz"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.rsqrt
  %res = "mhlo.rsqrt"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<4xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8> {lmhlo.params = 2
// CHECK-SAME: %[[ARG3:.*]]: memref<16xi8>
func.func @main(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.select
// CHECK: lmhlo.terminator
  %0 = "mhlo.select"(%pred, %lhs, %rhs) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.shift_left
// CHECK: lmhlo.terminator
  %res = "mhlo.shift_left"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.shift_right_arithmetic
// CHECK: lmhlo.terminator
  %res = "mhlo.shift_right_arithmetic"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.shift_right_logical
// CHECK: lmhlo.terminator
  %res = "mhlo.shift_right_logical"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.sign
  %res = "mhlo.sign"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.sine
  %res = "mhlo.sine"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.sqrt
  %res = "mhlo.sqrt"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.subtract
// CHECK: lmhlo.terminator
  %res = "mhlo.subtract"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.tanh
  %res = "mhlo.tanh"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<4xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<4xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
func.func @main(%value0: tensor<2x2xi1>, %value1: tensor<2x2xi1>) -> tensor<2x2xi1> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<4xi8> to memref<2x2xi1>
// CHECK: lmhlo.fusion
// CHECK: mhlo.xor
// CHECK: lmhlo.terminator
  %res = "mhlo.xor"(%value0, %value1) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  func.return %res : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<16xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func.func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.fusion
// CHECK: mhlo.xor
// CHECK: lmhlo.terminator
  %res = "mhlo.xor"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<100xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<100xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<100xi8> {lmhlo.output_index = dense<0>
// CHECK-SAME: %[[ARG3:.*]]: memref<100xi8> {lmhlo.output_index = dense<1>
// CHECK: %[[VIEW0:.*]] = memref.view %[[ARG0]]{{.*}} : memref<100xi8> to memref<5x5xi32>
// CHECK: %[[VIEW1:.*]] = memref.view %[[ARG1]]{{.*}} : memref<100xi8> to memref<5x5xf32>
// CHECK: %[[VIEW2:.*]] = memref.view %[[ARG2]]{{.*}} : memref<100xi8> to memref<5x5xi32>
// CHECK: %[[VIEW3:.*]] = memref.view %[[ARG3]]{{.*}} : memref<100xi8> to memref<5x5xf32>
// CHECK: "lmhlo.sort"(%[[VIEW0]], %[[VIEW1]], %[[VIEW2]], %[[VIEW3]])
func.func @main(%key: tensor<5x5xi32>, %value: tensor<5x5xf32>) -> (tensor<5x5xi32>, tensor<5x5xf32>) {
  %res:2 = "mhlo.sort"(%key, %value) ({
  ^bb0(%a: tensor<i32>, %b: tensor<i32>, %c: tensor<f32>, %d: tensor<f32>):
    %ret = "mhlo.compare"(%c, %d) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%ret) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true}: (tensor<5x5xi32>, tensor<5x5xf32>) -> (tensor<5x5xi32>, tensor<5x5xf32>)

  func.return %res#0, %res#1 : tensor<5x5xi32>, tensor<5x5xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<4xi8> {lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<4xi8> {lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<4xi8>
// CHECK: %[[VIEW0:.*]] = memref.view %[[ARG0]]{{.*}} : memref<4xi8> to memref<f32>
// CHECK: %[[VIEW1:.*]] = memref.view %[[ARG1]]{{.*}} : memref<4xi8> to memref<f32>
// CHECK: "lmhlo.fusion"() ({
// CHECK:   %[[VAR0:.*]] = bufferization.to_tensor %[[VIEW0]] : memref<f32>
// CHECK:   %[[VAR1:.*]] = bufferization.to_tensor %[[VIEW1]] : memref<f32>
// CHECK:   %[[VAR2:.*]] = mhlo.add %[[VAR0]], %[[VAR1]] : tensor<f32>
// CHECK:   tensor_store %[[VAR2]], %[[MEMREF:.*]] : memref<f32>
// CHECK:   "lmhlo.terminator"() : () -> ()
// CHECK: }) {backend_config = ""} : () -> ()
func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %result = "mhlo.fusion"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %result = "mhlo.add"(%arg2, %arg3): (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%result) : (tensor<f32>) -> ()
    }) { fusion_kind = #mhlo<fusion_kind kLoop> } : (tensor<f32>, tensor<f32>) -> tensor<f32>

  func.return %result : tensor<f32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "lmhlo.fusion"() ({
// CHECK:   %[[VAL0:.*]] = bufferization.to_tensor %{{.*}} : memref<f32>
// CHECK:   %[[VAL1:.*]] = bufferization.to_tensor %{{.*}} : memref<f32>
// CHECK:   %[[VAL2:.*]] = bufferization.to_tensor %{{.*}} : memref<f32>
// CHECK:   tensor_store %[[VAL0]], %{{.*}} : memref<f32>
// CHECK:   tensor_store %[[VAL1]], %{{.*}} : memref<f32>
// CHECK:   tensor_store %[[VAL2]], %{{.*}} : memref<f32>
// CHECK:   "lmhlo.terminator"() : () -> ()
// CHECK: }) {backend_config = ""} : () -> ()
func.func @main(%arg0: tuple<tuple<tensor<f32>>, tensor<f32>>, %arg1: tuple<tensor<f32>>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tuple<tensor<f32>>, tensor<f32>>) -> tuple<tensor<f32>>
  %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>
  %2 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tuple<tensor<f32>>, tensor<f32>>) -> tensor<f32>
  %3 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<f32>>) -> tensor<f32>
  %result:3 = "mhlo.fusion"(%1, %2, %3) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>):
      "mhlo.return"(%arg2, %arg3, %arg4) : (tensor<f32>, tensor<f32>, tensor<f32>) -> ()
    }) { fusion_kind = #mhlo<fusion_kind kLoop> } : (tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>)

  %4 = "mhlo.tuple"(%result#0, %result#1, %result#2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>>
  func.return %4 : tuple<tensor<f32>, tensor<f32>, tensor<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK:   mhlo.reduce
// CHECK:   (%[[VAL1:.*]]: tensor<f32>, %[[VAL3:.*]]: tensor<f32>)
// CHECK-SAME: (%[[VAL2:.*]]: tensor<i32>, %[[VAL4:.*]]: tensor<i32>)
// CHECK:     %[[VAL5:.*]] = mhlo.maximum %[[VAL1]], %[[VAL3]] : tensor<f32>
// CHECK:     %[[VAL6:.*]] = mhlo.maximum %[[VAL2]], %[[VAL4:.*]] : tensor<i32>
// CHECK:     mhlo.return %[[VAL5]], %[[VAL6:.*]] : tensor<f32>, tensor<i32>
// CHECK:   })
func.func @main(%arg0 : tensor<1x10xf32>, %arg1 : tensor<1x10xi32>, %arg2 : tensor<f32>, %arg3 : tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %result0, %result1 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%fa: tensor<f32>, %ia : tensor<i32>, %fb: tensor<f32>, %ib: tensor<i32>):
      %fmax = "mhlo.maximum"(%fa, %fb) {} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %imax = "mhlo.maximum"(%ia, %ib) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "mhlo.return"(%fmax, %imax) : (tensor<f32>, tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
  func.return %result0, %result1 : tensor<1xf32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "mhlo.concatenate"(%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]]) {dimension = 1 : i64} : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
func.func @main(%arg0 : tensor<5x2xf32>,
           %arg1 : tensor<5x5xf32>,
           %arg2 : tensor<5x7xf32>) -> tensor<5x14xf32> {
  %result = "mhlo.concatenate"(%arg0, %arg1, %arg2) {
    dimension = 1 : i64
  } : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  func.return %result : tensor<5x14xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<1x10xf32>
func.func @main() -> tensor<1x10xf32> {
  %result = "mhlo.iota"() {
    iota_dimension = 1 : i64
  } : () -> tensor<1x10xf32>
  func.return %result : tensor<1x10xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "mhlo.reverse"(%{{.*}}) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
func.func @main(%arg0 : tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %result = "mhlo.reverse"(%arg0) {
    dimensions = dense<[1,2]> : tensor<2xi64>
  } : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  func.return %result : tensor<10x11x12x13xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "mhlo.slice"(%{{.*}}) {
// CHECK-SAME: limit_indices = dense<[2, 4]> : tensor<2xi64>,
// CHECK-SAME: start_indices = dense<[1, 0]> : tensor<2xi64>,
// CHECK-SAME: strides = dense<[1, 2]> : tensor<2xi64>}
// CHECK-SAME: : (tensor<3x4xf32>) -> tensor<1x2xf32>
func.func @main(%arg: tensor<3x4xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xf32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "mhlo.gather"(%{{.*}}, %{{.*}}) {
// CHECK-SAME: dimension_numbers =
// CHECK-SAME:   offset_dims = [1]
// CHECK-SAME:   collapsed_slice_dims = [0, 1]
// CHECK-SAME:   start_index_map = [0, 1]
// CHECK-SAME:   index_vector_dim = 1
// CHECK-SAME: slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>}
// CHECK-SAME: : (tensor<200x100x300xf32>, tensor<10x2xi32>) -> tensor<10x300xf32>
func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<10x300xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 1,
      offset_dims = [1],
      start_index_map = [0, 1],
    >,
    indices_are_sorted = true,
    slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>) -> tensor<10x300xf32>
  func.return %0 : tensor<10x300xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: "mhlo.dynamic_slice"(%{{.*}}, %{{.*}}, %{{.*}}) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
func.func @main(%arg: tensor<3x4xf32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xf32> {
  %0 = "mhlo.dynamic_slice"(%arg, %start1, %start2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK: mhlo.dynamic_update_slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x4xf32>, tensor<1x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<1x4xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<4x4xf32> {
  %0 = "mhlo.dynamic_update_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<4x4xf32>, tensor<1x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
