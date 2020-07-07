// RUN: xla-opt -split-input-file -xla-hlo-to-lhlo-with-xla %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.abs
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %abs = "xla_hlo.abs"(%value) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %abs : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.add
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.add"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lhlo.and
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.and"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.ceil
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.ceil"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<1x2xf32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<1x2xf32>, %value1: tensor<1x2xf32>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lhlo.complex
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.complex"(%value0, %value1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lhlo.cosine
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.cosine"(%value0) : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>>
  return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.divide
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.divide"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.exponential
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.exponential"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.log
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.log"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.maximum
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.maximum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.minimum
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.minimum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.multiply
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.multiply"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.negate
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.negate"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<1x2xf32>
// CHECK: lhlo.real
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.real"(%value0) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  return %res : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<8xi8>
func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<8xi8> to memref<1x2xf32>
// CHECK: lhlo.imag
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.imag"(%value0) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  return %res : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lhlo.remainder
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.remainder"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.rsqrt
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.rsqrt"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi1> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<2x2xf32> {xla_lhlo.params = 2
// CHECK-SAME: %[[ARG3:.*]]: memref<16xi8>
func @main(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.select
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[VIEW]]
// CHECK-NEXT: return
  %0 = "xla_hlo.select"(%pred, %lhs, %rhs) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.sign
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.sign"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.sqrt
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.sqrt"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {xla_lhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lhlo.subtract
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "xla_hlo.subtract"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {xla_lhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lhlo.tanh
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
  %res = "xla_hlo.tanh"(%value0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<5x5xi32>
// CHECK-SAME: %[[ARG1:.*]]: memref<5x5xf32>
// CHECK-SAME: %[[ARG2:.*]]: memref<100xi8> {xla_lhlo.alloc = 0
// CHECK-SAME: %[[ARG3:.*]]: memref<100xi8> {xla_lhlo.alloc = 1
// CHECK: %[[VIEW0:.*]] = std.view %[[ARG2]]{{.*}} : memref<100xi8> to memref<5x5xi32>
// CHECK: %[[VIEW1:.*]] = std.view %[[ARG3]]{{.*}} : memref<100xi8> to memref<5x5xf32>
// CHECK: "xla_lhlo.sort"(%[[ARG0]], %[[ARG1]], %[[VIEW0]], %[[VIEW1]])
func @main(%key: tensor<5x5xi32>, %value: tensor<5x5xf32>) -> tuple<tensor<5x5xi32>, tensor<5x5xf32>> {
  %res = "xla_hlo.sort"(%key, %value) ({
  ^bb0(%a: tensor<i32>, %b: tensor<i32>, %c: tensor<f32>, %d: tensor<f32>):
    %ret = "xla_hlo.compare"(%c, %d) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%ret) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true}: (tensor<5x5xi32>, tensor<5x5xf32>) -> tuple<tensor<5x5xi32>, tensor<5x5xf32>>

  return %res : tuple<tensor<5x5xi32>, tensor<5x5xf32>>
}
