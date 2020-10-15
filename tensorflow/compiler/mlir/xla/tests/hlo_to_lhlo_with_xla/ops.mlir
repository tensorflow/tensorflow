// RUN: xla-opt -split-input-file -xla-hlo-to-lhlo-with-xla %s | FILECHECK_OPTS="" FileCheck --enable-var-scope %s

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.add
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.add"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.and
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.and"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<1x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<1x2xf32>, %value1: tensor<1x2xf32>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lmhlo.complex
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.complex"(%value0, %value1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<16xi8>
func @main(%value0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<1x2xcomplex<f32>>
// CHECK: lmhlo.cosine
// CHECK-SAME: %[[ARG0]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.cosine"(%value0) : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>>
  return %res : tensor<1x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.divide
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.divide"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.maximum
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.maximum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.minimum
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.minimum"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xf32>, %value1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.multiply
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.multiply"(%value0, %value1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %res : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<1x2xcomplex<f32>> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.remainder
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.remainder"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi1> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 2
// CHECK-SAME: %[[ARG3:.*]]: memref<16xi8>
func @main(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xf32>
// CHECK: lmhlo.select
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[VIEW]]
// CHECK-NEXT: return
  %0 = "mhlo.select"(%pred, %lhs, %rhs) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xi32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
// CHECK-SAME: %[[ARG1:.*]]: memref<2x2xi32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 1
// CHECK-SAME: %[[ARG2:.*]]: memref<16xi8>
func @main(%value0: tensor<2x2xi32>, %value1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[VIEW:.*]] = {{.*}} memref<16xi8> to memref<2x2xi32>
// CHECK: lmhlo.subtract
// CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[VIEW]]
// CHECK-NEXT: return
  %res = "mhlo.subtract"(%value0, %value1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %res : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: memref<2x2xf32> {lmhlo.alloc = {{[0-9]+}} : index, lmhlo.params = 0
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
// CHECK-SAME: %[[ARG0:.*]]: memref<5x5xi32>
// CHECK-SAME: %[[ARG1:.*]]: memref<5x5xf32>
// CHECK-SAME: %[[ARG2:.*]]: memref<100xi8> {lmhlo.alloc = 0
// CHECK-SAME: %[[ARG3:.*]]: memref<100xi8> {lmhlo.alloc = 1
// CHECK: %[[VIEW0:.*]] = std.view %[[ARG2]]{{.*}} : memref<100xi8> to memref<5x5xi32>
// CHECK: %[[VIEW1:.*]] = std.view %[[ARG3]]{{.*}} : memref<100xi8> to memref<5x5xf32>
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
// CHECK:   %[[VAR0:.*]] = tensor_load %[[ARG0]] : memref<f32>
// CHECK:   %[[VAR1:.*]] = tensor_load %[[ARG1]] : memref<f32>
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
// CHECK:   %[[VAL0:.*]] = tensor_load %{{.*}} : memref<f32>
// CHECK:   %[[VAL1:.*]] = tensor_load %{{.*}} : memref<f32>
// CHECK:   %[[VAL2:.*]] = tensor_load %{{.*}} : memref<f32>
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
