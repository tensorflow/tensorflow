// RUN: %S/../../mlir-opt %s -o - | FileCheck %s

// CHECK: #map{{[0-9]+}} = (d0, d1) -> ((d0 + 1), (d1 + 2))
#map5 = (d0, d1) -> (d0 + 1, d1 + 2)

// CHECK-LABEL: cfgfunc @cfgfunc_with_ops(f32) {
cfgfunc @cfgfunc_with_ops(f32) {
bb0(%a : f32):
  // CHECK: %1 = "getTensor"() : () -> tensor<4x4x?xf32>
  %t = "getTensor"() : () -> tensor<4x4x?xf32>

  // CHECK: %2 = dim %1, 2 : tensor<4x4x?xf32>
  %t2 = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> affineint

  // CHECK: %3 = addf %0, %0 : f32
  %x = "addf"(%a, %a) : (f32,f32) -> (f32)

  // CHECK:   return
  return
}

// CHECK-LABEL: cfgfunc @standard_instrs() {
cfgfunc @standard_instrs() {
bb42:       // CHECK: bb0:
  // CHECK: %0 = "getTensor"() : () -> tensor<4x4x?xf32>
  %42 = "getTensor"() : () -> tensor<4x4x?xf32>

  // CHECK: dim %0, 2 : tensor<4x4x?xf32>
  %a = "dim"(%42){index: 2} : (tensor<4x4x?xf32>) -> affineint

  // FIXME: Add support for fp attributes so this can use 'constant'.
  %f = "FIXMEConst"(){value: 1} : () -> f32

  // CHECK: %3 = addf %2, %2 : f32
  "addf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %4 = "constant"(){value: 42} : () -> i32
  %x = "constant"(){value: 42} : () -> i32
  return
}

// CHECK-LABEL: cfgfunc @affine_apply() {
cfgfunc @affine_apply() {
bb0:
  %i = "constant"() {value: 0} : () -> affineint
  %j = "constant"() {value: 1} : () -> affineint

  // CHECK: affine_apply map: (d0) -> ((d0 + 1))
  %x = "affine_apply" (%i) { map: (d0) -> (d0 + 1) } :
    (affineint) -> (affineint)

  // CHECK: affine_apply map: (d0, d1) -> ((d0 + 1), (d1 + 2))
  %y = "affine_apply" (%i, %j) { map: #map5 } :
    (affineint, affineint) -> (affineint, affineint)
  return
}