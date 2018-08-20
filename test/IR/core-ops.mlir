// RUN: %S/../../mlir-opt %s -o - | FileCheck %s

// CHECK: #map0 = (d0) -> (d0 + 1)

// CHECK: #map1 = (d0, d1) -> (d0 + 1, d1 + 2)
#map5 = (d0, d1) -> (d0 + 1, d1 + 2)

// CHECK: #map2 = (d0, d1)[s0, s1] -> (d0 + s1, d1 + s0)
// CHECK: #map3 = ()[s0] -> (s0 + 1)

// CHECK-LABEL: cfgfunc @cfgfunc_with_ops(f32) {
cfgfunc @cfgfunc_with_ops(f32) {
bb0(%a : f32):
  // CHECK: %0 = "getTensor"() : () -> tensor<4x4x?xf32>
  %t = "getTensor"() : () -> tensor<4x4x?xf32>

  // CHECK: %1 = dim %0, 2 : tensor<4x4x?xf32>
  %t2 = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> affineint

  // CHECK: %2 = addf %arg0, %arg0 : f32
  %x = "addf"(%a, %a) : (f32,f32) -> (f32)

  // CHECK:   return
  return
}

// CHECK-LABEL: cfgfunc @standard_instrs(tensor<4x4x?xf32>, f32) {
cfgfunc @standard_instrs(tensor<4x4x?xf32>, f32) {
// CHECK: bb0(%arg0: tensor<4x4x?xf32>, %arg1: f32):
bb42(%t: tensor<4x4x?xf32>, %f: f32):
  // CHECK: %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> affineint

  // CHECK: %1 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a2 = dim %t, 2 : tensor<4x4x?xf32>

  // CHECK: %2 = addf %arg1, %arg1 : f32
  %f2 = "addf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %3 = addf %2, %2 : f32
  %f3 = addf %f2, %f2 : f32

  // CHECK: %c42_i32 = constant 42 : i32
  %x = "constant"(){value: 42} : () -> i32

  // CHECK: %c42_i32_0 = constant 42 : i32
  %7 = constant 42 : i32

  // CHECK: %c43 = constant 43 {crazy: "foo"} : affineint
  %8 = constant 43 {crazy: "foo"} : affineint

  // CHECK: %cst = constant 4.300000e+01 : bf16
  %9 = constant 43.0 : bf16

  // CHECK: %f = constant @cfgfunc_with_ops : (f32) -> ()
  %10 = constant @cfgfunc_with_ops : (f32) -> ()

  // CHECK: %f_1 = constant @affine_apply : () -> ()
  %11 = constant @affine_apply : () -> ()

  return
}

// CHECK-LABEL: cfgfunc @affine_apply() {
cfgfunc @affine_apply() {
bb0:
  %i = "constant"() {value: 0} : () -> affineint
  %j = "constant"() {value: 1} : () -> affineint

  // CHECK: affine_apply #map0(%c0)
  %a = "affine_apply" (%i) { map: (d0) -> (d0 + 1) } :
    (affineint) -> (affineint)

  // CHECK: affine_apply #map1(%c0, %c1)
  %b = "affine_apply" (%i, %j) { map: #map5 } :
    (affineint, affineint) -> (affineint, affineint)

  // CHECK: affine_apply #map2(%c0, %c1)[%c1, %c0]
  %c = affine_apply (i,j)[m,n] -> (i+n, j+m)(%i, %j)[%j, %i]

  // CHECK: affine_apply #map3()[%c0]
  %d = affine_apply ()[x] -> (x+1)()[%i]

  return
}

// CHECK-LABEL: cfgfunc @load_store
cfgfunc @load_store(memref<4x4xi32>, affineint) {
bb0(%0: memref<4x4xi32>, %1: affineint):
  // CHECK: %0 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %2 = "load"(%0, %1, %1) : (memref<4x4xi32>, affineint, affineint)->i32

  // CHECK: %1 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %3 = load %0[%1, %1] : memref<4x4xi32>

  return
}

// CHECK-LABEL: mlfunc @return_op(%arg0 : i32) -> i32 {
mlfunc @return_op(%a : i32) -> i32 {
  // CHECK: return %arg0 : i32
  "return" (%a) : (i32)->()
}
