// RUN: mlir-opt %s | FileCheck %s

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
  %t2 = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> index

  // CHECK: %2 = addf %arg0, %arg0 : f32
  %x = "addf"(%a, %a) : (f32,f32) -> (f32)

  // CHECK:   return
  return
}

// CHECK-LABEL: cfgfunc @standard_instrs(tensor<4x4x?xf32>, f32, i32) {
cfgfunc @standard_instrs(tensor<4x4x?xf32>, f32, i32) {
// CHECK: bb0(%arg0: tensor<4x4x?xf32>, %arg1: f32, %arg2: i32):
bb42(%t: tensor<4x4x?xf32>, %f: f32, %i: i32):
  // CHECK: %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> index

  // CHECK: %1 = dim %arg0, 2 : tensor<4x4x?xf32>
  %a2 = dim %t, 2 : tensor<4x4x?xf32>
  
  // CHECK: %2 = addf %arg1, %arg1 : f32
  %f2 = "addf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %3 = addf %2, %2 : f32
  %f3 = addf %f2, %f2 : f32
   
  // CHECK: %4 = addi %arg2, %arg2 : i32
  %i2 = "addi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %5 = addi %4, %4 : i32
  %i3 = addi %i2, %i2 : i32
  
  // CHECK: %6 = subf %arg1, %arg1 : f32
  %f4 = "subf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %7 = subf %6, %6 : f32
  %f5 = subf %f4, %f4 : f32
 
  // CHECK: %8 = subi %arg2, %arg2 : i32
  %i4 = "subi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %9 = subi %8, %8 : i32
  %i5 = subi %i4, %i4 : i32
 
  // CHECK: %10 = mulf %2, %2 : f32
  %f6 = mulf %f2, %f2 : f32
  
  // CHECK: %11 = muli %4, %4 : i32
  %i6 = muli %i2, %i2 : i32

  // CHECK: %c42_i32 = constant 42 : i32
  %x = "constant"(){value: 42} : () -> i32

  // CHECK: %c42_i32_0 = constant 42 : i32
  %7 = constant 42 : i32

  // CHECK: %c43 = constant 43 {crazy: "foo"} : index
  %8 = constant 43 {crazy: "foo"} : index

  // CHECK: %cst = constant 4.300000e+01 : bf16
  %9 = constant 43.0 : bf16

  // CHECK: %f = constant @cfgfunc_with_ops : (f32) -> ()
  %10 = constant @cfgfunc_with_ops : (f32) -> ()

  // CHECK: %f_1 = constant @affine_apply : () -> ()
  %11 = constant @affine_apply : () -> ()

  // CHECK: %f_2 = constant @affine_apply : () -> ()
  %12 = constant @affine_apply : () -> ()

  return
}

// CHECK-LABEL: cfgfunc @affine_apply() {
cfgfunc @affine_apply() {
bb0:
  %i = "constant"() {value: 0} : () -> index
  %j = "constant"() {value: 1} : () -> index

  // CHECK: affine_apply #map0(%c0)
  %a = "affine_apply" (%i) { map: (d0) -> (d0 + 1) } :
    (index) -> (index)

  // CHECK: affine_apply #map1(%c0, %c1)
  %b = "affine_apply" (%i, %j) { map: #map5 } :
    (index, index) -> (index, index)

  // CHECK: affine_apply #map2(%c0, %c1)[%c1, %c0]
  %c = affine_apply (i,j)[m,n] -> (i+n, j+m)(%i, %j)[%j, %i]

  // CHECK: affine_apply #map3()[%c0]
  %d = affine_apply ()[x] -> (x+1)()[%i]

  return
}

// CHECK-LABEL: cfgfunc @load_store
cfgfunc @load_store(memref<4x4xi32>, index) {
bb0(%0: memref<4x4xi32>, %1: index):
  // CHECK: %0 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %2 = "load"(%0, %1, %1) : (memref<4x4xi32>, index, index)->i32

  // CHECK: %1 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %3 = load %0[%1, %1] : memref<4x4xi32>

  return
}

// CHECK-LABEL: mlfunc @return_op(%arg0 : i32) -> i32 {
mlfunc @return_op(%a : i32) -> i32 {
  // CHECK: return %arg0 : i32
  "return" (%a) : (i32)->()
}

// CHECK-LABEL: mlfunc @calls(%arg0 : i32) {
mlfunc @calls(%arg0 : i32) {
  // CHECK: %0 = call @return_op(%arg0) : (i32) -> i32
  %x = call @return_op(%arg0) : (i32) -> i32
  // CHECK: %1 = call @return_op(%0) : (i32) -> i32
  %y = call @return_op(%x) : (i32) -> i32
  // CHECK: %2 = call @return_op(%0) : (i32) -> i32
  %z = "call"(%x) {callee: @return_op : (i32) -> i32} : (i32) -> i32

  // CHECK: %f = constant @affine_apply : () -> ()
  %f = constant @affine_apply : () -> ()

  // CHECK: call_indirect %f() : () -> ()
  call_indirect %f() : () -> ()

  // CHECK: %f_0 = constant @return_op : (i32) -> i32
  %f_0 = constant @return_op : (i32) -> i32

  // CHECK: %3 = call_indirect %f_0(%arg0) : (i32) -> i32
  %2 = call_indirect %f_0(%arg0) : (i32) -> i32

  // CHECK: %4 = call_indirect %f_0(%arg0) : (i32) -> i32
  %3 = "call_indirect"(%f_0, %arg0) : ((i32) -> i32, i32) -> i32

  return
}

// CHECK-LABEL: mlfunc @extract_element(%arg0 : tensor<*xi32>, %arg1 : tensor<4x4xf32>) -> i32 {
mlfunc @extract_element(%arg0 : tensor<*xi32>, %arg1 : tensor<4x4xf32>) -> i32 {
  %c0 = "constant"() {value: 0} : () -> index

  // CHECK: %0 = extract_element %arg0[%c0, %c0, %c0, %c0] : tensor<*xi32>
  %0 = extract_element %arg0[%c0, %c0, %c0, %c0] : tensor<*xi32>

  // CHECK: %1 = extract_element %arg1[%c0, %c0] : tensor<4x4xf32>
  %1 = extract_element %arg1[%c0, %c0] : tensor<4x4xf32>

  return %0 : i32
}

// CHECK-LABEL: mlfunc @shape_cast(%arg0
mlfunc @shape_cast(%arg0 : tensor<*xf32>, %arg1 : tensor<4x4xf32>, %arg2 : tensor<?x?xf32>) {
  // CHECK: %0 = shape_cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  %0 = shape_cast %arg0 : tensor<*xf32> to tensor<?x?xf32>

  // CHECK: %1 = shape_cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  %1 = shape_cast %arg1 : tensor<4x4xf32> to tensor<*xf32>

  // CHECK: %2 = shape_cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  %2 = shape_cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>

  // CHECK: %3 = shape_cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  %3 = shape_cast %2 : tensor<4x?xf32> to tensor<?x?xf32>

  return
}

// CHECK-LABEL: mlfunc @test_dimop(%arg0
mlfunc @test_dimop(%arg0 : tensor<4x4x?xf32>) {
  // CHECK: %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  %0 = dim %arg0, 2 : tensor<4x4x?xf32>
  // use dim as an affine_int to ensure type correctness
  %1 = affine_apply (d0) -> (d0)(%0)
  return
}

