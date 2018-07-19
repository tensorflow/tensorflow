// TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
// statements (perhaps through using lit config substitutions).
//
// RUN: %S/../../mlir-opt %s -o - | FileCheck %s

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2, d3, d4) [s0] -> (d0, d1, d2, d3, d4)
#map0 = (d0, d1, d2, d3, d4) [s0] -> (d0, d1, d2, d3, d4)

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (d0)
#map1 = (d0) -> (d0)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2) -> (d0, d1, d2)
#map2 = (d0, d1, d2) -> (d0, d1, d2)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2) -> (d1, d0, d2)
#map3 = (d0, d1, d2) -> (d1, d0, d2)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2) -> (d2, d1, d0)
#map4 = (d0, d1, d2) -> (d2, d1, d0)

// CHECK: extfunc @foo(i32, i64) -> f32
extfunc @foo(i32, i64) -> f32

// CHECK: extfunc @bar()
extfunc @bar() -> ()

// CHECK: extfunc @baz() -> (i1, affineint, f32)
extfunc @baz() -> (i1, affineint, f32)

// CHECK: extfunc @missingReturn()
extfunc @missingReturn()

// CHECK: extfunc @int_types(i1, i2, i4, i7, i87) -> (i1, affineint, i19)
extfunc @int_types(i1, i2, i4, i7, i87) -> (i1, affineint, i19)


// CHECK: extfunc @vectors(vector<1xf32>, vector<2x4xf32>)
extfunc @vectors(vector<1 x f32>, vector<2x4xf32>)

// CHECK: extfunc @tensors(tensor<??f32>, tensor<??vector<2x4xf32>>, tensor<1x?x4x?x?xaffineint>, tensor<i8>)
extfunc @tensors(tensor<?? f32>, tensor<?? vector<2x4xf32>>,
                 tensor<1x?x4x?x?xaffineint>, tensor<i8>)

// CHECK: extfunc @memrefs(memref<1x?x4x?x?xaffineint, #map{{[0-9]+}}, 0>, memref<i8, #map{{[0-9]+}}, 0>)
extfunc @memrefs(memref<1x?x4x?x?xaffineint, #map0, 0>, memref<i8, #map1, 0>)

// Test memref affine map compositions.

// CHECK: extfunc @memrefs2(memref<2x4x8xi8, #map{{[0-9]+}}, 1>)
extfunc @memrefs2(memref<2x4x8xi8, #map2, 1>)

// CHECK: extfunc @memrefs23(memref<2x4x8xi8, #map{{[0-9]+}}, #map{{[0-9]+}}, 0>)
extfunc @memrefs23(memref<2x4x8xi8, #map2, #map3, 0>)

// CHECK: extfunc @memrefs234(memref<2x4x8xi8, #map{{[0-9]+}}, #map{{[0-9]+}}, #map{{[0-9]+}}, 3>)
extfunc @memrefs234(memref<2x4x8xi8, #map2, #map3, #map4, 3>)

// Test memref inline affine map compositions.

// CHECK: extfunc @memrefs2(memref<2x4x8xi8, #map{{[0-9]+}}, 0>)
extfunc @memrefs2(memref<2x4x8xi8, (d0, d1, d2) -> (d0, d1, d2), 0>)

// CHECK: extfunc @memrefs23(memref<2x4x8xi8, #map{{[0-9]+}}, #map{{[0-9]+}}, 1>)
extfunc @memrefs23(memref<2x4x8xi8, (d0, d1, d2) -> (d0, d1, d2), (d0, d1, d2) -> (d1, d0, d2), 1>)

// CHECK: extfunc @functions((memref<1x?x4x?x?xaffineint, (d0, d1, d2, d3, d4) [s0] -> (d0, d1, d2, d3, d4), 0>, memref<i8, (d0) -> (d0), 0>) -> (), () -> ())
extfunc @functions((memref<1x?x4x?x?xaffineint, #map0, 0>, memref<i8, #map1, 0>) -> (), ()->())

// CHECK-LABEL: cfgfunc @simpleCFG(i32, f32) {
cfgfunc @simpleCFG(i32, f32) {
// CHECK: bb0:
bb42: // (%0: i32, %f: f32):    TODO(clattner): implement bbargs.
  // CHECK: "foo"() : () -> i64
  %1 = "foo"() : ()->i64
  // CHECK: "bar"() : (i64) -> (i1, i1, i1)
  "bar"(%1) : (i64) -> (i1,i1,i1)
  // CHECK: return
  return
// CHECK: }
}

// CHECK-LABEL: cfgfunc @multiblock() -> i32 {
cfgfunc @multiblock() -> i32 {
bb0:         // CHECK: bb0:
  return     // CHECK:   return
bb1:         // CHECK: bb1:
  br bb4     // CHECK:   br bb3
bb2:         // CHECK: bb2:
  br bb2     // CHECK:   br bb2
bb4:         // CHECK: bb3:
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: mlfunc @emptyMLF() {
mlfunc @emptyMLF() {
  return     // CHECK:  return
}            // CHECK: }

// CHECK-LABEL: cfgfunc @cfgfunc_with_ops() {
cfgfunc @cfgfunc_with_ops() {
bb0:
  %t = "getTensor"() : () -> tensor<4x4x?xf32>
  // CHECK: dim xxx, 2 : sometype
  %a = "dim"(%t){index: 2} : (tensor<4x4x?xf32>) -> affineint

  // CHECK: addf xx, yy : sometype
  "addf"() : () -> ()
  return
}

// CHECK-LABEL: mlfunc @loops() {
mlfunc @loops() {
  for {      // CHECK:   for {
    for {    // CHECK:     for {
    }        // CHECK:     }
  }          // CHECK:   }
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: mlfunc @ifstmt() {
mlfunc @ifstmt() {
  for {             // CHECK   for {
    if () {         // CHECK     if () {
    } else if () {  // CHECK     } else if () {
    } else {        // CHECK     } else {
    }               // CHECK     }
  }                 // CHECK   }
  return            // CHECK   return
}                   // CHECK }

// CHECK-LABEL: cfgfunc @attributes() {
cfgfunc @attributes() {
bb42:       // CHECK: bb0:

  // CHECK: "foo"()
  "foo"(){} : ()->()

  // CHECK: "foo"(){a: 1, b: -423, c: [true, false]}  : () -> ()
  "foo"(){a: 1, b: -423, c: [true, false] } : () -> ()

  // CHECK: "foo"(){map1: #map{{[0-9]+}}}
  "foo"(){map1: #map1} : () -> ()

  // CHECK: "foo"(){map2: #map{{[0-9]+}}}
  "foo"(){map2: (d0, d1, d2) -> (d0, d1, d2)} : () -> ()

  // CHECK: "foo"(){map12: [#map{{[0-9]+}}, #map{{[0-9]+}}]}
  "foo"(){map12: [#map1, #map2]} : () -> ()

  // CHECK: "foo"(){cfgfunc: [], i123: 7, if: "foo"} : () -> ()
  "foo"(){if: "foo", cfgfunc: [], i123: 7} : () -> ()

  return
}

// CHECK-LABEL: cfgfunc @standard_instrs() {
cfgfunc @standard_instrs() {
bb42:       // CHECK: bb0:
  %42 = "getTensor"() : () -> tensor<4x4x?xf32>

  // CHECK: dim xxx, 2 : sometype
  %a = "dim"(%42){index: 2} : (tensor<4x4x?xf32>) -> affineint

  %f = "Const"(){value: 1} : () -> f32
  // CHECK: addf xx, yy : sometype
  "addf"(%f, %f) : (f32,f32) -> f32
  return
}
