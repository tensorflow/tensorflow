// TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
// statements (perhaps through using lit config substitutions).
//
// RUN: %S/../../mlir-opt %s -o - | FileCheck %s

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2, d3, d4)[s0] -> (d0, d1, d2, d3, d4)
#map0 = (d0, d1, d2, d3, d4)[s0] -> (d0, d1, d2, d3, d4)

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

// CHECK: extfunc @tensors(tensor<??f32>, tensor<??vector<2x4xf32>>, tensor<1x?x4x?x?xi32>, tensor<i8>)
extfunc @tensors(tensor<?? f32>, tensor<?? vector<2x4xf32>>,
                 tensor<1x?x4x?x?xi32>, tensor<i8>)

// CHECK: extfunc @memrefs(memref<1x?x4x?x?xi32, #map{{[0-9]+}}>, memref<i8, #map{{[0-9]+}}>)
extfunc @memrefs(memref<1x?x4x?x?xi32, #map0>, memref<i8, #map1>)

// Test memref affine map compositions.

// CHECK: extfunc @memrefs2(memref<2x4x8xi8, #map{{[0-9]+}}, 1>)
extfunc @memrefs2(memref<2x4x8xi8, #map2, 1>)

// CHECK: extfunc @memrefs23(memref<2x4x8xi8, #map{{[0-9]+}}, #map{{[0-9]+}}>)
extfunc @memrefs23(memref<2x4x8xi8, #map2, #map3, 0>)

// CHECK: extfunc @memrefs234(memref<2x4x8xi8, #map{{[0-9]+}}, #map{{[0-9]+}}, #map{{[0-9]+}}, 3>)
extfunc @memrefs234(memref<2x4x8xi8, #map2, #map3, #map4, 3>)

// Test memref inline affine map compositions.

// CHECK: extfunc @memrefs2(memref<2x4x8xi8, #map{{[0-9]+}}>)
extfunc @memrefs2(memref<2x4x8xi8, (d0, d1, d2) -> (d0, d1, d2)>)

// CHECK: extfunc @memrefs23(memref<2x4x8xi8, #map{{[0-9]+}}, #map{{[0-9]+}}, 1>)
extfunc @memrefs23(memref<2x4x8xi8, (d0, d1, d2) -> (d0, d1, d2), (d0, d1, d2) -> (d1, d0, d2), 1>)

// CHECK: extfunc @functions((memref<1x?x4x?x?xi32, #map0>, memref<i8, #map1>) -> (), () -> ())
extfunc @functions((memref<1x?x4x?x?xi32, #map0, 0>, memref<i8, #map1, 0>) -> (), ()->())

// CHECK-LABEL: cfgfunc @simpleCFG(i32, f32) -> i1 {
cfgfunc @simpleCFG(i32, f32) -> i1 {
// CHECK: bb0(%0: i32, %1: f32):
bb42 (%0: i32, %f: f32):
  // CHECK: %2 = "foo"() : () -> i64
  %1 = "foo"() : ()->i64
  // CHECK: "bar"(%2) : (i64) -> (i1, i1, i1)
  %2 = "bar"(%1) : (i64) -> (i1,i1,i1)
  // CHECK: return %3#1
  return %2#1 : i1
// CHECK: }
}

// CHECK-LABEL: cfgfunc @simpleCFGUsingBBArgs(i32, i64) {
cfgfunc @simpleCFGUsingBBArgs(i32, i64) {
// CHECK: bb0(%0: i32, %1: i64):
bb42 (%0: i32, %f: i64):
  // CHECK: "bar"(%1) : (i64) -> (i1, i1, i1)
  %2 = "bar"(%f) : (i64) -> (i1,i1,i1)
  // CHECK: return
  return
// CHECK: }
}

// CHECK-LABEL: cfgfunc @multiblock() {
cfgfunc @multiblock() {
bb0:         // CHECK: bb0:
  return     // CHECK:   return
bb1:         // CHECK: bb1:   // no predecessors
  br bb4     // CHECK:   br bb3
bb2:         // CHECK: bb2:   // pred: bb2
  br bb2     // CHECK:   br bb2
bb4:         // CHECK: bb3:   // pred: bb1
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: mlfunc @emptyMLF() {
mlfunc @emptyMLF() {
  return     // CHECK:  return
}            // CHECK: }

// CHECK-LABEL: mlfunc @mlfunc_with_args(f16) {
mlfunc @mlfunc_with_args(%a : f16) {
  return  %a  // CHECK: return
}

// CHECK-LABEL: mlfunc @mlfunc_with_ops() {
mlfunc @mlfunc_with_ops() {
  // CHECK: %0 = "foo"() : () -> i64
  %a = "foo"() : ()->i64
  // CHECK: for x = 1 to 10 {
  for %i = 1 to 10 {
    // CHECK: %1 = "doo"() : () -> f32
    %b = "doo"() : ()->f32
    // CHECK: "bar"(%0, %1) : (i64, f32) -> ()
    "bar"(%a, %b) : (i64, f32) -> ()
  // CHECK: }
  }
  // CHECK: return
  return
  // CHECK: }
}


// CHECK-LABEL: mlfunc @loops() {
mlfunc @loops() {
  // CHECK: for x = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: for x = 1 to 200 {
    for %j = 1 to 200 {
    }        // CHECK:     }
  }          // CHECK:   }
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: mlfunc @ifstmt() {
mlfunc @ifstmt() {
  for %i = 1 to 10 {    // CHECK   for x = 1 to 10 {
    if () {             // CHECK     if () {
    } else if () {      // CHECK     } else if () {
    } else {            // CHECK     } else {
    }                   // CHECK     }
  }                     // CHECK   }
  return                // CHECK   return
}                       // CHECK }

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

// CHECK-LABEL: cfgfunc @ssa_values() -> (i16, i8) {
cfgfunc @ssa_values() -> (i16, i8) {
bb0:       // CHECK: bb0:
  // CHECK: %0 = "foo"() : () -> (i1, i17)
  %0 = "foo"() : () -> (i1, i17)
  br bb2

bb1:       // CHECK: bb1: // pred: bb2
  // CHECK: %1 = "baz"(%2#1, %2#0, %0#1) : (f32, i11, i17) -> (i16, i8)
  %1 = "baz"(%2#1, %2#0, %0#1) : (f32, i11, i17) -> (i16, i8)

  // CHECK: return %1#0, %1#1 : i16, i8
  return %1#0, %1#1 : i16, i8

bb2:       // CHECK: bb2:  // pred: bb0
  // CHECK: %2 = "bar"(%0#0, %0#1) : (i1, i17) -> (i11, f32)
  %2 = "bar"(%0#0, %0#1) : (i1, i17) -> (i11, f32)
  br bb1
}

// CHECK-LABEL: cfgfunc @bbargs() -> (i16, i8) {
cfgfunc @bbargs() -> (i16, i8) {
bb0:       // CHECK: bb0:
  // CHECK: %0 = "foo"() : () -> (i1, i17)
  %0 = "foo"() : () -> (i1, i17)
  br bb1(%0#1, %0#0 : i17, i1)

bb1(%x: i17, %y: i1):       // CHECK: bb1(%1: i17, %2: i1):
  // CHECK: %3 = "baz"(%1, %2, %0#1) : (i17, i1, i17) -> (i16, i8)
  %1 = "baz"(%x, %y, %0#1) : (i17, i1, i17) -> (i16, i8)
  return %1#0, %1#1 : i16, i8
}

// CHECK-LABEL: cfgfunc @condbr_simple
cfgfunc @condbr_simple() -> (i32) {
bb0:
  %cond = "foo"() : () -> i1
  %a = "bar"() : () -> i32
  %b = "bar"() : () -> i64
  // CHECK: cond_br %0, bb1(%1 : i32), bb2(%2 : i64)
  cond_br %cond, bb1(%a : i32), bb2(%b : i64)

// CHECK: bb1({{.*}}: i32): // pred: bb0
bb1(%x : i32):
  br bb2(%b: i64)

// CHECK: bb2({{.*}}: i64): // 2 preds: bb0, bb1
bb2(%y : i64):
  %z = "foo"() : () -> i32
  return %z : i32
}

// CHECK-LABEL: cfgfunc @condbr_moarargs
cfgfunc @condbr_moarargs() -> (i32) {
bb0:
  %cond = "foo"() : () -> i1
  %a = "bar"() : () -> i32
  %b = "bar"() : () -> i64
  // CHECK: cond_br %0, bb1(%1, %2 : i32, i64), bb2(%2, %1, %1 : i64, i32, i32)
  cond_br %cond, bb1(%a, %b : i32, i64), bb2(%b, %a, %a : i64, i32, i32)

bb1(%x : i32, %y : i64):
  return %x : i32

bb2(%x2 : i64, %y2 : i32, %z2 : i32):
  %z = "foo"() : () -> i32
  return %z : i32
}
