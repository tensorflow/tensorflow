// RUN: mlir-opt %s | FileCheck %s

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2, d3, d4)[s0] -> (d0, d1, d2, d4, d3)
#map0 = (d0, d1, d2, d3, d4)[s0] -> (d0, d1, d2, d4, d3)

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (d0)
#map1 = (d0) -> (d0)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2) -> (d0, d1, d2)
#map2 = (d0, d1, d2) -> (d0, d1, d2)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2) -> (d1, d0, d2)
#map3 = (d0, d1, d2) -> (d1, d0, d2)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1, d2) -> (d2, d1, d0)
#map4 = (d0, d1, d2) -> (d2, d1, d0)

// CHECK-DAG: #map{{[0-9]+}} = ()[s0] -> (0, s0 - 1)
#inline_map_minmax_loop1 = ()[s0] -> (0, s0 - 1)

// CHECK-DAG: #map{{[0-9]+}} = ()[s0] -> (100, s0 + 1)
#inline_map_minmax_loop2 = ()[s0] -> (100, s0 + 1)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1)[s0] -> (d0 + d1 + s0)
#bound_map1 = (i, j)[s] -> (i + j + s)

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1) -> (d0 + d1)
#inline_map_loop_bounds2 = (d0, d1) -> (d0 + d1)

// CHECK-DAG: #map{{[0-9]+}} = (d0)[s0] -> (d0 + s0, d0 - s0)
#bound_map2 = (i)[s] -> (i + s, i - s)

// All maps appear in arbitrary order before all sets, in arbitrary order.
// CHECK-EMPTY

// CHECK-DAG: #set{{[0-9]+}} = (d0)[s0, s1] : (d0 >= 0, -d0 + s0 >= 0, s0 - 5 == 0, -d0 + s1 + 1 >= 0)
#set0 = (i)[N, M] : (i >= 0, -i + N >= 0, N - 5 == 0, -i + M + 1 >= 0)

// CHECK-DAG: #set{{[0-9]+}} = (d0, d1)[s0] : (d0 >= 0, d1 >= 0)
#set1 = (d0, d1)[s0] : (d0 >= 0, d1 >= 0)

// CHECK-DAG: #set{{[0-9]+}} = (d0) : (d0 - 1 == 0)
#set2 = (d0) : (d0 - 1 == 0)

// CHECK-DAG: [[SET_TRUE:#set[0-9]+]] = () : (0 == 0)

// CHECK-DAG: #set{{[0-9]+}} = (d0)[s0] : (d0 - 2 >= 0, -d0 + 4 >= 0)

// CHECK: func @foo(i32, i64) -> f32
func @foo(i32, i64) -> f32

// CHECK: func @bar()
func @bar() -> ()

// CHECK: func @baz() -> (i1, index, f32)
func @baz() -> (i1, index, f32)

// CHECK: func @missingReturn()
func @missingReturn()

// CHECK: func @int_types(i1, i2, i4, i7, i87) -> (i1, index, i19)
func @int_types(i1, i2, i4, i7, i87) -> (i1, index, i19)


// CHECK: func @vectors(vector<1xf32>, vector<2x4xf32>)
func @vectors(vector<1 x f32>, vector<2x4xf32>)

// CHECK: func @tensors(tensor<*xf32>, tensor<*xvector<2x4xf32>>, tensor<1x?x4x?x?xi32>, tensor<i8>)
func @tensors(tensor<* x f32>, tensor<* x vector<2x4xf32>>,
                 tensor<1x?x4x?x?xi32>, tensor<i8>)

// CHECK: func @memrefs(memref<1x?x4x?x?xi32, #map{{[0-9]+}}>, memref<8xi8>)
func @memrefs(memref<1x?x4x?x?xi32, #map0>, memref<8xi8, #map1, #map1>)

// Test memref affine map compositions.

// CHECK: func @memrefs2(memref<2x4x8xi8, 1>)
func @memrefs2(memref<2x4x8xi8, #map2, 1>)

// CHECK: func @memrefs23(memref<2x4x8xi8, #map{{[0-9]+}}>)
func @memrefs23(memref<2x4x8xi8, #map2, #map3, 0>)

// CHECK: func @memrefs234(memref<2x4x8xi8, #map{{[0-9]+}}, #map{{[0-9]+}}, 3>)
func @memrefs234(memref<2x4x8xi8, #map2, #map3, #map4, 3>)

// Test memref inline affine map compositions, minding that identity maps are removed.

// CHECK: func @memrefs3(memref<2x4x8xi8>)
func @memrefs3(memref<2x4x8xi8, (d0, d1, d2) -> (d0, d1, d2)>)

// CHECK: func @memrefs33(memref<2x4x8xi8, #map{{[0-9]+}}, 1>)
func @memrefs33(memref<2x4x8xi8, (d0, d1, d2) -> (d0, d1, d2), (d0, d1, d2) -> (d1, d0, d2), 1>)

// CHECK: func @memrefs_drop_triv_id_inline(memref<2xi8>)
func @memrefs_drop_triv_id_inline(memref<2xi8, (d0) -> (d0)>)

// CHECK: func @memrefs_drop_triv_id_inline0(memref<2xi8>)
func @memrefs_drop_triv_id_inline0(memref<2xi8, (d0) -> (d0), 0>)

// CHECK: func @memrefs_drop_triv_id_inline1(memref<2xi8, 1>)
func @memrefs_drop_triv_id_inline1(memref<2xi8, (d0) -> (d0), 1>)

// Identity maps should be dropped from the composition, but not the pair of
// "interchange" maps that, if composed, would be also an identity.
// CHECK: func @memrefs_drop_triv_id_composition(memref<2x2xi8, #map{{[0-9]+}}, #map{{[0-9]+}}>)
func @memrefs_drop_triv_id_composition(memref<2x2xi8,
                                                (d0, d1) -> (d1, d0),
                                                (d0, d1) -> (d0, d1),
                                                (d0, d1) -> (d1, d0),
                                                (d0, d1) -> (d0, d1),
                                                (d0, d1) -> (d0, d1)>)

// CHECK: func @memrefs_drop_triv_id_trailing(memref<2x2xi8, #map{{[0-9]+}}>)
func @memrefs_drop_triv_id_trailing(memref<2x2xi8, (d0, d1) -> (d1, d0),
                                              (d0, d1) -> (d0, d1)>)

// CHECK: func @memrefs_drop_triv_id_middle(memref<2x2xi8, #map{{[0-9]+}}, #map{{[0-9]+}}>)
func @memrefs_drop_triv_id_middle(memref<2x2xi8, (d0, d1) -> (d0, d1 + 1),
                                            (d0, d1) -> (d0, d1),
					    (d0, d1) -> (d0 + 1, d1)>)

// CHECK: func @memrefs_drop_triv_id_multiple(memref<2xi8>)
func @memrefs_drop_triv_id_multiple(memref<2xi8, (d0) -> (d0), (d0) -> (d0)>)

// These maps appeared before, so they must be uniqued and hoisted to the beginning.
// Identity map should be removed.
// CHECK: func @memrefs_compose_with_id(memref<2x2xi8, #map{{[0-9]+}}>)
func @memrefs_compose_with_id(memref<2x2xi8, (d0, d1) -> (d0, d1),
                                        (d0, d1) -> (d1, d0)>)

// CHECK: func @functions((memref<1x?x4x?x?xi32, #map0>, memref<8xi8>) -> (), () -> ())
func @functions((memref<1x?x4x?x?xi32, #map0, 0>, memref<8xi8, #map1, 0>) -> (), ()->())

// CHECK-LABEL: func @simpleCFG(%arg0: i32, %arg1: f32) -> i1 {
func @simpleCFG(%arg0: i32, %f: f32) -> i1 {
  // CHECK: %0 = "foo"() : () -> i64
  %1 = "foo"() : ()->i64
  // CHECK: "bar"(%0) : (i64) -> (i1, i1, i1)
  %2 = "bar"(%1) : (i64) -> (i1,i1,i1)
  // CHECK: return %1#1
  return %2#1 : i1
// CHECK: }
}

// CHECK-LABEL: func @simpleCFGUsingBBArgs(%arg0: i32, %arg1: i64) {
func @simpleCFGUsingBBArgs(i32, i64) {
^bb42 (%arg0: i32, %f: i64):
  // CHECK: "bar"(%arg1) : (i64) -> (i1, i1, i1)
  %2 = "bar"(%f) : (i64) -> (i1,i1,i1)
  // CHECK: return{{$}}
  return
// CHECK: }
}

// CHECK-LABEL: func @multiblock() {
func @multiblock() {
  return     // CHECK:   return
^bb1:         // CHECK: ^bb1:   // no predecessors
  br ^bb4     // CHECK:   br ^bb3
^bb2:         // CHECK: ^bb2:   // pred: ^bb2
  br ^bb2     // CHECK:   br ^bb2
^bb4:         // CHECK: ^bb3:   // pred: ^bb1
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: func @emptyMLF() {
func @emptyMLF() {
  return     // CHECK:  return
}            // CHECK: }

// CHECK-LABEL: func @func_with_one_arg(%arg0: i1) -> i2 {
func @func_with_one_arg(%c : i1) -> i2 {
  // CHECK: %0 = "foo"(%arg0) : (i1) -> i2
  %b = "foo"(%c) : (i1) -> (i2)
  return %b : i2   // CHECK: return %0 : i2
} // CHECK: }

// CHECK-LABEL: func @func_with_two_args(%arg0: f16, %arg1: i8) -> (i1, i32) {
func @func_with_two_args(%a : f16, %b : i8) -> (i1, i32) {
  // CHECK: %0 = "foo"(%arg0, %arg1) : (f16, i8) -> (i1, i32)
  %c = "foo"(%a, %b) : (f16, i8)->(i1, i32)
  return %c#0, %c#1 : i1, i32  // CHECK: return %0#0, %0#1 : i1, i32
} // CHECK: }

// CHECK-LABEL: func @func_ops_in_loop() {
func @func_ops_in_loop() {
  // CHECK: %0 = "foo"() : () -> i64
  %a = "foo"() : ()->i64
  // CHECK: for %i0 = 1 to 10 {
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


// CHECK-LABEL: func @loops() {
func @loops() {
  // CHECK: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: for %i1 = 1 to 200 {
    for %j = 1 to 200 {
    }        // CHECK:     }
  }          // CHECK:   }
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: func @complex_loops() {
func @complex_loops() {
  for %i1 = 1 to 100 {      // CHECK:   for %i0 = 1 to 100 {
    for %j1 = 1 to 100 {    // CHECK:     for %i1 = 1 to 100 {
       // CHECK: "foo"(%i0, %i1) : (index, index) -> ()
       "foo"(%i1, %j1) : (index,index) -> ()
    }                       // CHECK:     }
    "boo"() : () -> ()      // CHECK:     "boo"() : () -> ()
    for %j2 = 1 to 10 {     // CHECK:     for %i2 = 1 to 10 {
      for %k2 = 1 to 10 {   // CHECK:       for %i3 = 1 to 10 {
        "goo"() : () -> ()  // CHECK:         "goo"() : () -> ()
      }                     // CHECK:       }
    }                       // CHECK:     }
  }                         // CHECK:   }
  return                    // CHECK:   return
}                           // CHECK: }

// CHECK: func @triang_loop(%arg0: index, %arg1: memref<?x?xi32>) {
func @triang_loop(%arg0: index, %arg1: memref<?x?xi32>) {
  %c = constant 0 : i32       // CHECK: %c0_i32 = constant 0 : i32
  for %i0 = 1 to %arg0 {      // CHECK: for %i0 = 1 to %arg0 {
    for %i1 = (d0)[]->(d0)(%i0)[] to %arg0 {  // CHECK:   for %i1 = #map{{[0-9]+}}(%i0) to %arg0 {
      store %c, %arg1[%i0, %i1] : memref<?x?xi32>  // CHECK: store %c0_i32, %arg1[%i0, %i1]
    }          // CHECK:     }
  }            // CHECK:   }
  return       // CHECK:   return
}              // CHECK: }

// CHECK: func @minmax_loop(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
func @minmax_loop(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
  // CHECK: for %i0 = max #map{{.*}}()[%arg0] to min #map{{.*}}()[%arg1] {
  for %i0 = max()[s]->(0,s-1)()[%arg0] to min()[s]->(100,s+1)()[%arg1] {
    // CHECK: "foo"(%arg2, %i0) : (memref<100xf32>, index) -> ()
    "foo"(%arg2, %i0) : (memref<100xf32>, index) -> ()
  }      // CHECK:   }
  return // CHECK:   return
}        // CHECK: }

// CHECK-LABEL: func @loop_bounds(%arg0: index) {
func @loop_bounds(%N : index) {
  // CHECK: %0 = "foo"(%arg0) : (index) -> index
  %s = "foo"(%N) : (index) -> index
  // CHECK: for %i0 = %0 to %arg0
  for %i = %s to %N {
    // CHECK: for %i1 = #map{{[0-9]+}}(%i0) to 0
    for %j = (d0)[]->(d0)(%i)[] to 0 step 1 {
       // CHECK: %1 = affine_apply #map{{.*}}(%i0, %i1)[%0]
       %w1 = affine_apply(d0, d1)[s0] -> (d0+d1) (%i, %j) [%s]
       // CHECK: %2 = affine_apply #map{{.*}}(%i0, %i1)[%0]
       %w2 = affine_apply(d0, d1)[s0] -> (s0+1) (%i, %j) [%s]
       // CHECK: for %i2 = #map{{.*}}(%1, %i0)[%arg0] to #map{{.*}}(%2, %i1)[%0] {
       for %k = #bound_map1 (%w1, %i)[%N] to (i, j)[s] -> (i + j + s) (%w2, %j)[%s] {
          // CHECK: "foo"(%i0, %i1, %i2) : (index, index, index) -> ()
          "foo"(%i, %j, %k) : (index, index, index)->()
          // CHECK: %c30 = constant 30 : index
          %c = constant 30 : index
          // CHECK: %3 = affine_apply #map{{.*}}(%arg0, %c30)
          %u = affine_apply (d0, d1)->(d0+d1) (%N, %c)
          // CHECK: for %i3 = max #map{{.*}}(%i0)[%3] to min #map{{.*}}(%i2)[%c30] {
          for %l = max #bound_map2(%i)[%u] to min #bound_map2(%k)[%c] {
            // CHECK: "bar"(%i3) : (index) -> ()
            "bar"(%l) : (index) -> ()
          } // CHECK:           }
       }    // CHECK:         }
     }      // CHECK:       }
  }         // CHECK:     }
  return    // CHECK:   return
}           // CHECK: }

// CHECK-LABEL: func @ifinst(%arg0: index) {
func @ifinst(%N: index) {
  %c = constant 200 : index // CHECK   %c200 = constant 200
  for %i = 1 to 10 {           // CHECK   for %i0 = 1 to 10 {
    if #set0(%i)[%N, %c] {     // CHECK     if #set0(%i0)[%arg0, %c200] {
      %x = constant 1 : i32
       // CHECK: %c1_i32 = constant 1 : i32
      %y = "add"(%x, %i) : (i32, index) -> i32 // CHECK: %0 = "add"(%c1_i32, %i0) : (i32, index) -> i32
      %z = "mul"(%y, %y) : (i32, i32) -> i32 // CHECK: %1 = "mul"(%0, %0) : (i32, i32) -> i32
    } else { // CHECK } else {
      if (i)[N] : (i - 2 >= 0, 4 - i >= 0)(%i)[%N]  {      // CHECK  if (#set1(%i0)[%arg0]) {
        // CHECK: %c1 = constant 1 : index
        %u = constant 1 : index
        // CHECK: %2 = affine_apply #map{{.*}}(%i0, %i0)[%c1]
        %w = affine_apply (d0,d1)[s0] -> (d0+d1+s0) (%i, %i) [%u]
      } else {            // CHECK     } else {
        %v = constant 3 : i32 // %c3_i32 = constant 3 : i32
      }
    }       // CHECK     }
  }         // CHECK   }
  return    // CHECK   return
}           // CHECK }

// CHECK-LABEL: func @simple_ifinst(%arg0: index) {
func @simple_ifinst(%N: index) {
  %c = constant 200 : index // CHECK   %c200 = constant 200
  for %i = 1 to 10 {           // CHECK   for %i0 = 1 to 10 {
    if #set0(%i)[%N, %c] {     // CHECK     if #set0(%i0)[%arg0, %c200] {
      %x = constant 1 : i32
       // CHECK: %c1_i32 = constant 1 : i32
      %y = "add"(%x, %i) : (i32, index) -> i32 // CHECK: %0 = "add"(%c1_i32, %i0) : (i32, index) -> i32
      %z = "mul"(%y, %y) : (i32, i32) -> i32 // CHECK: %1 = "mul"(%0, %0) : (i32, i32) -> i32
    }       // CHECK     }
  }         // CHECK   }
  return    // CHECK   return
}           // CHECK }

// CHECK-LABEL: func @attributes() {
func @attributes() {
  // CHECK: "foo"()
  "foo"(){} : ()->()

  // CHECK: "foo"() {a: 1, b: -423, c: [true, false], d: 1.600000e+01}  : () -> ()
  "foo"() {a: 1, b: -423, c: [true, false], d: 16.0 } : () -> ()

  // CHECK: "foo"() {map1: #map{{[0-9]+}}}
  "foo"() {map1: #map1} : () -> ()

  // CHECK: "foo"() {map2: #map{{[0-9]+}}}
  "foo"() {map2: (d0, d1, d2) -> (d0, d1, d2)} : () -> ()

  // CHECK: "foo"() {map12: [#map{{[0-9]+}}, #map{{[0-9]+}}]}
  "foo"() {map12: [#map1, #map2]} : () -> ()

  // CHECK: "foo"() {set1: #set{{[0-9]+}}}
  "foo"() {set1: #set1} : () -> ()

  // CHECK: "foo"() {set2: #set{{[0-9]+}}}
  "foo"() {set2: (d0, d1, d2) : (d0 >= 0, d1 >= 0, d2 - d1 == 0)} : () -> ()

  // CHECK: "foo"() {set12: [#set{{[0-9]+}}, #set{{[0-9]+}}]}
  "foo"() {set12: [#set1, #set2]} : () -> ()

  // CHECK: "foo"() {d: 1.000000e-09, func: [], i123: 7, if: "foo"} : () -> ()
  "foo"() {if: "foo", func: [], i123: 7, d: 1.e-9} : () -> ()

  // CHECK: "foo"() {fn: @attributes : () -> (), if: @ifinst : (index) -> ()} : () -> ()
  "foo"() {fn: @attributes : () -> (), if: @ifinst : (index) -> ()} : () -> ()
  return
}

// CHECK-LABEL: func @ssa_values() -> (i16, i8) {
func @ssa_values() -> (i16, i8) {
  // CHECK: %0 = "foo"() : () -> (i1, i17)
  %0 = "foo"() : () -> (i1, i17)
  br ^bb2

^bb1:       // CHECK: ^bb1: // pred: ^bb2
  // CHECK: %1 = "baz"(%2#1, %2#0, %0#1) : (f32, i11, i17) -> (i16, i8)
  %1 = "baz"(%2#1, %2#0, %0#1) : (f32, i11, i17) -> (i16, i8)

  // CHECK: return %1#0, %1#1 : i16, i8
  return %1#0, %1#1 : i16, i8

^bb2:       // CHECK: ^bb2:  // pred: ^bb0
  // CHECK: %2 = "bar"(%0#0, %0#1) : (i1, i17) -> (i11, f32)
  %2 = "bar"(%0#0, %0#1) : (i1, i17) -> (i11, f32)
  br ^bb1
}

// CHECK-LABEL: func @bbargs() -> (i16, i8) {
func @bbargs() -> (i16, i8) {
  // CHECK: %0 = "foo"() : () -> (i1, i17)
  %0 = "foo"() : () -> (i1, i17)
  br ^bb1(%0#1, %0#0 : i17, i1)

^bb1(%x: i17, %y: i1):       // CHECK: ^bb1(%1: i17, %2: i1):
  // CHECK: %3 = "baz"(%1, %2, %0#1) : (i17, i1, i17) -> (i16, i8)
  %1 = "baz"(%x, %y, %0#1) : (i17, i1, i17) -> (i16, i8)
  return %1#0, %1#1 : i16, i8
}

// CHECK-LABEL: func @verbose_terminators() -> (i1, i17)
func @verbose_terminators() -> (i1, i17) {
  %0 = "foo"() : () -> (i1, i17)
// CHECK:  br ^bb1(%0#0, %0#1 : i1, i17)
  "br"()[^bb1(%0#0, %0#1 : i1, i17)] : () -> ()

^bb1(%x : i1, %y : i17):
// CHECK:  cond_br %1, ^bb2(%2 : i17), ^bb3(%1, %2 : i1, i17)
  "cond_br"(%x)[^bb2(%y : i17), ^bb3(%x, %y : i1, i17)] : (i1) -> ()

^bb2(%a : i17):
  %true = constant 1 : i1
// CHECK:  return %true, %3 : i1, i17
  "return"(%true, %a) : (i1, i17) -> ()

^bb3(%b : i1, %c : i17):
// CHECK:  return %4, %5 : i1, i17
  "return"(%b, %c) : (i1, i17) -> ()
}

// CHECK-LABEL: func @condbr_simple
func @condbr_simple() -> (i32) {
  %cond = "foo"() : () -> i1
  %a = "bar"() : () -> i32
  %b = "bar"() : () -> i64
  // CHECK: cond_br %0, ^bb1(%1 : i32), ^bb2(%2 : i64)
  cond_br %cond, ^bb1(%a : i32), ^bb2(%b : i64)

// CHECK: ^bb1({{.*}}: i32): // pred: ^bb0
^bb1(%x : i32):
  br ^bb2(%b: i64)

// CHECK: ^bb2({{.*}}: i64): // 2 preds: ^bb0, ^bb1
^bb2(%y : i64):
  %z = "foo"() : () -> i32
  return %z : i32
}

// CHECK-LABEL: func @condbr_moarargs
func @condbr_moarargs() -> (i32) {
  %cond = "foo"() : () -> i1
  %a = "bar"() : () -> i32
  %b = "bar"() : () -> i64
  // CHECK: cond_br %0, ^bb1(%1, %2 : i32, i64), ^bb2(%2, %1, %1 : i64, i32, i32)
  cond_br %cond, ^bb1(%a, %b : i32, i64), ^bb2(%b, %a, %a : i64, i32, i32)

^bb1(%x : i32, %y : i64):
  return %x : i32

^bb2(%x2 : i64, %y2 : i32, %z2 : i32):
  %z = "foo"() : () -> i32
  return %z : i32
}


// Test pretty printing of constant names.
// CHECK-LABEL: func @constants
func @constants() -> (i32, i23, i23, i1, i1) {
  // CHECK: %c42_i32 = constant 42 : i32
  %x = constant 42 : i32
  // CHECK: %c17_i23 = constant 17 : i23
  %y = constant 17 : i23

  // This is a redundant definition of 17, the asmprinter gives it a unique name
  // CHECK: %c17_i23_0 = constant 17 : i23
  %z = constant 17 : i23

  // CHECK: %true = constant 1 : i1
  %t = constant 1 : i1
  // CHECK: %false = constant 0 : i1
  %f = constant 0 : i1

  // CHECK: return %c42_i32, %c17_i23, %c17_i23_0, %true, %false
  return %x, %y, %z, %t, %f : i32, i23, i23, i1, i1
}

// CHECK-LABEL: func @typeattr
func @typeattr() -> () {
^bb0:
// CHECK: "foo"() {bar: tensor<*xf32>} : () -> ()
  "foo"(){bar: tensor<*xf32>} : () -> ()
  return
}

// CHECK-LABEL: func @stringquote
func @stringquote() -> () {
^bb0:
  // CHECK: "foo"() {bar: "a\22quoted\22string"} : () -> ()
  "foo"(){bar: "a\"quoted\"string"} : () -> ()
  return
}

// CHECK-LABEL: func @floatAttrs
func @floatAttrs() -> () {
^bb0:
  // CHECK: "foo"() {a: 4.000000e+00, b: 2.000000e+00, c: 7.100000e+00, d: -0.000000e+00} : () -> ()
  "foo"(){a: 4.0, b: 2.0, c: 7.1, d: -0.0} : () -> ()
  return
}

// CHECK-LABEL: func @externalfuncattr
func @externalfuncattr() -> ()
  // CHECK: attributes {a: "a\22quoted\22string", b: 4.000000e+00, c: tensor<*xf32>}
  attributes {a: "a\"quoted\"string", b: 4.0, c: tensor<*xf32>}

// CHECK-LABEL: func @funcattrempty
func @funcattrempty() -> ()
  // CHECK-EMPTY
  attributes {}

// CHECK-LABEL: func @funcattr
func @funcattr() -> ()
  // CHECK: attributes {a: "a\22quoted\22string", b: 4.000000e+00, c: tensor<*xf32>}
  attributes {a: "a\"quoted\"string", b: 4.0, c: tensor<*xf32>} {
^bb0:
  return
}

// CHECK-LABEL: func @funcattrwithblock
func @funcattrwithblock() -> ()
  // CHECK-EMPTY
  attributes {} {
^bb0:
  return
}

// CHECK-label func @funcsimplemap
#map_simple0 = ()[] -> (10)
#map_simple1 = ()[s0] -> (s0)
#map_non_simple0 = (d0)[] -> (d0)
#map_non_simple1 = (d0)[s0] -> (d0 + s0)
#map_non_simple2 = ()[s0, s1] -> (s0 + s1)
#map_non_simple3 = ()[s0] -> (s0 + 3)
func @funcsimplemap(%arg0: index, %arg1: index) -> () {
  for %i0 = 0 to #map_simple0()[] {
  // CHECK: for %i0 = 0 to 10 {
    for %i1 = 0 to #map_simple1()[%arg1] {
    // CHECK: for %i1 = 0 to %arg1 {
      for %i2 = 0 to #map_non_simple0(%i0)[] {
      // CHECK: for %i2 = 0 to #map{{[a-z_0-9]*}}(%i0) {
        for %i3 = 0 to #map_non_simple1(%i0)[%arg1] {
        // CHECK: for %i3 = 0 to #map{{[a-z_0-9]*}}(%i0)[%arg1] {
          for %i4 = 0 to #map_non_simple2()[%arg1, %arg0] {
          // CHECK: for %i4 = 0 to #map{{[a-z_0-9]*}}()[%arg1, %arg0] {
            for %i5 = 0 to #map_non_simple3()[%arg0] {
            // CHECK: for %i5 = 0 to #map{{[a-z_0-9]*}}()[%arg0] {
              %c42_i32 = constant 42 : i32
            }
          }
        }
      }
    }
  }
  return
}

// CHECK-LABEL: func @splattensorattr
func @splattensorattr() -> () {
^bb0:
// CHECK: "splatIntTensor"() {bar: splat<tensor<2x1x4xi32>, 5>} : () -> ()
  "splatIntTensor"(){bar: splat<tensor<2x1x4xi32>, 5>} : () -> ()
// CHECK: "splatFloatTensor"() {bar: splat<tensor<2x1x4xf32>, -5.000000e+00>} : () -> ()
  "splatFloatTensor"(){bar: splat<tensor<2x1x4xf32>, -5.0>} : () -> ()
// CHECK: "splatIntVector"() {bar: splat<vector<2x1x4xi64>, 5>} : () -> ()
  "splatIntVector"(){bar: splat<vector<2x1x4xi64>, 5>} : () -> ()
// CHECK: "splatFloatVector"() {bar: splat<vector<2x1x4xf16>, -5.000000e+00>} : () -> ()
  "splatFloatVector"(){bar: splat<vector<2x1x4xf16>, -5.0>} : () -> ()
  return
}

// CHECK-LABEL: func @opaquetensorattr
func @opaquetensorattr() -> () {
^bb0:
// CHECK: "opaqueIntTensor"() {bar: opaque<tensor<2x1x4xi32>, "0x68656C6C6F">} : () -> ()
  "opaqueIntTensor"(){bar: opaque<tensor<2x1x4xi32>, "0x68656C6C6F">} : () -> ()
// CHECK: "opaqueFloatTensor"() {bar: opaque<tensor<2x1x4xf32>, "0x68656C6C6F">} : () -> ()
  "opaqueFloatTensor"(){bar: opaque<tensor<2x1x4xf32>, "0x68656C6C6F">} : () -> ()
  
// CHECK: "opaqueStringTensor"() {bar: opaque<tensor<2x1x4x!tf<"string">>, "0x68656C6C6F">} : () -> ()
  "opaqueStringTensor"(){bar: opaque<tensor<2x1x4x!tf<"string">>, "0x68656C6C6F">} : () -> ()
// CHECK: "opaqueResourceTensor"() {bar: opaque<tensor<2x1x4x!tf<"resource">>, "0x68656C6C6F">} : () -> ()
  "opaqueResourceTensor"(){bar: opaque<tensor<2x1x4x!tf<"resource">>, "0x68656C6C6F">} : () -> ()
  return
}

// CHECK-LABEL: func @densetensorattr
func @densetensorattr() -> () {
^bb0:

// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi3"() {bar: dense<tensor<2x1x4xi3>, {{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]>} : () -> ()
  "fooi3"(){bar: dense<tensor<2x1x4xi3>, [[[1, -2, 1, 2]], [[0, 2, -1, 2]]]>} : () -> ()
// CHECK: "fooi6"() {bar: dense<tensor<2x1x4xi6>, {{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]>} : () -> ()
  "fooi6"(){bar: dense<tensor<2x1x4xi6>, [[[5, -6, 1, 2]], [[7, 8, 3, 4]]]>} : () -> ()
// CHECK: "fooi8"() {bar: dense<tensor<1x1x1xi8>, {{\[\[\[}}5]]]>} : () -> ()
  "fooi8"(){bar: dense<tensor<1x1x1xi8>, [[[5]]]>} : () -> ()
// CHECK: "fooi13"() {bar: dense<tensor<2x1x4xi13>, {{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]>} : () -> ()
  "fooi13"(){bar: dense<tensor<2x1x4xi13>, [[[1, -2, 1, 2]], [[0, 2, -1, 2]]]>} : () -> ()
// CHECK: "fooi16"() {bar: dense<tensor<1x1x1xi16>, {{\[\[\[}}-5]]]>} : () -> ()
  "fooi16"(){bar: dense<tensor<1x1x1xi16>, [[[-5]]]>} : () -> ()
// CHECK: "fooi23"() {bar: dense<tensor<2x1x4xi23>, {{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]>} : () -> ()
  "fooi23"(){bar: dense<tensor<2x1x4xi23>, [[[1, -2, 1, 2]], [[0, 2, -1, 2]]]>} : () -> ()
// CHECK: "fooi32"() {bar: dense<tensor<1x1x1xi32>, {{\[\[\[}}5]]]>} : () -> ()
  "fooi32"(){bar: dense<tensor<1x1x1xi32>, [[[5]]]>} : () -> ()
// CHECK: "fooi33"() {bar: dense<tensor<2x1x4xi33>, {{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]>} : () -> ()
  "fooi33"(){bar: dense<tensor<2x1x4xi33>, [[[1, -2, 1, 2]], [[0, 2, -1, 2]]]>} : () -> ()
// CHECK: "fooi43"() {bar: dense<tensor<2x1x4xi43>, {{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]>} : () -> ()
  "fooi43"(){bar: dense<tensor<2x1x4xi43>, [[[1, -2, 1, 2]], [[0, 2, -1, 2]]]>} : () -> ()
// CHECK: "fooi53"() {bar: dense<tensor<2x1x4xi53>, {{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]>} : () -> ()
  "fooi53"(){bar: dense<tensor<2x1x4xi53>, [[[1, -2, 1, 2]], [[0, 2, -1, 2]]]>} : () -> ()
// CHECK: "fooi64"() {bar: dense<tensor<2x1x4xi64>, {{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 3, -1, 2]]]>} : () -> ()
  "fooi64"(){bar: dense<tensor<2x1x4xi64>, [[[1, -2, 1, 2]], [[0, 3, -1, 2]]]>} : () -> ()
// CHECK: "fooi64"() {bar: dense<tensor<1x1x1xi64>, {{\[\[\[}}-5]]]>} : () -> ()
  "fooi64"(){bar: dense<tensor<1x1x1xi64>, [[[-5]]]>} : () -> ()
// CHECK: "fooi67"() {bar: dense<vector<1x1x4xi67>, {{\[\[\[}}-5, 4, 6, 2]]]>} : () -> ()
  "fooi67"(){bar: dense<vector<1x1x4xi67>, [[[-5, 4, 6, 2]]]>} : () -> ()

// CHECK: "foo2"() {bar: dense<tensor<0xi32>, []>} : () -> ()
  "foo2"(){bar: dense<tensor<0 x i32>, []>} : () -> ()
// CHECK: "foo2"() {bar: dense<tensor<1x0xi32>, {{\[\[}}]]>} : () -> ()
  "foo2"(){bar: dense<tensor<1x0 x i32>, [[]]>} : () -> ()
// CHECK: "foo3"() {bar: dense<tensor<2x1x4xi32>, {{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]>} : () -> ()
  "foo3"(){bar: dense<tensor<2x1x4xi32>, [[[5, -6, 1, 2]], [[7, 8, 3, 4]]]>} : () -> ()

// CHECK: "float1"() {bar: dense<tensor<1x1x1xf32>, {{\[\[\[}}5.000000e+00]]]>} : () -> ()
  "float1"(){bar: dense<tensor<1x1x1xf32>, [[[5.0]]]>} : () -> ()
// CHECK: "float2"() {bar: dense<tensor<0xf32>, []>} : () -> ()
  "float2"(){bar: dense<tensor<0 x f32>, []>} : () -> ()
// CHECK: "float2"() {bar: dense<tensor<1x0xf32>, {{\[\[}}]]>} : () -> ()
  "float2"(){bar: dense<tensor<1x0 x f32>, [[]]>} : () -> ()

// CHECK: "bfloat16"() {bar: dense<tensor<2x1x4xbf16>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "bfloat16"(){bar: dense<tensor<2x1x4xbf16>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
// CHECK: "float16"() {bar: dense<tensor<2x1x4xf16>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "float16"(){bar: dense<tensor<2x1x4xf16>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
// CHECK: "float32"() {bar: dense<tensor<2x1x4xf32>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "float32"(){bar: dense<tensor<2x1x4xf32>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
// CHECK: "float64"() {bar: dense<tensor<2x1x4xf64>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "float64"(){bar: dense<tensor<2x1x4xf64>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
  return
}

// CHECK-LABEL: func @densevectorattr
func @densevectorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar: dense<vector<1x1x1xi8>, {{\[\[\[}}5]]]>} : () -> ()
  "fooi8"(){bar: dense<vector<1x1x1xi8>, [[[5]]]>} : () -> ()
// CHECK: "fooi16"() {bar: dense<vector<1x1x1xi16>, {{\[\[\[}}-5]]]>} : () -> ()
  "fooi16"(){bar: dense<vector<1x1x1xi16>, [[[-5]]]>} : () -> ()
// CHECK: "foo32"() {bar: dense<vector<1x1x1xi32>, {{\[\[\[}}5]]]>} : () -> ()
  "foo32"(){bar: dense<vector<1x1x1xi32>, [[[5]]]>} : () -> ()
// CHECK: "fooi64"() {bar: dense<vector<1x1x1xi64>, {{\[\[\[}}-5]]]>} : () -> ()
  "fooi64"(){bar: dense<vector<1x1x1xi64>, [[[-5]]]>} : () -> ()

// CHECK: "foo2"() {bar: dense<vector<0xi32>, []>} : () -> ()
  "foo2"(){bar: dense<vector<0 x i32>, []>} : () -> ()
// CHECK: "foo2"() {bar: dense<vector<1x0xi32>, {{\[\[}}]]>} : () -> ()
  "foo2"(){bar: dense<vector<1x0 x i32>, [[]]>} : () -> ()
// CHECK: "foo3"() {bar: dense<vector<2x1x4xi32>, {{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]>} : () -> ()
  "foo3"(){bar: dense<vector<2x1x4xi32>, [[[5, -6, 1, 2]], [[7, 8, 3, 4]]]>} : () -> ()

// CHECK: "float1"() {bar: dense<vector<1x1x1xf32>, {{\[\[\[}}5.000000e+00]]]>} : () -> ()
  "float1"(){bar: dense<vector<1x1x1xf32>, [[[5.0]]]>} : () -> ()
// CHECK: "float2"() {bar: dense<vector<0xf32>, []>} : () -> ()
  "float2"(){bar: dense<vector<0 x f32>, []>} : () -> ()
// CHECK: "float2"() {bar: dense<vector<1x0xf32>, {{\[\[}}]]>} : () -> ()
  "float2"(){bar: dense<vector<1x0 x f32>, [[]]>} : () -> ()

// CHECK: "bfloat16"() {bar: dense<vector<2x1x4xbf16>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "bfloat16"(){bar: dense<vector<2x1x4xbf16>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
// CHECK: "float16"() {bar: dense<vector<2x1x4xf16>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "float16"(){bar: dense<vector<2x1x4xf16>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
// CHECK: "float32"() {bar: dense<vector<2x1x4xf32>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "float32"(){bar: dense<vector<2x1x4xf32>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
// CHECK: "float64"() {bar: dense<vector<2x1x4xf64>, {{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]>} : () -> ()
  "float64"(){bar: dense<vector<2x1x4xf64>, [[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]>} : () -> ()
  return
}

// CHECK-LABEL: func @sparsetensorattr
func @sparsetensorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar: sparse<tensor<1x1x1xi8>, {{\[\[}}0, 0, 0]], {{\[}}-2]>} : () -> ()
  "fooi8"(){bar: sparse<tensor<1x1x1xi8>, [[0, 0, 0]], [-2]>} : () -> ()
// CHECK: "fooi16"() {bar: sparse<tensor<2x2x2xi16>, {{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2, -1, 5]>} : () -> ()
  "fooi16"(){bar: sparse<tensor<2x2x2xi16>, [[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]>} : () -> ()
// CHECK: "fooi32"() {bar: sparse<tensor<1x1xi32>, {{\[}}], {{\[}}]>} : () -> ()
  "fooi32"(){bar: sparse<tensor<1x1xi32>, [], []>} : () -> ()
// CHECK: "fooi64"() {bar: sparse<tensor<1xi64>, {{\[\[}}0]], {{\[}}-1]>} : () -> ()
  "fooi64"(){bar: sparse<tensor<1xi64>, [[0]], [-1]>} : () -> ()
// CHECK: "foo2"() {bar: sparse<tensor<0xi32>, {{\[}}], {{\[}}]>} : () -> ()
  "foo2"(){bar: sparse<tensor<0 x i32>, [], []>} : () -> ()
  
// CHECK: "foof16"() {bar: sparse<tensor<1x1x1xf16>, {{\[\[}}0, 0, 0]], {{\[}}-2.000000e+00]>} : () -> ()
  "foof16"(){bar: sparse<tensor<1x1x1xf16>, [[0, 0, 0]], [-2.0]>} : () -> ()
// CHECK: "foobf16"() {bar: sparse<tensor<2x2x2xbf16>, {{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2.000000e+00, -1.000000e+00, 5.000000e+00]>} : () -> ()
  "foobf16"(){bar: sparse<tensor<2x2x2xbf16>, [[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.0, -1.0, 5.0]>} : () -> ()
// CHECK: "foof32"() {bar: sparse<tensor<1x1xf32>, {{\[}}], {{\[}}]>} : () -> ()
  "foof32"(){bar: sparse<tensor<1x0x1xf32>, [], []>} : () -> ()
// CHECK:  "foof64"() {bar: sparse<tensor<1xf64>, {{\[\[}}0]], {{\[}}-1.000000e+00]>} : () -> ()
  "foof64"(){bar: sparse<tensor<1xf64>, [[0]], [-1.0]>} : () -> ()
// CHECK: "foof320"() {bar: sparse<tensor<0xf32>, {{\[}}], {{\[}}]>} : () -> ()
  "foof320"(){bar: sparse<tensor<0 x f32>, [], []>} : () -> ()
  return
}

// CHECK-LABEL: func @sparsevectorattr
func @sparsevectorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar: sparse<vector<1x1x1xi8>, {{\[\[}}0, 0, 0]], {{\[}}-2]>} : () -> ()
  "fooi8"(){bar: sparse<vector<1x1x1xi8>, [[0, 0, 0]], [-2]>} : () -> ()
// CHECK: "fooi16"() {bar: sparse<vector<2x2x2xi16>, {{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2, -1, 5]>} : () -> ()
  "fooi16"(){bar: sparse<vector<2x2x2xi16>, [[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]>} : () -> ()
// CHECK: "fooi32"() {bar: sparse<vector<1x1xi32>, {{\[}}], {{\[}}]>} : () -> ()
  "fooi32"(){bar: sparse<vector<1x1xi32>, [], []>} : () -> ()
// CHECK: "fooi64"() {bar: sparse<vector<1xi64>, {{\[\[}}0]], {{\[}}-1]>} : () -> ()
  "fooi64"(){bar: sparse<vector<1xi64>, [[0]], [-1]>} : () -> ()
// CHECK: "foo2"() {bar: sparse<vector<0xi32>, {{\[}}], {{\[}}]>} : () -> ()
  "foo2"(){bar: sparse<vector<0 x i32>, [], []>} : () -> ()

// CHECK: "foof16"() {bar: sparse<vector<1x1x1xf16>, {{\[\[}}0, 0, 0]], {{\[}}-2.000000e+00]>} : () -> ()
  "foof16"(){bar: sparse<vector<1x1x1xf16>, [[0, 0, 0]], [-2.0]>} : () -> ()
// CHECK: "foobf16"() {bar: sparse<vector<2x2x2xbf16>, {{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2.000000e+00, -1.000000e+00, 5.000000e+00]>} : () -> ()
  "foobf16"(){bar: sparse<vector<2x2x2xbf16>, [[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.0, -1.0, 5.0]>} : () -> ()
// CHECK: "foof32"() {bar: sparse<vector<1x1xf32>, {{\[}}], {{\[}}]>} : () -> ()
  "foof32"(){bar: sparse<vector<1x0x1xf32>, [], []>} : () -> ()
// CHECK:  "foof64"() {bar: sparse<vector<1xf64>, {{\[\[}}0]], {{\[}}-1.000000e+00]>} : () -> ()
  "foof64"(){bar: sparse<vector<1xf64>, [[0]], [-1.0]>} : () -> ()
// CHECK: "foof320"() {bar: sparse<vector<0xf32>, {{\[}}], {{\[}}]>} : () -> ()
  "foof320"(){bar: sparse<vector<0 x f32>, [], []>} : () -> ()
  return
}

// CHECK-LABEL: func @loops_with_blockids() {
func @loops_with_blockids() {
^block0:
  for %i = 1 to 100 step 2 {
  ^block1:
    for %j = 1 to 200 {
    ^block2:
    }
  }
  return
}

// CHECK-LABEL: func @unknown_dialect_type() -> !bar<""> {
func @unknown_dialect_type() -> !bar<""> {
  // Unregistered dialect 'bar'.
  // CHECK: "foo"() : () -> !bar<"">
  %0 = "foo"() : () -> !bar<"">

  // CHECK: "foo"() : () -> !bar<"baz">
  %1 = "foo"() : () -> !bar<"baz">

  return %0 : !bar<"">
}

// CHECK-LABEL: func @type_alias() -> i32 {
!i32_type_alias = type i32
func @type_alias() -> !i32_type_alias {

  // Return a non-aliased i32 type.
  %0 = "foo"() : () -> i32
  return %0 : i32
}

// CHECK-LABEL: func @no_integer_set_constraints(
func @no_integer_set_constraints() {
  // CHECK: if [[SET_TRUE]]() {
  if () : () () {
  }
  return
}

// CHECK-LABEL: func @verbose_if(
func @verbose_if(%N: index) {
  %c = constant 200 : index

  // CHECK: if #set0(%c200)[%arg0, %c200] {
  "if"(%c, %N, %c) { condition: #set0 } : (index, index, index) -> () {
    // CHECK-NEXT: "add"
    %y = "add"(%c, %N) : (index, index) -> index
    // CHECK-NEXT: } else {
  } { // The else block list.
    // CHECK-NEXT: "add"
    %z = "add"(%c, %c) : (index, index) -> index
  }
  return
}
