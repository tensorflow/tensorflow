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
// CHECK-NOT: Placeholder

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


// CHECK: func @complex_types(complex<i1>) -> complex<f32>
func @complex_types(complex<i1>) -> complex<f32>

// CHECK: func @functions((memref<1x?x4x?x?xi32, #map0>, memref<8xi8>) -> (), () -> ())
func @functions((memref<1x?x4x?x?xi32, #map0, 0>, memref<8xi8, #map1, 0>) -> (), ()->())

// CHECK-LABEL: func @simpleCFG(%{{.*}}: i32, %{{.*}}: f32) -> i1 {
func @simpleCFG(%arg0: i32, %f: f32) -> i1 {
  // CHECK: %{{.*}} = "foo"() : () -> i64
  %1 = "foo"() : ()->i64
  // CHECK: "bar"(%{{.*}}) : (i64) -> (i1, i1, i1)
  %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
  // CHECK: return %{{.*}}#1
  return %2#1 : i1
// CHECK: }
}

// CHECK-LABEL: func @simpleCFGUsingBBArgs(%{{.*}}: i32, %{{.*}}: i64) {
func @simpleCFGUsingBBArgs(i32, i64) {
^bb42 (%arg0: i32, %f: i64):
  // CHECK: "bar"(%{{.*}}) : (i64) -> (i1, i1, i1)
  %2:3 = "bar"(%f) : (i64) -> (i1,i1,i1)
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

// CHECK-LABEL: func @func_with_one_arg(%{{.*}}: i1) -> i2 {
func @func_with_one_arg(%c : i1) -> i2 {
  // CHECK: %{{.*}} = "foo"(%{{.*}}) : (i1) -> i2
  %b = "foo"(%c) : (i1) -> (i2)
  return %b : i2   // CHECK: return %{{.*}} : i2
} // CHECK: }

// CHECK-LABEL: func @func_with_two_args(%{{.*}}: f16, %{{.*}}: i8) -> (i1, i32) {
func @func_with_two_args(%a : f16, %b : i8) -> (i1, i32) {
  // CHECK: %{{.*}}:2 = "foo"(%{{.*}}, %{{.*}}) : (f16, i8) -> (i1, i32)
  %c:2 = "foo"(%a, %b) : (f16, i8)->(i1, i32)
  return %c#0, %c#1 : i1, i32  // CHECK: return %{{.*}}#0, %{{.*}}#1 : i1, i32
} // CHECK: }

// CHECK-LABEL: func @second_order_func() -> (() -> ()) {
func @second_order_func() -> (() -> ()) {
// CHECK-NEXT: %{{.*}} = constant @emptyMLF : () -> ()
  %c = constant @emptyMLF : () -> ()
// CHECK-NEXT: return %{{.*}} : () -> ()
  return %c : () -> ()
}

// CHECK-LABEL: func @third_order_func() -> (() -> (() -> ())) {
func @third_order_func() -> (() -> (() -> ())) {
// CHECK-NEXT:  %{{.*}} = constant @second_order_func : () -> (() -> ())
  %c = constant @second_order_func : () -> (() -> ())
// CHECK-NEXT:  return %{{.*}} : () -> (() -> ())
  return %c : () -> (() -> ())
}

// CHECK-LABEL: func @identity_functor(%{{.*}}: () -> ()) -> (() -> ())  {
func @identity_functor(%a : () -> ()) -> (() -> ())  {
// CHECK-NEXT: return %{{.*}} : () -> ()
  return %a : () -> ()
}

// CHECK-LABEL: func @func_ops_in_loop() {
func @func_ops_in_loop() {
  // CHECK: %{{.*}} = "foo"() : () -> i64
  %a = "foo"() : ()->i64
  // CHECK: affine.for %{{.*}} = 1 to 10 {
  affine.for %i = 1 to 10 {
    // CHECK: %{{.*}} = "doo"() : () -> f32
    %b = "doo"() : ()->f32
    // CHECK: "bar"(%{{.*}}, %{{.*}}) : (i64, f32) -> ()
    "bar"(%a, %b) : (i64, f32) -> ()
  // CHECK: }
  }
  // CHECK: return
  return
  // CHECK: }
}


// CHECK-LABEL: func @loops() {
func @loops() {
  // CHECK: affine.for %{{.*}} = 1 to 100 step 2 {
  affine.for %i = 1 to 100 step 2 {
    // CHECK: affine.for %{{.*}} = 1 to 200 {
    affine.for %j = 1 to 200 {
    }        // CHECK:     }
  }          // CHECK:   }
  return     // CHECK:   return
}            // CHECK: }

// CHECK-LABEL: func @complex_loops() {
func @complex_loops() {
  affine.for %i1 = 1 to 100 {      // CHECK:   affine.for %{{.*}} = 1 to 100 {
    affine.for %j1 = 1 to 100 {    // CHECK:     affine.for %{{.*}} = 1 to 100 {
       // CHECK: "foo"(%{{.*}}, %{{.*}}) : (index, index) -> ()
       "foo"(%i1, %j1) : (index,index) -> ()
    }                       // CHECK:     }
    "boo"() : () -> ()      // CHECK:     "boo"() : () -> ()
    affine.for %j2 = 1 to 10 {     // CHECK:     affine.for %{{.*}} = 1 to 10 {
      affine.for %k2 = 1 to 10 {   // CHECK:       affine.for %{{.*}} = 1 to 10 {
        "goo"() : () -> ()  // CHECK:         "goo"() : () -> ()
      }                     // CHECK:       }
    }                       // CHECK:     }
  }                         // CHECK:   }
  return                    // CHECK:   return
}                           // CHECK: }

// CHECK: func @triang_loop(%{{.*}}: index, %{{.*}}: memref<?x?xi32>) {
func @triang_loop(%arg0: index, %arg1: memref<?x?xi32>) {
  %c = constant 0 : i32       // CHECK: %{{.*}} = constant 0 : i32
  affine.for %i0 = 1 to %arg0 {      // CHECK: affine.for %{{.*}} = 1 to %{{.*}} {
    affine.for %i1 = (d0)[]->(d0)(%i0)[] to %arg0 {  // CHECK:   affine.for %{{.*}} = #map{{[0-9]+}}(%{{.*}}) to %{{.*}} {
      store %c, %arg1[%i0, %i1] : memref<?x?xi32>  // CHECK: store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
    }          // CHECK:     }
  }            // CHECK:   }
  return       // CHECK:   return
}              // CHECK: }

// CHECK: func @minmax_loop(%{{.*}}: index, %{{.*}}: index, %{{.*}}: memref<100xf32>) {
func @minmax_loop(%arg0: index, %arg1: index, %arg2: memref<100xf32>) {
  // CHECK: affine.for %{{.*}} = max #map{{.*}}()[%{{.*}}] to min #map{{.*}}()[%{{.*}}] {
  affine.for %i0 = max()[s]->(0,s-1)()[%arg0] to min()[s]->(100,s+1)()[%arg1] {
    // CHECK: "foo"(%{{.*}}, %{{.*}}) : (memref<100xf32>, index) -> ()
    "foo"(%arg2, %i0) : (memref<100xf32>, index) -> ()
  }      // CHECK:   }
  return // CHECK:   return
}        // CHECK: }

// CHECK-LABEL: func @loop_bounds(%{{.*}}: index) {
func @loop_bounds(%N : index) {
  // CHECK: %{{.*}} = "foo"(%{{.*}}) : (index) -> index
  %s = "foo"(%N) : (index) -> index
  // CHECK: affine.for %{{.*}} = %{{.*}} to %{{.*}}
  affine.for %i = %s to %N {
    // CHECK: affine.for %{{.*}} = #map{{[0-9]+}}(%{{.*}}) to 0
    affine.for %j = (d0)[]->(d0)(%i)[] to 0 step 1 {
       // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]
       %w1 = affine.apply(d0, d1)[s0] -> (d0+d1) (%i, %j) [%s]
       // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]
       %w2 = affine.apply(d0, d1)[s0] -> (s0+1) (%i, %j) [%s]
       // CHECK: affine.for %{{.*}} = #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}] to #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}] {
       affine.for %k = #bound_map1 (%w1, %i)[%N] to (i, j)[s] -> (i + j + s) (%w2, %j)[%s] {
          // CHECK: "foo"(%{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index) -> ()
          "foo"(%i, %j, %k) : (index, index, index)->()
          // CHECK: %{{.*}} = constant 30 : index
          %c = constant 30 : index
          // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})
          %u = affine.apply (d0, d1)->(d0+d1) (%N, %c)
          // CHECK: affine.for %{{.*}} = max #map{{.*}}(%{{.*}})[%{{.*}}] to min #map{{.*}}(%{{.*}})[%{{.*}}] {
          affine.for %l = max #bound_map2(%i)[%u] to min #bound_map2(%k)[%c] {
            // CHECK: "bar"(%{{.*}}) : (index) -> ()
            "bar"(%l) : (index) -> ()
          } // CHECK:           }
       }    // CHECK:         }
     }      // CHECK:       }
  }         // CHECK:     }
  return    // CHECK:   return
}           // CHECK: }

// CHECK-LABEL: func @ifinst(%{{.*}}: index) {
func @ifinst(%N: index) {
  %c = constant 200 : index // CHECK   %{{.*}} = constant 200
  affine.for %i = 1 to 10 {           // CHECK   affine.for %{{.*}} = 1 to 10 {
    affine.if #set0(%i)[%N, %c] {     // CHECK     affine.if #set0(%{{.*}})[%{{.*}}, %{{.*}}] {
      %x = constant 1 : i32
       // CHECK: %{{.*}} = constant 1 : i32
      %y = "add"(%x, %i) : (i32, index) -> i32 // CHECK: %{{.*}} = "add"(%{{.*}}, %{{.*}}) : (i32, index) -> i32
      %z = "mul"(%y, %y) : (i32, i32) -> i32 // CHECK: %{{.*}} = "mul"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    } else { // CHECK } else {
      affine.if (i)[N] : (i - 2 >= 0, 4 - i >= 0)(%i)[%N]  {      // CHECK  affine.if (#set1(%{{.*}})[%{{.*}}]) {
        // CHECK: %{{.*}} = constant 1 : index
        %u = constant 1 : index
        // CHECK: %{{.*}} = affine.apply #map{{.*}}(%{{.*}}, %{{.*}})[%{{.*}}]
        %w = affine.apply (d0,d1)[s0] -> (d0+d1+s0) (%i, %i) [%u]
      } else {            // CHECK     } else {
        %v = constant 3 : i32 // %c3_i32 = constant 3 : i32
      }
    }       // CHECK     }
  }         // CHECK   }
  return    // CHECK   return
}           // CHECK }

// CHECK-LABEL: func @simple_ifinst(%{{.*}}: index) {
func @simple_ifinst(%N: index) {
  %c = constant 200 : index // CHECK   %{{.*}} = constant 200
  affine.for %i = 1 to 10 {           // CHECK   affine.for %{{.*}} = 1 to 10 {
    affine.if #set0(%i)[%N, %c] {     // CHECK     affine.if #set0(%{{.*}})[%{{.*}}, %{{.*}}] {
      %x = constant 1 : i32
       // CHECK: %{{.*}} = constant 1 : i32
      %y = "add"(%x, %i) : (i32, index) -> i32 // CHECK: %{{.*}} = "add"(%{{.*}}, %{{.*}}) : (i32, index) -> i32
      %z = "mul"(%y, %y) : (i32, i32) -> i32 // CHECK: %{{.*}} = "mul"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
    }       // CHECK     }
  }         // CHECK   }
  return    // CHECK   return
}           // CHECK }

// CHECK-LABEL: func @attributes() {
func @attributes() {
  // CHECK: "foo"()
  "foo"(){} : ()->()

  // CHECK: "foo"() {a = 1 : i64, b = -423 : i64, c = [true, false], d = 1.600000e+01 : f64}  : () -> ()
  "foo"() {a = 1, b = -423, c = [true, false], d = 16.0 } : () -> ()

  // CHECK: "foo"() {map1 = #map{{[0-9]+}}}
  "foo"() {map1 = #map1} : () -> ()

  // CHECK: "foo"() {map2 = #map{{[0-9]+}}}
  "foo"() {map2 = (d0, d1, d2) -> (d0, d1, d2)} : () -> ()

  // CHECK: "foo"() {map12 = [#map{{[0-9]+}}, #map{{[0-9]+}}]}
  "foo"() {map12 = [#map1, #map2]} : () -> ()

  // CHECK: "foo"() {set1 = #set{{[0-9]+}}}
  "foo"() {set1 = #set1} : () -> ()

  // CHECK: "foo"() {set2 = #set{{[0-9]+}}}
  "foo"() {set2 = (d0, d1, d2) : (d0 >= 0, d1 >= 0, d2 - d1 == 0)} : () -> ()

  // CHECK: "foo"() {set12 = [#set{{[0-9]+}}, #set{{[0-9]+}}]}
  "foo"() {set12 = [#set1, #set2]} : () -> ()

  // CHECK: "foo"() {dictionary = {bool = true, fn = @ifinst}}
  "foo"() {dictionary = {bool = true, fn = @ifinst}} : () -> ()

  // Check that the dictionary attribute elements are sorted.
  // CHECK: "foo"() {dictionary = {bar = false, bool = true, fn = @ifinst}}
  "foo"() {dictionary = {fn = @ifinst, bar = false, bool = true}} : () -> ()

  // CHECK: "foo"() {d = 1.000000e-09 : f64, func = [], i123 = 7 : i64, if = "foo"} : () -> ()
  "foo"() {if = "foo", func = [], i123 = 7, d = 1.e-9} : () -> ()

  // CHECK: "foo"() {fn = @attributes, if = @ifinst} : () -> ()
  "foo"() {fn = @attributes, if = @ifinst} : () -> ()

  // CHECK: "foo"() {int = 0 : i42} : () -> ()
  "foo"() {int = 0 : i42} : () -> ()
  return
}

// CHECK-LABEL: func @ssa_values() -> (i16, i8) {
func @ssa_values() -> (i16, i8) {
  // CHECK: %{{.*}}:2 = "foo"() : () -> (i1, i17)
  %0:2 = "foo"() : () -> (i1, i17)
  br ^bb2

^bb1:       // CHECK: ^bb1: // pred: ^bb2
  // CHECK: %{{.*}}:2 = "baz"(%{{.*}}#1, %{{.*}}#0, %{{.*}}#1) : (f32, i11, i17) -> (i16, i8)
  %1:2 = "baz"(%2#1, %2#0, %0#1) : (f32, i11, i17) -> (i16, i8)

  // CHECK: return %{{.*}}#0, %{{.*}}#1 : i16, i8
  return %1#0, %1#1 : i16, i8

^bb2:       // CHECK: ^bb2:  // pred: ^bb0
  // CHECK: %{{.*}}:2 = "bar"(%{{.*}}#0, %{{.*}}#1) : (i1, i17) -> (i11, f32)
  %2:2 = "bar"(%0#0, %0#1) : (i1, i17) -> (i11, f32)
  br ^bb1
}

// CHECK-LABEL: func @bbargs() -> (i16, i8) {
func @bbargs() -> (i16, i8) {
  // CHECK: %{{.*}}:2 = "foo"() : () -> (i1, i17)
  %0:2 = "foo"() : () -> (i1, i17)
  br ^bb1(%0#1, %0#0 : i17, i1)

^bb1(%x: i17, %y: i1):       // CHECK: ^bb1(%{{.*}}: i17, %{{.*}}: i1):
  // CHECK: %{{.*}}:2 = "baz"(%{{.*}}, %{{.*}}, %{{.*}}#1) : (i17, i1, i17) -> (i16, i8)
  %1:2 = "baz"(%x, %y, %0#1) : (i17, i1, i17) -> (i16, i8)
  return %1#0, %1#1 : i16, i8
}

// CHECK-LABEL: func @verbose_terminators() -> (i1, i17)
func @verbose_terminators() -> (i1, i17) {
  %0:2 = "foo"() : () -> (i1, i17)
// CHECK:  br ^bb1(%{{.*}}#0, %{{.*}}#1 : i1, i17)
  "std.br"()[^bb1(%0#0, %0#1 : i1, i17)] : () -> ()

^bb1(%x : i1, %y : i17):
// CHECK:  cond_br %{{.*}}, ^bb2(%{{.*}} : i17), ^bb3(%{{.*}}, %{{.*}} : i1, i17)
  "std.cond_br"(%x)[^bb2(%y : i17), ^bb3(%x, %y : i1, i17)] : (i1) -> ()

^bb2(%a : i17):
  %true = constant 1 : i1
// CHECK:  return %{{.*}}, %{{.*}} : i1, i17
  "std.return"(%true, %a) : (i1, i17) -> ()

^bb3(%b : i1, %c : i17):
// CHECK:  return %{{.*}}, %{{.*}} : i1, i17
  "std.return"(%b, %c) : (i1, i17) -> ()
}

// CHECK-LABEL: func @condbr_simple
func @condbr_simple() -> (i32) {
  %cond = "foo"() : () -> i1
  %a = "bar"() : () -> i32
  %b = "bar"() : () -> i64
  // CHECK: cond_br %{{.*}}, ^bb1(%{{.*}} : i32), ^bb2(%{{.*}} : i64)
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
  // CHECK: cond_br %{{.*}}, ^bb1(%{{.*}}, %{{.*}} : i32, i64), ^bb2(%{{.*}}, %{{.*}}, %{{.*}} : i64, i32, i32)
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
  // CHECK: %{{.*}} = constant 42 : i32
  %x = constant 42 : i32
  // CHECK: %{{.*}} = constant 17 : i23
  %y = constant 17 : i23

  // This is a redundant definition of 17, the asmprinter gives it a unique name
  // CHECK: %{{.*}} = constant 17 : i23
  %z = constant 17 : i23

  // CHECK: %{{.*}} = constant 1 : i1
  %t = constant 1 : i1
  // CHECK: %{{.*}} = constant 0 : i1
  %f = constant 0 : i1

  // The trick to parse type declarations should not interfere with hex
  // literals.
  // CHECK: %{{.*}} = constant 3890 : i32
  %h = constant 0xf32 : i32

  // CHECK: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  return %x, %y, %z, %t, %f : i32, i23, i23, i1, i1
}

// CHECK-LABEL: func @typeattr
func @typeattr() -> () {
^bb0:
// CHECK: "foo"() {bar = tensor<*xf32>} : () -> ()
  "foo"(){bar = tensor<*xf32>} : () -> ()
  return
}

// CHECK-LABEL: func @stringquote
func @stringquote() -> () {
^bb0:
  // CHECK: "foo"() {bar = "a\22quoted\22string"} : () -> ()
  "foo"(){bar = "a\"quoted\"string"} : () -> ()

  // CHECK-NEXT: "typed_string" : !foo.string
  "foo"(){bar = "typed_string" : !foo.string} : () -> ()
  return
}

// CHECK-LABEL: func @unitAttrs
func @unitAttrs() -> () {
  // CHECK-NEXT: "foo"() {unitAttr} : () -> ()
  "foo"() {unitAttr = unit} : () -> ()

  // CHECK-NEXT: "foo"() {unitAttr} : () -> ()
  "foo"() {unitAttr} : () -> ()
  return
}

// CHECK-LABEL: func @floatAttrs
func @floatAttrs() -> () {
^bb0:
  // CHECK: "foo"() {a = 4.000000e+00 : f64, b = 2.000000e+00 : f64, c = 7.100000e+00 : f64, d = -0.000000e+00 : f64} : () -> ()
  "foo"(){a = 4.0, b = 2.0, c = 7.1, d = -0.0} : () -> ()
  return
}

// CHECK-LABEL: func @externalfuncattr
func @externalfuncattr() -> ()
  // CHECK: attributes {dialect.a = "a\22quoted\22string", dialect.b = 4.000000e+00 : f64, dialect.c = tensor<*xf32>}
  attributes {dialect.a = "a\"quoted\"string", dialect.b = 4.0, dialect.c = tensor<*xf32>}

// CHECK-LABEL: func @funcattrempty
func @funcattrempty() -> ()
  attributes {}

// CHECK-LABEL: func @funcattr
func @funcattr() -> ()
  // CHECK: attributes {dialect.a = "a\22quoted\22string", dialect.b = 4.000000e+00 : f64, dialect.c = tensor<*xf32>}
  attributes {dialect.a = "a\"quoted\"string", dialect.b = 4.0, dialect.c = tensor<*xf32>} {
^bb0:
  return
}

// CHECK-LABEL: func @funcattrwithblock
func @funcattrwithblock() -> ()
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
  affine.for %i0 = 0 to #map_simple0()[] {
  // CHECK: affine.for %{{.*}} = 0 to 10 {
    affine.for %i1 = 0 to #map_simple1()[%arg1] {
    // CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
      affine.for %i2 = 0 to #map_non_simple0(%i0)[] {
      // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}(%{{.*}}) {
        affine.for %i3 = 0 to #map_non_simple1(%i0)[%arg1] {
        // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}(%{{.*}})[%{{.*}}] {
          affine.for %i4 = 0 to #map_non_simple2()[%arg1, %arg0] {
          // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}()[%{{.*}}, %{{.*}}] {
            affine.for %i5 = 0 to #map_non_simple3()[%arg0] {
            // CHECK: affine.for %{{.*}} = 0 to #map{{[a-z_0-9]*}}()[%{{.*}}] {
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
  // CHECK: "splatBoolTensor"() {bar = dense<0> : tensor<i1>} : () -> ()
  "splatBoolTensor"(){bar = dense<false> : tensor<i1>} : () -> ()

  // CHECK: "splatIntTensor"() {bar = dense<5> : tensor<2x1x4xi32>} : () -> ()
  "splatIntTensor"(){bar = dense<5> : tensor<2x1x4xi32>} : () -> ()

  // CHECK: "splatFloatTensor"() {bar = dense<-5.000000e+00> : tensor<2x1x4xf32>} : () -> ()
  "splatFloatTensor"(){bar = dense<-5.0> : tensor<2x1x4xf32>} : () -> ()

  // CHECK: "splatIntVector"() {bar = dense<5> : vector<2x1x4xi64>} : () -> ()
  "splatIntVector"(){bar = dense<5> : vector<2x1x4xi64>} : () -> ()

  // CHECK: "splatFloatVector"() {bar = dense<-5.000000e+00> : vector<2x1x4xf16>} : () -> ()
  "splatFloatVector"(){bar = dense<-5.0> : vector<2x1x4xf16>} : () -> ()

  // CHECK: "splatIntScalar"() {bar = dense<5> : tensor<i9>} : () -> ()
  "splatIntScalar"() {bar = dense<5> : tensor<i9>} : () -> ()
  // CHECK: "splatFloatScalar"() {bar = dense<-5.000000e+00> : tensor<f16>} : () -> ()
  "splatFloatScalar"() {bar = dense<-5.0> : tensor<f16>} : () -> ()
  return
}

// CHECK-LABEL: func @densetensorattr
func @densetensorattr() -> () {
^bb0:

// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi3"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi3>} : () -> ()
  "fooi3"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi3>} : () -> ()
// CHECK: "fooi6"() {bar = dense<{{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]> : tensor<2x1x4xi6>} : () -> ()
  "fooi6"(){bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : tensor<2x1x4xi6>} : () -> ()
// CHECK: "fooi8"() {bar = dense<5> : tensor<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = dense<[[[5]]]> : tensor<1x1x1xi8>} : () -> ()
// CHECK: "fooi13"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi13>} : () -> ()
  "fooi13"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi13>} : () -> ()
// CHECK: "fooi16"() {bar = dense<-5> : tensor<1x1x1xi16>} : () -> ()
  "fooi16"(){bar = dense<[[[-5]]]> : tensor<1x1x1xi16>} : () -> ()
// CHECK: "fooi23"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi23>} : () -> ()
  "fooi23"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi23>} : () -> ()
// CHECK: "fooi32"() {bar = dense<5> : tensor<1x1x1xi32>} : () -> ()
  "fooi32"(){bar = dense<[[[5]]]> : tensor<1x1x1xi32>} : () -> ()
// CHECK: "fooi33"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi33>} : () -> ()
  "fooi33"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi33>} : () -> ()
// CHECK: "fooi43"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi43>} : () -> ()
  "fooi43"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi43>} : () -> ()
// CHECK: "fooi53"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 2, -1, 2]]]> : tensor<2x1x4xi53>} : () -> ()
  "fooi53"(){bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2x1x4xi53>} : () -> ()
// CHECK: "fooi64"() {bar = dense<{{\[\[\[}}1, -2, 1, 2]], {{\[\[}}0, 3, -1, 2]]]> : tensor<2x1x4xi64>} : () -> ()
  "fooi64"(){bar = dense<[[[1, -2, 1, 2]], [[0, 3, -1, 2]]]> : tensor<2x1x4xi64>} : () -> ()
// CHECK: "fooi64"() {bar = dense<-5> : tensor<1x1x1xi64>} : () -> ()
  "fooi64"(){bar = dense<[[[-5]]]> : tensor<1x1x1xi64>} : () -> ()
// CHECK: "fooi67"() {bar = dense<{{\[\[\[}}-5, 4, 6, 2]]]> : vector<1x1x4xi67>} : () -> ()
  "fooi67"(){bar = dense<[[[-5, 4, 6, 2]]]> : vector<1x1x4xi67>} : () -> ()

// CHECK: "foo2"() {bar = dense<[]> : tensor<0xi32>} : () -> ()
  "foo2"(){bar = dense<[]> : tensor<0xi32>} : () -> ()
// CHECK: "foo2"() {bar = dense<{{\[\[}}]]> : tensor<1x0xi32>} : () -> ()
  "foo2"(){bar = dense<[[]]> : tensor<1x0xi32>} : () -> ()
// CHECK: "foo3"() {bar = dense<{{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]> : tensor<2x1x4xi32>} : () -> ()
  "foo3"(){bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : tensor<2x1x4xi32>} : () -> ()

// CHECK: "float1"() {bar = dense<5.000000e+00> : tensor<1x1x1xf32>} : () -> ()
  "float1"(){bar = dense<[[[5.0]]]> : tensor<1x1x1xf32>} : () -> ()
// CHECK: "float2"() {bar = dense<[]> : tensor<0xf32>} : () -> ()
  "float2"(){bar = dense<[]> : tensor<0xf32>} : () -> ()
// CHECK: "float2"() {bar = dense<{{\[\[}}]]> : tensor<1x0xf32>} : () -> ()
  "float2"(){bar = dense<[[]]> : tensor<1x0xf32>} : () -> ()

// CHECK: "bfloat16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xbf16>} : () -> ()
  "bfloat16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xbf16>} : () -> ()
// CHECK: "float16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xf16>} : () -> ()
  "float16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xf16>} : () -> ()
// CHECK: "float32"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xf32>} : () -> ()
  "float32"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xf32>} : () -> ()
// CHECK: "float64"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : tensor<2x1x4xf64>} : () -> ()
  "float64"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : tensor<2x1x4xf64>} : () -> ()

// CHECK: "intscalar"() {bar = dense<1> : tensor<i32>} : () -> ()
  "intscalar"(){bar = dense<1> : tensor<i32>} : () -> ()
// CHECK: "floatscalar"() {bar = dense<5.000000e+00> : tensor<f32>} : () -> ()
  "floatscalar"(){bar = dense<5.0> : tensor<f32>} : () -> ()
  return
}

// CHECK-LABEL: func @densevectorattr
func @densevectorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar = dense<5> : vector<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = dense<[[[5]]]> : vector<1x1x1xi8>} : () -> ()
// CHECK: "fooi16"() {bar = dense<-5> : vector<1x1x1xi16>} : () -> ()
  "fooi16"(){bar = dense<[[[-5]]]> : vector<1x1x1xi16>} : () -> ()
// CHECK: "foo32"() {bar = dense<5> : vector<1x1x1xi32>} : () -> ()
  "foo32"(){bar = dense<[[[5]]]> : vector<1x1x1xi32>} : () -> ()
// CHECK: "fooi64"() {bar = dense<-5> : vector<1x1x1xi64>} : () -> ()
  "fooi64"(){bar = dense<[[[-5]]]> : vector<1x1x1xi64>} : () -> ()

// CHECK: "foo3"() {bar = dense<{{\[\[\[}}5, -6, 1, 2]], {{\[\[}}7, 8, 3, 4]]]> : vector<2x1x4xi32>} : () -> ()
  "foo3"(){bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : vector<2x1x4xi32>} : () -> ()

// CHECK: "float1"() {bar = dense<5.000000e+00> : vector<1x1x1xf32>} : () -> ()
  "float1"(){bar = dense<[[[5.0]]]> : vector<1x1x1xf32>} : () -> ()

// CHECK: "bfloat16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xbf16>} : () -> ()
  "bfloat16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xbf16>} : () -> ()
// CHECK: "float16"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xf16>} : () -> ()
  "float16"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xf16>} : () -> ()
// CHECK: "float32"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xf32>} : () -> ()
  "float32"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xf32>} : () -> ()
// CHECK: "float64"() {bar = dense<{{\[\[\[}}-5.000000e+00, 6.000000e+00, 1.000000e+00, 2.000000e+00]], {{\[\[}}7.000000e+00, -8.000000e+00, 3.000000e+00, 4.000000e+00]]]> : vector<2x1x4xf64>} : () -> ()
  "float64"(){bar = dense<[[[-5.0, 6.0, 1.0, 2.0]], [[7.0, -8.0, 3.0, 4.0]]]> : vector<2x1x4xf64>} : () -> ()
  return
}

// CHECK-LABEL: func @sparsetensorattr
func @sparsetensorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar = sparse<0, -2> : tensor<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = sparse<0, -2> : tensor<1x1x1xi8>} : () -> ()
// CHECK: "fooi16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2, -1, 5]> : tensor<2x2x2xi16>} : () -> ()
  "fooi16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]> : tensor<2x2x2xi16>} : () -> ()
// CHECK: "fooi32"() {bar = sparse<{{\[}}], {{\[}}]> : tensor<1x1xi32>} : () -> ()
  "fooi32"(){bar = sparse<[], []> : tensor<1x1xi32>} : () -> ()
// CHECK: "fooi64"() {bar = sparse<0, -1> : tensor<1xi64>} : () -> ()
  "fooi64"(){bar = sparse<[[0]], [-1]> : tensor<1xi64>} : () -> ()
// CHECK: "foo2"() {bar = sparse<{{\[}}], {{\[}}]> : tensor<0xi32>} : () -> ()
  "foo2"(){bar = sparse<[], []> : tensor<0xi32>} : () -> ()
// CHECK: "foo3"() {bar = sparse<{{\[}}], {{\[}}]> : tensor<i32>} : () -> ()
  "foo3"(){bar = sparse<[], []> : tensor<i32>} : () -> ()

// CHECK: "foof16"() {bar = sparse<0, -2.000000e+00> : tensor<1x1x1xf16>} : () -> ()
  "foof16"(){bar = sparse<0, -2.0> : tensor<1x1x1xf16>} : () -> ()
// CHECK: "foobf16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2.000000e+00, -1.000000e+00, 5.000000e+00]> : tensor<2x2x2xbf16>} : () -> ()
  "foobf16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.0, -1.0, 5.0]> : tensor<2x2x2xbf16>} : () -> ()
// CHECK: "foof32"() {bar = sparse<{{\[}}], {{\[}}]> : tensor<1x0x1xf32>} : () -> ()
  "foof32"(){bar = sparse<[], []> : tensor<1x0x1xf32>} : () -> ()
// CHECK:  "foof64"() {bar = sparse<0, -1.000000e+00> : tensor<1xf64>} : () -> ()
  "foof64"(){bar = sparse<[[0]], [-1.0]> : tensor<1xf64>} : () -> ()
// CHECK: "foof320"() {bar = sparse<{{\[}}], {{\[}}]> : tensor<0xf32>} : () -> ()
  "foof320"(){bar = sparse<[], []> : tensor<0xf32>} : () -> ()
// CHECK: "foof321"() {bar = sparse<{{\[}}], {{\[}}]> : tensor<f32>} : () -> ()
  "foof321"(){bar = sparse<[], []> : tensor<f32>} : () -> ()
  return
}

// CHECK-LABEL: func @sparsevectorattr
func @sparsevectorattr() -> () {
^bb0:
// NOTE: The {{\[\[}} syntax is because "[[" confuses FileCheck.
// CHECK: "fooi8"() {bar = sparse<0, -2> : vector<1x1x1xi8>} : () -> ()
  "fooi8"(){bar = sparse<0, -2> : vector<1x1x1xi8>} : () -> ()
// CHECK: "fooi16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2, -1, 5]> : vector<2x2x2xi16>} : () -> ()
  "fooi16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]> : vector<2x2x2xi16>} : () -> ()
// CHECK: "fooi32"() {bar = sparse<{{\[}}], {{\[}}]> : vector<1x1xi32>} : () -> ()
  "fooi32"(){bar = sparse<[], []> : vector<1x1xi32>} : () -> ()
// CHECK: "fooi64"() {bar = sparse<0, -1> : vector<1xi64>} : () -> ()
  "fooi64"(){bar = sparse<[[0]], [-1]> : vector<1xi64>} : () -> ()

// CHECK: "foof16"() {bar = sparse<0, -2.000000e+00> : vector<1x1x1xf16>} : () -> ()
  "foof16"(){bar = sparse<0, -2.0> : vector<1x1x1xf16>} : () -> ()
// CHECK: "foobf16"() {bar = sparse<{{\[\[}}1, 1, 0], {{\[}}0, 1, 0], {{\[}}0, 0, 1]], {{\[}}2.000000e+00, -1.000000e+00, 5.000000e+00]> : vector<2x2x2xbf16>} : () -> ()
  "foobf16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.0, -1.0, 5.0]> : vector<2x2x2xbf16>} : () -> ()
// CHECK:  "foof64"() {bar = sparse<0, -1.000000e+00> : vector<1xf64>} : () -> ()
  "foof64"(){bar = sparse<0, [-1.0]> : vector<1xf64>} : () -> ()
  return
}

// CHECK-LABEL: func @unknown_dialect_type() -> !bar<""> {
func @unknown_dialect_type() -> !bar<""> {
  // Unregistered dialect 'bar'.
  // CHECK: "foo"() : () -> !bar<"">
  %0 = "foo"() : () -> !bar<"">

  // CHECK: "foo"() : () -> !bar.baz
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
  // CHECK: affine.if [[SET_TRUE]]() {
  affine.if () : () () {
  }
  return
}

// CHECK-LABEL: func @verbose_if(
func @verbose_if(%N: index) {
  %c = constant 200 : index

  // CHECK: affine.if #set{{.*}}(%{{.*}})[%{{.*}}, %{{.*}}] {
  "affine.if"(%c, %N, %c) ({
    // CHECK-NEXT: "add"
    %y = "add"(%c, %N) : (index, index) -> index
    "affine.terminator"() : () -> ()
    // CHECK-NEXT: } else {
  }, { // The else region.
    // CHECK-NEXT: "add"
    %z = "add"(%c, %c) : (index, index) -> index
    "affine.terminator"() : () -> ()
  })
  { condition = #set0 } : (index, index, index) -> ()
  return
}

// CHECK-LABEL: func @terminator_with_regions
func @terminator_with_regions() {
  // Combine successors and regions in the same operation.
  // CHECK: "region"()[^bb1] ( {
  // CHECK: }) : () -> ()
  "region"()[^bb2] ({}) : () -> ()
^bb2:
  return
}

// CHECK-LABEL: func @unregistered_term
func @unregistered_term(%arg0 : i1) -> i1 {
  // CHECK-NEXT: "unregistered_br"()[^bb1(%{{.*}} : i1)] : () -> ()
  "unregistered_br"()[^bb1(%arg0 : i1)] : () -> ()

^bb1(%arg1 : i1):
  return %arg1 : i1
}

// CHECK-LABEL: func @dialect_attrs
func @dialect_attrs()
    // CHECK-NEXT: attributes  {dialect.attr = 10
    attributes {dialect.attr = 10} {
  return
}

// CHECK-LABEL: func @_valid.function$name
func @_valid.function$name()

// CHECK-LABEL: func @external_func_arg_attrs(i32, i1 {dialect.attr = 10 : i64}, i32)
func @external_func_arg_attrs(i32, i1 {dialect.attr = 10 : i64}, i32)

// CHECK-LABEL: func @func_arg_attrs(%{{.*}}: i1 {dialect.attr = 10 : i64})
func @func_arg_attrs(%arg0: i1 {dialect.attr = 10 : i64}) {
  return
}

// CHECK-LABEL: func @empty_tuple(tuple<>)
func @empty_tuple(tuple<>)

// CHECK-LABEL: func @tuple_single_element(tuple<i32>)
func @tuple_single_element(tuple<i32>)

// CHECK-LABEL: func @tuple_multi_element(tuple<i32, i16, f32>)
func @tuple_multi_element(tuple<i32, i16, f32>)

// CHECK-LABEL: func @tuple_nested(tuple<tuple<tuple<i32>>>)
func @tuple_nested(tuple<tuple<tuple<i32>>>)

// CHECK-LABEL: func @pretty_form_multi_result
func @pretty_form_multi_result() -> (i16, i16) {
  // CHECK: %{{.*}}:2 = "foo_div"() : () -> (i16, i16)
  %quot, %rem = "foo_div"() : () -> (i16, i16)
  return %quot, %rem : i16, i16
}

// CHECK-LABEL: func @pretty_dialect_attribute()
func @pretty_dialect_attribute() {

  // CHECK: "foo.unknown_op"() {foo = #foo.simpleattr} : () -> ()
  "foo.unknown_op"() {foo = #foo.simpleattr} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.complexattr<abcd>} : () -> ()
  "foo.unknown_op"() {foo = #foo.complexattr<abcd>} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.complexattr<abcd<f32>>} : () -> ()
  "foo.unknown_op"() {foo = #foo.complexattr<abcd<f32>>} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.complexattr<abcd<[f]$$[32]>>} : () -> ()
  "foo.unknown_op"() {foo = #foo.complexattr<abcd<[f]$$[32]>>} : () -> ()

  // CHECK: "foo.unknown_op"() {foo = #foo.dialect<!x@#!@#>} : () -> ()
  "foo.unknown_op"() {foo = #foo.dialect<!x@#!@#>} : () -> ()

  // Extraneous extra > character can't use the pretty syntax.
  // CHECK: "foo.unknown_op"() {foo = #foo<"dialect<!x@#!@#>>">} : () -> ()
  "foo.unknown_op"() {foo = #foo<"dialect<!x@#!@#>>">} : () -> ()

  return
}

// CHECK-LABEL: func @pretty_dialect_type()
func @pretty_dialect_type() {

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.simpletype
  %0 = "foo.unknown_op"() : () -> !foo.simpletype

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.complextype<abcd>
  %1 = "foo.unknown_op"() : () -> !foo.complextype<abcd>

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.complextype<abcd<f32>>
  %2 = "foo.unknown_op"() : () -> !foo.complextype<abcd<f32>>

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.complextype<abcd<[f]$$[32]>>
  %3 = "foo.unknown_op"() : () -> !foo.complextype<abcd<[f]$$[32]>>

  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo.dialect<!x@#!@#>
  %4 = "foo.unknown_op"() : () -> !foo.dialect<!x@#!@#>

  // Extraneous extra > character can't use the pretty syntax.
  // CHECK: %{{.*}} = "foo.unknown_op"() : () -> !foo<"dialect<!x@#!@#>>">
  %5 = "foo.unknown_op"() : () -> !foo<"dialect<!x@#!@#>>">

  return
}

// CHECK-LABEL: func @none_type
func @none_type() {
  // CHECK: "foo.unknown_op"() : () -> none
  %none_val = "foo.unknown_op"() : () -> none
  return
}

// CHECK-LABEL: func @scoped_names
func @scoped_names() {
  // CHECK-NEXT: "foo.region_op"
  "foo.region_op"() ({
    // CHECK-NEXT: "foo.unknown_op"
    %scoped_name = "foo.unknown_op"() : () -> none
    "foo.terminator"() : () -> ()
  }, {
    // CHECK: "foo.unknown_op"
    %scoped_name = "foo.unknown_op"() : () -> none
    "foo.terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @loc_attr(i1 {foo.loc_attr = loc(callsite("foo" at "mysource.cc":10:8))})
func @loc_attr(i1 {foo.loc_attr = loc(callsite("foo" at "mysource.cc":10:8))})

// CHECK-LABEL: func @dialect_attribute_with_type
func @dialect_attribute_with_type() {
  // CHECK-NEXT: foo = #foo.attr : i32
  "foo.unknown_op"() {foo = #foo.attr : i32} : () -> ()
}
