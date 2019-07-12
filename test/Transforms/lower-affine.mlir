// RUN: mlir-opt -lower-affine %s | FileCheck %s

// CHECK-LABEL: func @empty() {
func @empty() {
  return     // CHECK:  return
}            // CHECK: }

func @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: func @simple_loop
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1_0:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c1]] to %[[c42]] step %[[c1_0]] {
// CHECK-NEXT:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @simple_loop() {
  affine.for %i = 1 to 42 {
    call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @pre(index) -> ()
func @body2(index, index) -> ()
func @post(index) -> ()

// CHECK-LABEL: func @imperfectly_nested_loops
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:     %[[c56:.*]] = constant 56 : index
// CHECK-NEXT:     %[[c2:.*]] = constant 2 : index
// CHECK-NEXT:     for %{{.*}} = %[[c7]] to %[[c56]] step %[[c2]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
      call @body2(%i, %j) : (index, index) -> ()
    }
    call @post(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @mid(index) -> ()
func @body3(index, index) -> ()

// CHECK-LABEL: func @more_imperfectly_nested_loops
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:     %[[c56:.*]] = constant 56 : index
// CHECK-NEXT:     %[[c2:.*]] = constant 2 : index
// CHECK-NEXT:     for %{{.*}} = %[[c7]] to %[[c56]] step %[[c2]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @mid(%{{.*}}) : (index) -> ()
// CHECK-NEXT:     %[[c18:.*]] = constant 18 : index
// CHECK-NEXT:     %[[c37:.*]] = constant 37 : index
// CHECK-NEXT:     %[[c3:.*]] = constant 3 : index
// CHECK-NEXT:     for %{{.*}} = %[[c18]] to %[[c37]] step %[[c3]] {
// CHECK-NEXT:       call @body3(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @more_imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
      call @body2(%i, %j) : (index, index) -> ()
    }
    call @mid(%i) : (index) -> ()
    affine.for %k = 18 to 37 step 3 {
      call @body3(%i, %k) : (index, index) -> ()
    }
    call @post(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @affine_apply_loops_shorthand
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %{{.*}} step %[[c1]] {
// CHECK-NEXT:     %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:     %[[c1_0:.*]] = constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %{{.*}} to %[[c42]] step %[[c1_0]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @affine_apply_loops_shorthand(%N : index) {
  affine.for %i = 0 to %N {
    affine.for %j = (d0)[]->(d0)(%i)[] to 42 {
      call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @get_idx() -> (index)

#set1 = (d0) : (20 - d0 >= 0)
#set2 = (d0) : (d0 - 10 >= 0)

// CHECK-LABEL: func @if_only
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi "sge", %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_only() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    call @body(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_else
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi "sge", %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   } else {
// CHECK-NEXT:     call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_else() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @nested_ifs
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi "sge", %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     %[[c0_0:.*]] = constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]] = constant -10 : index
// CHECK-NEXT:     %[[v4:.*]] = addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:     %[[v5:.*]] = cmpi "sge", %[[v4]], %[[c0_0]] : index
// CHECK-NEXT:     if %[[v5]] {
// CHECK-NEXT:       call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[c0_0:.*]] = constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]] = constant -10 : index
// CHECK-NEXT:     %{{.*}} = addi %[[v0]], %[[cm10]] : index
// CHECK-NEXT:     %{{.*}} = cmpi "sge", %{{.*}}, %[[c0_0]] : index
// CHECK-NEXT:     if %{{.*}} {
// CHECK-NEXT:       call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @nested_ifs() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    affine.if #set2(%i) {
      call @body(%i) : (index) -> ()
    }
  } else {
    affine.if #set2(%i) {
      call @mid(%i) : (index) -> ()
    }
  }
  return
}

#setN = (d0)[N,M,K,L] : (N - d0 + 1 >= 0, N - 1 >= 0, M - 1 >= 0, K - 1 >= 0, L - 42 == 0)

// CHECK-LABEL: func @multi_cond
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %{{.*}} : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[v3:.*]] = addi %[[v2]], %[[c1]] : index
// CHECK-NEXT:   %[[v4:.*]] = cmpi "sge", %[[v3]], %[[c0]] : index
// CHECK-NEXT:   %[[cm1_0:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v5:.*]] = addi %{{.*}}, %[[cm1_0]] : index
// CHECK-NEXT:   %[[v6:.*]] = cmpi "sge", %[[v5]], %[[c0]] : index
// CHECK-NEXT:   %[[v7:.*]] = and %[[v4]], %[[v6]] : i1
// CHECK-NEXT:   %[[cm1_1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v8:.*]] = addi %{{.*}}, %[[cm1_1]] : index
// CHECK-NEXT:   %[[v9:.*]] = cmpi "sge", %[[v8]], %[[c0]] : index
// CHECK-NEXT:   %[[v10:.*]] = and %[[v7]], %[[v9]] : i1
// CHECK-NEXT:   %[[cm1_2:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v11:.*]] = addi %{{.*}}, %[[cm1_2]] : index
// CHECK-NEXT:   %[[v12:.*]] = cmpi "sge", %[[v11]], %[[c0]] : index
// CHECK-NEXT:   %[[v13:.*]] = and %[[v10]], %[[v12]] : i1
// CHECK-NEXT:   %[[cm42:.*]] = constant -42 : index
// CHECK-NEXT:   %[[v14:.*]] = addi %{{.*}}, %[[cm42]] : index
// CHECK-NEXT:   %[[v15:.*]] = cmpi "eq", %[[v14]], %[[c0]] : index
// CHECK-NEXT:   %[[v16:.*]] = and %[[v13]], %[[v15]] : i1
// CHECK-NEXT:   if %[[v16]] {
// CHECK-NEXT:     call @body(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   } else {
// CHECK-NEXT:     call @mid(%[[v0:.*]]) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @multi_cond(%N : index, %M : index, %K : index, %L : index) {
  %i = call @get_idx() : () -> (index)
  affine.if #setN(%i)[%N,%M,%K,%L] {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_for
func @if_for() {
// CHECK-NEXT:   %[[v0:.*]] = call @get_idx() : () -> index
  %i = call @get_idx() : () -> (index)
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:   %[[v1:.*]] = muli %[[v0]], %[[cm1]] : index
// CHECK-NEXT:   %[[c20:.*]] = constant 20 : index
// CHECK-NEXT:   %[[v2:.*]] = addi %[[v1]], %[[c20]] : index
// CHECK-NEXT:   %[[v3:.*]] = cmpi "sge", %[[v2]], %[[c0]] : index
// CHECK-NEXT:   if %[[v3]] {
// CHECK-NEXT:     %[[c0:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:     %[[c42:.*]]{{.*}} = constant 42 : index
// CHECK-NEXT:     %[[c1:.*]]{{.*}} = constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %[[c0:.*]]{{.*}} to %[[c42:.*]]{{.*}} step %[[c1:.*]]{{.*}} {
// CHECK-NEXT:       %[[c0_:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:       %[[cm10:.*]] = constant -10 : index
// CHECK-NEXT:       %[[v4:.*]] = addi %{{.*}}, %[[cm10]] : index
// CHECK-NEXT:       %[[v5:.*]] = cmpi "sge", %[[v4]], %[[c0_:.*]]{{.*}} : index
// CHECK-NEXT:       if %[[v5]] {
// CHECK-NEXT:         call @body2(%[[v0]], %{{.*}}) : (index, index) -> ()
  affine.if #set1(%i) {
    affine.for %j = 0 to 42 {
      affine.if #set2(%j) {
        call @body2(%i, %j) : (index, index) -> ()
      }
    }
  }
//      CHECK:   %[[c0:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]]{{.*}} = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]]{{.*}} = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0:.*]]{{.*}} to %[[c42:.*]]{{.*}} step %[[c1:.*]]{{.*}} {
// CHECK-NEXT:     %[[c0:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:     %[[cm10:.*]]{{.*}} = constant -10 : index
// CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %[[cm10:.*]]{{.*}} : index
// CHECK-NEXT:     %{{.*}} = cmpi "sge", %{{.*}}, %[[c0:.*]]{{.*}} : index
// CHECK-NEXT:     if %{{.*}} {
// CHECK-NEXT:       %[[c0_:.*]]{{.*}} = constant 0 : index
// CHECK-NEXT:       %[[c42_:.*]]{{.*}} = constant 42 : index
// CHECK-NEXT:       %[[c1_:.*]]{{.*}} = constant 1 : index
// CHECK-NEXT:       for %{{.*}} = %[[c0_:.*]]{{.*}} to %[[c42_:.*]]{{.*}} step %[[c1_:.*]]{{.*}} {
  affine.for %k = 0 to 42 {
    affine.if #set2(%k) {
      affine.for %l = 0 to 42 {
        call @body3(%k, %l) : (index, index) -> ()
      }
    }
  }
//      CHECK:   return
  return
}

#lbMultiMap = (d0)[s0] -> (d0, s0 - d0)
#ubMultiMap = (d0)[s0] -> (s0, d0 + 10)

// CHECK-LABEL: func @loop_min_max
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[c42]] step %[[c1]] {
// CHECK-NEXT:     %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT:     %[[a:.*]] = muli %{{.*}}, %[[cm1]] : index
// CHECK-NEXT:     %[[b:.*]] = addi %[[a]], %{{.*}} : index
// CHECK-NEXT:     %[[c:.*]] = cmpi "sgt", %{{.*}}, %[[b]] : index
// CHECK-NEXT:     %[[d:.*]] = select %[[c]], %{{.*}}, %[[b]] : index
// CHECK-NEXT:     %[[c10:.*]] = constant 10 : index
// CHECK-NEXT:     %[[e:.*]] = addi %{{.*}}, %[[c10]] : index
// CHECK-NEXT:     %[[f:.*]] = cmpi "slt", %{{.*}}, %[[e]] : index
// CHECK-NEXT:     %[[g:.*]] = select %[[f]], %{{.*}}, %[[e]] : index
// CHECK-NEXT:     %[[c1_0:.*]] = constant 1 : index
// CHECK-NEXT:     for %{{.*}} = %[[v3]] to %[[v6]] step %[[c1_0]] {
// CHECK-NEXT:       call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @loop_min_max(%N : index) {
  affine.for %i = 0 to 42 {
    affine.for %j = max #lbMultiMap(%i)[%N] to min #ubMultiMap(%i)[%N] {
      call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

#map_7_values = (i) -> (i, i, i, i, i, i, i)

// Check that the "min" (cmpi "slt" + select) reduction sequence is emitted
// correctly for a an affine map with 7 results.

// CHECK-LABEL: func @min_reduction_tree
// CHECK-NEXT:   %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[c01:.+]] = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[r01:.+]] = select %[[c01]], %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[c012:.+]] = cmpi "slt", %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[r012:.+]] = select %[[c012]], %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123:.+]] = cmpi "slt", %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123:.+]] = select %[[c0123]], %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[c01234:.+]] = cmpi "slt", %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[r01234:.+]] = select %[[c01234]], %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[c012345:.+]] = cmpi "slt", %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[r012345:.+]] = select %[[c012345]], %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123456:.+]] = cmpi "slt", %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123456:.+]] = select %[[c0123456]], %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:   for %{{.*}} = %[[c0]] to %[[v11]] step %[[c1]] {
// CHECK-NEXT:     call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @min_reduction_tree(%v : index) {
  affine.for %i = 0 to min #map_7_values(%v)[] {
    call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

#map0 = () -> (0)
#map1 = ()[s0] -> (s0)
#map2 = (d0) -> (d0)
#map3 = (d0)[s0] -> (d0 + s0 + 1)
#map4 = (d0,d1,d2,d3)[s0,s1,s2] -> (d0 + 2*d1 + 3*d2 + 4*d3 + 5*s0 + 6*s1 + 7*s2)
#map5 = (d0,d1,d2) -> (d0,d1,d2)
#map6 = (d0,d1,d2) -> (d0 + d1 + d2)

// CHECK-LABEL: func @affine_applies(
func @affine_applies() {
^bb0:
// CHECK: %[[c0:.*]] = constant 0 : index
  %zero = affine.apply #map0()

// Identity maps are just discarded.
// CHECK-NEXT: %[[c101:.*]] = constant 101 : index
  %101 = constant 101 : index
  %symbZero = affine.apply #map1()[%zero]
// CHECK-NEXT: %[[c102:.*]] = constant 102 : index
  %102 = constant 102 : index
  %copy = affine.apply #map2(%zero)

// CHECK-NEXT: %[[v0:.*]] = addi %[[c0]], %[[c0]] : index
// CHECK-NEXT: %[[c1:.*]] = constant 1 : index
// CHECK-NEXT: %[[v1:.*]] = addi %[[v0]], %[[c1]] : index
  %one = affine.apply #map3(%symbZero)[%zero]

// CHECK-NEXT: %[[c103:.*]] = constant 103 : index
// CHECK-NEXT: %[[c104:.*]] = constant 104 : index
// CHECK-NEXT: %[[c105:.*]] = constant 105 : index
// CHECK-NEXT: %[[c106:.*]] = constant 106 : index
// CHECK-NEXT: %[[c107:.*]] = constant 107 : index
// CHECK-NEXT: %[[c108:.*]] = constant 108 : index
// CHECK-NEXT: %[[c109:.*]] = constant 109 : index
  %103 = constant 103 : index
  %104 = constant 104 : index
  %105 = constant 105 : index
  %106 = constant 106 : index
  %107 = constant 107 : index
  %108 = constant 108 : index
  %109 = constant 109 : index
// CHECK-NEXT: %[[c2:.*]] = constant 2 : index
// CHECK-NEXT: %[[v2:.*]] = muli %[[c104]], %[[c2]] : index
// CHECK-NEXT: %[[v3:.*]] = addi %[[c103]], %[[v2]] : index
// CHECK-NEXT: %[[c3:.*]] = constant 3 : index
// CHECK-NEXT: %[[v4:.*]] = muli %[[c105]], %[[c3]] : index
// CHECK-NEXT: %[[v5:.*]] = addi %[[v3]], %[[v4]] : index
// CHECK-NEXT: %[[c4:.*]] = constant 4 : index
// CHECK-NEXT: %[[v6:.*]] = muli %[[c106]], %[[c4]] : index
// CHECK-NEXT: %[[v7:.*]] = addi %[[v5]], %[[v6]] : index
// CHECK-NEXT: %[[c5:.*]] = constant 5 : index
// CHECK-NEXT: %[[v8:.*]] = muli %[[c107]], %[[c5]] : index
// CHECK-NEXT: %[[v9:.*]] = addi %[[v7]], %[[v8]] : index
// CHECK-NEXT: %[[c6:.*]] = constant 6 : index
// CHECK-NEXT: %[[v10:.*]] = muli %[[c108]], %[[c6]] : index
// CHECK-NEXT: %[[v11:.*]] = addi %[[v9]], %[[v10]] : index
// CHECK-NEXT: %[[c7:.*]] = constant 7 : index
// CHECK-NEXT: %[[v12:.*]] = muli %[[c109]], %[[c7]] : index
// CHECK-NEXT: %[[v13:.*]] = addi %[[v11]], %[[v12]] : index
  %four = affine.apply #map4(%103,%104,%105,%106)[%107,%108,%109]
  return
}

// CHECK-LABEL: func @args_ret_affine_apply(
func @args_ret_affine_apply(index, index) -> (index, index) {
^bb0(%0 : index, %1 : index):
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, index
  %00 = affine.apply #map2 (%0)
  %11 = affine.apply #map1 ()[%1]
  return %00, %11 : index, index
}

//===---------------------------------------------------------------------===//
// Test lowering of Euclidean (floor) division, ceil division and modulo
// operation used in affine expressions.  In addition to testing the
// operation-level output, check that the obtained results are correct by
// applying constant folding transformation after affine lowering.
//===---------------------------------------------------------------------===//

#mapmod = (i) -> (i mod 42)

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_mod
func @affine_apply_mod(%arg0 : index) -> (index) {
// CHECK-NEXT: %[[c42:.*]] = constant 42 : index
// CHECK-NEXT: %[[v0:.*]] = remis %{{.*}}, %[[c42]] : index
// CHECK-NEXT: %[[c0:.*]] = constant 0 : index
// CHECK-NEXT: %[[v1:.*]] = cmpi "slt", %[[v0]], %[[c0]] : index
// CHECK-NEXT: %[[v2:.*]] = addi %[[v0]], %[[c42]] : index
// CHECK-NEXT: %[[v3:.*]] = select %[[v1]], %[[v2]], %[[v0]] : index
  %0 = affine.apply #mapmod (%arg0)
  return %0 : index
}

#mapfloordiv = (i) -> (i floordiv 42)

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_floordiv
func @affine_apply_floordiv(%arg0 : index) -> (index) {
// CHECK-NEXT: %[[c42:.*]] = constant 42 : index
// CHECK-NEXT: %[[c0:.*]] = constant 0 : index
// CHECK-NEXT: %[[cm1:.*]] = constant -1 : index
// CHECK-NEXT: %[[v0:.*]] = cmpi "slt", %{{.*}}, %[[c0]] : index
// CHECK-NEXT: %[[v1:.*]] = subi %[[cm1]], %{{.*}} : index
// CHECK-NEXT: %[[v2:.*]] = select %[[v0]], %[[v1]], %{{.*}} : index
// CHECK-NEXT: %[[v3:.*]] = divis %[[v2]], %[[c42]] : index
// CHECK-NEXT: %[[v4:.*]] = subi %[[cm1]], %[[v3]] : index
// CHECK-NEXT: %[[v5:.*]] = select %[[v0]], %[[v4]], %[[v3]] : index
  %0 = affine.apply #mapfloordiv (%arg0)
  return %0 : index
}

#mapceildiv = (i) -> (i ceildiv 42)

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_ceildiv
func @affine_apply_ceildiv(%arg0 : index) -> (index) {
// CHECK-NEXT:  %[[c42:.*]] = constant 42 : index
// CHECK-NEXT:  %[[c0:.*]] = constant 0 : index
// CHECK-NEXT:  %[[c1:.*]] = constant 1 : index
// CHECK-NEXT:  %[[v0:.*]] = cmpi "sle", %{{.*}}, %[[c0]] : index
// CHECK-NEXT:  %[[v1:.*]] = subi %[[c0]], %{{.*}} : index
// CHECK-NEXT:  %[[v2:.*]] = subi %{{.*}}, %[[c1]] : index
// CHECK-NEXT:  %[[v3:.*]] = select %[[v0]], %[[v1]], %[[v2]] : index
// CHECK-NEXT:  %[[v4:.*]] = divis %[[v3]], %[[c42]] : index
// CHECK-NEXT:  %[[v5:.*]] = subi %[[c0]], %[[v4]] : index
// CHECK-NEXT:  %[[v6:.*]] = addi %[[v4]], %[[c1]] : index
// CHECK-NEXT:  %[[v7:.*]] = select %[[v0]], %[[v5]], %[[v6]] : index
  %0 = affine.apply #mapceildiv (%arg0)
  return %0 : index
}

// CHECK-LABEL: func @affine_load
func @affine_load(%arg0 : index) {
  %0 = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %1 = affine.load %0[%i0 + symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %[[a:.*]] = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %[[c7:.*]] = constant 7 : index
// CHECK-NEXT:  %[[b:.*]] = addi %[[a]], %[[c7]] : index
// CHECK-NEXT:  %{{.*}} = load %[[v0:.*]][%[[b]]] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_store
func @affine_store(%arg0 : index) {
  %0 = alloc() : memref<10xf32>
  %1 = constant 11.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %1, %0[%i0 - symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %c-1 = constant -1 : index
// CHECK-NEXT:  %[[a:.*]] = muli %arg0, %c-1 : index
// CHECK-NEXT:  %[[b:.*]] = addi %{{.*}}, %[[a]] : index
// CHECK-NEXT:  %c7 = constant 7 : index
// CHECK-NEXT:  %[[c:.*]] = addi %[[b]], %c7 : index
// CHECK-NEXT:  store %cst, %0[%[[c]]] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_dma_start
func @affine_dma_start(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = alloc() : memref<100xf32, 2>
  %2 = alloc() : memref<1xi32>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_start %0[%i0 + 7], %1[%arg0 + 11], %2[%c0], %c64
        : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  }
// CHECK:       %c7 = constant 7 : index
// CHECK-NEXT:  %[[a:.*]] = addi %{{.*}}, %c7 : index
// CHECK-NEXT:  %c11 = constant 11 : index
// CHECK-NEXT:  %[[b:.*]] = addi %arg0, %c11 : index
// CHECK-NEXT:  dma_start %0[%[[a]]], %1[%[[b]]], %c64, %2[%c0] : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  return
}

// CHECK-LABEL: func @affine_dma_wait
func @affine_dma_wait(%arg0 : index) {
  %2 = alloc() : memref<1xi32>
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_wait %2[%i0 + %arg0 + 17], %c64 : memref<1xi32>
  }
// CHECK:       %[[a:.*]] = addi %{{.*}}, %arg0 : index
// CHECK-NEXT:  %c17 = constant 17 : index
// CHECK-NEXT:  %[[b:.*]] = addi %[[a]], %c17 : index
// CHECK-NEXT:  dma_wait %0[%[[b]]], %c64 : memref<1xi32>
  return
}
