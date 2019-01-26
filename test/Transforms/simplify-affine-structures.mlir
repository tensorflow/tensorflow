// RUN: mlir-opt %s -simplify-affine-structures | FileCheck %s

// CHECK-DAG: [[SET_EMPTY_2D:#set[0-9]+]] = (d0, d1) : (1 == 0)
// CHECK-DAG: #set1 = (d0, d1) : (d0 - 100 == 0, d1 - 10 == 0, -d0 + 100 >= 0, d1 >= 0, d1 + 101 >= 0)
// CHECK-DAG: #set2 = (d0, d1)[s0, s1] : (1 == 0)
// CHECK-DAG: #set3 = (d0, d1)[s0, s1] : (d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0, d0 * 5 - d1 * 11 + s0 * 7 + s1 == 0, d0 * 11 + d1 * 7 - s0 * 5 + s1 == 0, d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0)
// CHECK-DAG: [[SET_EMPTY_1D:#set[0-9]+]] = (d0) : (1 == 0)
// CHECK-DAG: [[SET_EMPTY_1D_2S:#set[0-9]+]] = (d0)[s0, s1] : (1 == 0)
// CHECK-DAG: [[SET_EMPTY_3D:#set[0-9]+]] = (d0, d1, d2) : (1 == 0)

// Set for test case: test_gaussian_elimination_non_empty_set2
// #set2 = (d0, d1) : (d0 - 100 == 0, d1 - 10 == 0, -d0 + 100 >= 0, d1 >= 0, d1 + 101 >= 0)
#set2 = (d0, d1) : (d0 - 100 == 0, d1 - 10 == 0, -d0 + 100 >= 0, d1 >= 0, d1 + 101 >= 0)

// Set for test case: test_gaussian_elimination_empty_set3
// #set3 = (d0, d1)[s0, s1] : (1 == 0)
#set3 = (d0, d1)[s0, s1] : (d0 - s0 == 0, d0 + s0 == 0, s0 - 1 == 0)

// Set for test case: test_gaussian_elimination_non_empty_set4
#set4 = (d0, d1)[s0, s1] : (d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                            d0 * 5 - d1 * 11 + s0 * 7 + s1 == 0,
                            d0 * 11 + d1 * 7 - s0 * 5 + s1 == 0,
                            d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0)

// Add invalid constraints to previous non-empty set to make it empty.
// Set for test case: test_gaussian_elimination_empty_set5
#set5 = (d0, d1)[s0, s1] : (d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                             d0 * 5 - d1 * 11 + s0 * 7 + s1 == 0,
                             d0 * 11 + d1 * 7 - s0 * 5 + s1 == 0,
                             d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                             d0 - 1 == 0, d0 + 2 == 0)

// This is an artifically created system to exercise the worst case behavior of
// FM elimination - as a safeguard against improperly constructed constraint
// systems or fuzz input.
#set_fuzz_virus = (d0, d1, d2, d3, d4, d5) : ( 1089234*d0 + 203472*d1 + 82342 >= 0,
                            -55*d0 + 24*d1 + 238*d2 - 234*d3 - 9743 >= 0,
                            -5445*d0 - 284*d1 + 23*d2 + 34*d3 - 5943 >= 0,
                            -5445*d0 + 284*d1 + 238*d2 - 34*d3 >= 0,
                            445*d0 + 284*d1 + 238*d2 + 39*d3 >= 0,
                            -545*d0 + 214*d1 + 218*d2 - 94*d3 >= 0,
                            44*d0 - 184*d1 - 231*d2 + 14*d3 >= 0,
                            -45*d0 + 284*d1 + 138*d2 - 39*d3 >= 0,
                            154*d0 - 84*d1 + 238*d2 - 34*d3 >= 0,
                            54*d0 - 284*d1 - 223*d2 + 384*d3 >= 0,
                            -55*d0 + 284*d1 + 23*d2 + 34*d3 >= 0,
                            54*d0 - 84*d1 + 28*d2 - 34*d3 >= 0,
                            54*d0 - 24*d1 - 23*d2 + 34*d3 >= 0,
                            -55*d0 + 24*d1 + 23*d2 + 4*d3 >= 0,
                            15*d0 - 84*d1 + 238*d2 - 3*d3 >= 0,
                            5*d0 - 24*d1 - 223*d2 + 84*d3 >= 0,
                            -5*d0 + 284*d1 + 23*d2 - 4*d3 >= 0,
                            14*d0 + 4*d2 + 7234 >= 0,
                            -174*d0 - 534*d2 + 9834 >= 0,
                            194*d0 - 954*d2 + 9234 >= 0,
                            47*d0 - 534*d2 + 9734 >= 0,
                            -194*d0 - 934*d2 + 984 >= 0,
                            -947*d0 - 953*d2 + 234 >= 0,
                            184*d0 - 884*d2 + 884 >= 0,
                            -174*d0 + 834*d2 + 234 >= 0,
                            844*d0 + 634*d2 + 9874 >= 0,
                            -797*d2 - 79*d3 + 257 >= 0,
                            2039*d0 + 793*d2 - 99*d3 - 24*d4 + 234*d5 >= 0,
                            78*d2 - 788*d5 + 257 >= 0,
                            d3 - (d5 + 97*d0) floordiv 423 >= 0,
                            234* (d0 + d3 mod 5 floordiv 2342) mod 2309 
                            + (d0 + 2038*d3) floordiv 208 >= 0,
                            239* (d0 + 2300 * d3) floordiv 2342 
                            mod 2309 mod 239423 == 0,
                            d0 + d3 mod 2642 + (d3 + 2*d0) mod 1247 
                            mod 2038 mod 2390 mod 2039 floordiv 55 >= 0
)

// CHECK-LABEL: func @test_gaussian_elimination_empty_set0() {
func @test_gaussian_elimination_empty_set0() {
  for %i0 = 1 to 10 {
    for %i1 = 1 to 100 {
      // CHECK: [[SET_EMPTY_2D]](%i0, %i1)
      if (d0, d1) : (2 == 0)(%i0, %i1) {
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_empty_set1() {
func @test_gaussian_elimination_empty_set1() {
  for %i0 = 1 to 10 {
    for %i1 = 1 to 100 {
      // CHECK: [[SET_EMPTY_2D]](%i0, %i1)
      if (d0, d1) : (1 >= 0, -1 >= 0) (%i0, %i1) {
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_non_empty_set2() {
func @test_gaussian_elimination_non_empty_set2() {
  for %i0 = 1 to 10 {
    for %i1 = 1 to 100 {
      // CHECK: #set1(%i0, %i1)
      if #set2(%i0, %i1) {
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_empty_set3() {
func @test_gaussian_elimination_empty_set3() {
  %c7 = constant 7 : index
  %c11 = constant 11 : index
  for %i0 = 1 to 10 {
    for %i1 = 1 to 100 {
      // CHECK: #set2(%i0, %i1)[%c7, %c11]
      if #set3(%i0, %i1)[%c7, %c11] {
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_non_empty_set4() {
func @test_gaussian_elimination_non_empty_set4() {
  %c7 = constant 7 : index
  %c11 = constant 11 : index
  for %i0 = 1 to 10 {
    for %i1 = 1 to 100 {
      // CHECK: #set3(%i0, %i1)[%c7, %c11]
      if #set4(%i0, %i1)[%c7, %c11] {
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_gaussian_elimination_empty_set5() {
func @test_gaussian_elimination_empty_set5() {
  %c7 = constant 7 : index
  %c11 = constant 11 : index
  for %i0 = 1 to 10 {
    for %i1 = 1 to 100 {
      // CHECK: #set2(%i0, %i1)[%c7, %c11]
      if #set5(%i0, %i1)[%c7, %c11] {
      }
    }
  }
  return
}

// CHECK-LABEL: func @test_fuzz_explosion
func @test_fuzz_explosion(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) {
  for %i0 = 1 to 10 {
    for %i1 = 1 to 100 {
      if #set_fuzz_virus(%i0, %i1, %arg0, %arg1, %arg2, %arg3) {
      }
    }
  }
  return
}


// CHECK-LABEL: func @test_empty_set(%arg0: index) {
func @test_empty_set(%N : index) {
  for %i = 0 to 10 {
    for %j = 0 to 10 {
      // CHECK: if [[SET_EMPTY_2D]](%i0, %i1)
      if (d0, d1) : (d0 - d1 >= 0, d1 - d0 - 1 >= 0)(%i, %j) {
        "foo"() : () -> ()
      }
      // CHECK: if [[SET_EMPTY_1D]](%i0)
      if (d0) : (d0 >= 0, -d0 - 1 >= 0)(%i) {
        "bar"() : () -> ()
      }
      // CHECK: if [[SET_EMPTY_1D]](%i0)
      if (d0) : (d0 >= 0, -d0 - 1 >= 0)(%i) {
        "foo"() : () -> ()
      }
      // CHECK: if [[SET_EMPTY_1D_2S]](%i0)[%arg0, %arg0]
      if (d0)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0, -s0 >= 0)(%i)[%N, %N] {
        "bar"() : () -> ()
      }
      // CHECK: if [[SET_EMPTY_3D]](%i0, %i1, %arg0)
      // The set below implies d0 = d1; so d1 >= d0, but d0 >= d1 + 1.
      if (d0, d1, d2) : (d0 - d1 == 0, d2 - d0 >= 0, d0 - d1 - 1 >= 0)(%i, %j, %N) {
        "foo"() : () -> ()
      }
      // CHECK: if [[SET_EMPTY_2D]](%i0, %i1)
      // The set below has rational solutions but no integer solutions; GCD test catches it.
      if (d0, d1) : (d0*2 -d1*2 - 1 == 0, d0 >= 0, -d0 + 100 >= 0, d1 >= 0, -d1 + 100 >= 0)(%i, %j) {
        "foo"() : () -> ()
      }
      // CHECK: if [[SET_EMPTY_2D]](%i0, %i1)
      if (d0, d1) : (d1 == 0, d0 - 1 >= 0, - d0 - 1 >= 0)(%i, %j) {
        "foo"() : () -> ()
      }
    }
  }
  // The tests below test GCDTightenInequalities().
  for %k = 0 to 10 {
    for %l = 0 to 10 {
      // Empty because no multiple of 8 lies between 4 and 7.
      // CHECK: if [[SET_EMPTY_1D]](%i2)
      if (d0) : (8*d0 - 4 >= 0, -8*d0 + 7 >= 0)(%k) {
        "foo"() : () -> ()
      }
      // Same as above but with equalities and inequalities.
      // CHECK: if [[SET_EMPTY_2D]](%i2, %i3)
      if (d0, d1) : (d0 - 4*d1 == 0, 4*d1 - 5 >= 0, -4*d1 + 7 >= 0)(%k, %l) {
        "foo"() : () -> ()
      }
      // Same as above but with a combination of multiple identifiers. 4*d0 +
      // 8*d1 here is a multiple of 4, and so can't lie between 9 and 11. GCD
      // tightening will tighten constraints to 4*d0 + 8*d1 >= 12 and 4*d0 +
      // 8*d1 <= 8; hence infeasible.
      // CHECK: if [[SET_EMPTY_2D]](%i2, %i3)
      if (d0, d1) : (4*d0 + 8*d1 - 9 >= 0, -4*d0 - 8*d1 + 11 >=  0)(%k, %l) {
        "foo"() : () -> ()
      }
      // Same as above but with equalities added into the mix.
      // CHECK: if [[SET_EMPTY_3D]](%i2, %i2, %i3)
      if (d0, d1, d2) : (d0 - 4*d2 == 0, d0 + 8*d1 - 9 >= 0, -d0 - 8*d1 + 11 >=  0)(%k, %k, %l) {
        "foo"() : () -> ()
      }
    }
  }

  for %m = 0 to 10 {
    // CHECK: if [[SET_EMPTY_1D]](%i{{[0-9]+}})
    if (d0) : (d0 mod 2 - 3 == 0) (%m) {
      "foo"() : () -> ()
    }
  }

  return
}
