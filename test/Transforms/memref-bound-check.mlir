// RUN: mlir-opt %s -memref-bound-check -split-input-file -verify | FileCheck %s

// -----

// CHECK-LABEL: func @test() {
func @test() {
  %zero = constant 0 : index
  %minusone = constant -1 : index
  %sym = constant 111 : index

  %A = alloc() : memref<9 x 9 x i32>
  %B = alloc() : memref<111 x i32>

  for %i = -1 to 10 {
    for %j = -1 to 10 {
      %idx0 = affine.apply (d0, d1) -> (d0)(%i, %j)
      %idx1 = affine.apply (d0, d1) -> (d1)(%i, %j)
      // Out of bound access.
      %x  = load %A[%idx0, %idx1] : memref<9 x 9 x i32>  
      // expected-error@-1 {{'std.load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'std.load' op memref out of lower bound access along dimension #1}}
      // expected-error@-3 {{'std.load' op memref out of upper bound access along dimension #2}}
      // expected-error@-4 {{'std.load' op memref out of lower bound access along dimension #2}}
      // This will access 0 to 110 - hence an overflow.
      %idy = affine.apply (d0, d1) -> (10*d0 - d1 + 19)(%i, %j)
      %y = load %B[%idy] : memref<111 x i32>
    }
  }

  for %k = 0 to 10 {
      // In bound.
      %u = load %B[%zero] : memref<111 x i32>
      // Out of bounds.
      %v = load %B[%sym] : memref<111 x i32> // expected-error {{'std.load' op memref out of upper bound access along dimension #1}}
      // Out of bounds.
      store %v, %B[%minusone] : memref<111 x i32>  // expected-error {{'std.store' op memref out of lower bound access along dimension #1}}
  }
  return
}

// CHECK-LABEL: func @test_mod_floordiv_ceildiv
func @test_mod_floordiv_ceildiv() {
  %zero = constant 0 : index
  %A = alloc() : memref<128 x 64 x 64 x i32>

  for %i = 0 to 256 {
    for %j = 0 to 256 {
      %idx0 = affine.apply (d0, d1, d2) -> (d0 mod 128 + 1)(%i, %j, %j)
      %idx1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4 + 1)(%i, %j, %j)
      %idx2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4)(%i, %j, %j)
      %x  = load %A[%idx0, %idx1, %idx2] : memref<128 x 64 x 64 x i32>
      // expected-error@-1 {{'std.load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'std.load' op memref out of upper bound access along dimension #2}}
      // expected-error@-3 {{'std.load' op memref out of upper bound access along dimension #3}}
      %idy0 = affine.apply (d0, d1, d2) -> (d0 mod 128)(%i, %j, %j)
      %idy1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4)(%i, %j, %j)
      %idy2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4 - 1)(%i, %j, %j)
      store %x, %A[%idy0, %idy1, %idy2] : memref<128 x 64 x 64 x i32> // expected-error {{'std.store' op memref out of lower bound access along dimension #3}}
      // CHECK-EMPTY
    } // CHECK }
  } // CHECK }
  return
}

// CHECK-LABEL: func @test_no_out_of_bounds()
func @test_no_out_of_bounds() {
  %zero = constant 0 : index
  %A = alloc() : memref<257 x 256 x i32>
  %C = alloc() : memref<257 x i32>
  %B = alloc() : memref<1 x i32>

  for %i = 0 to 256 {
    for %j = 0 to 256 {
      // All of these accesses are in bound; check that no errors are emitted.
      // CHECK: %3 = affine.apply {{#map.*}}(%i0, %i1)
      // CHECK-NEXT: %4 = load %0[%3, %c0] : memref<257x256xi32>
      // CHECK-NEXT: %5 = affine.apply {{#map.*}}(%i0, %i0)
      // CHECK-NEXT: %6 = load %2[%5] : memref<1xi32>
      %idx0 = affine.apply (d0, d1) -> ( 64 * (d0 ceildiv 64))(%i, %j)
      // Without GCDTightenInequalities(), the upper bound on the region
      // accessed along first memref dimension would have come out as d0 <= 318
      // (instead of d0 <= 256), and led to a false positive out of bounds.
      %x  = load %A[%idx0, %zero] : memref<257 x 256 x i32>
      %idy = affine.apply (d0, d1) -> (d0 floordiv 256)(%i, %i)
      %y  = load %B[%idy] : memref<1 x i32>
    } // CHECK-NEXT }
  }
  return
}

// CHECK-LABEL: func @mod_div
func @mod_div() {
  %zero = constant 0 : index
  %A = alloc() : memref<128 x 64 x 64 x i32>

  for %i = 0 to 256 {
    for %j = 0 to 256 {
      %idx0 = affine.apply (d0, d1, d2) -> (d0 mod 128 + 1)(%i, %j, %j)
      %idx1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4 + 1)(%i, %j, %j)
      %idx2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4)(%i, %j, %j)
      %x  = load %A[%idx0, %idx1, %idx2] : memref<128 x 64 x 64 x i32>
      // expected-error@-1 {{'std.load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'std.load' op memref out of upper bound access along dimension #2}}
      // expected-error@-3 {{'std.load' op memref out of upper bound access along dimension #3}}
      %idy0 = affine.apply (d0, d1, d2) -> (d0 mod 128)(%i, %j, %j)
      %idy1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4)(%i, %j, %j)
      %idy2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4 - 1)(%i, %j, %j)
      store %x, %A[%idy0, %idy1, %idy2] : memref<128 x 64 x 64 x i32> // expected-error {{'std.store' op memref out of lower bound access along dimension #3}}
    }
  }
  return
}

// Tests with nested mod's and floordiv's.
// CHECK-LABEL: func @mod_floordiv_nested() {
func @mod_floordiv_nested() {
  %A = alloc() : memref<256 x 256 x i32>
  for %i = 0 to 256 {
    for %j = 0 to 256 {
      %idx0 = affine.apply (d0, d1) -> ((d0 mod 1024) floordiv 4)(%i, %j)
      %idx1 = affine.apply (d0, d1) -> ((((d1 mod 128) mod 32) ceildiv 4) * 32)(%i, %j)
      load %A[%idx0, %idx1] : memref<256 x 256 x i32> // expected-error {{'std.load' op memref out of upper bound access along dimension #2}}
    }
  }
  return
}

// CHECK-LABEL: func @test_semi_affine_bailout
func @test_semi_affine_bailout(%N : index) {
  %B = alloc() : memref<10 x i32>
  for %i = 0 to 10 {
    %idx = affine.apply (d0)[s0] -> (d0 * s0)(%i)[%N]
    %y = load %B[%idx] : memref<10 x i32>
    // expected-error@-1 {{getMemRefRegion: compose affine map failed}}
  }
  return
}

// CHECK-LABEL: func @multi_mod_floordiv
func @multi_mod_floordiv() {
  %A = alloc() : memref<2x2xi32>
  for %ii = 0 to 64 {
      %idx0 = affine.apply (d0) -> ((d0 mod 147456) floordiv 1152) (%ii)
      %idx1 = affine.apply (d0) -> (((d0 mod 147456) mod 1152) floordiv 384) (%ii)
      %v = load %A[%idx0, %idx1] : memref<2x2xi32>
  }
  return
}

// CHECK-LABEL: func @delinearize_mod_floordiv
func @delinearize_mod_floordiv() {
  %c0 = constant 0 : index
  %in = alloc() : memref<2x2x3x3x16x1xi32>
  %out = alloc() : memref<64x9xi32>

  // Reshape '%in' into '%out'.
  for %ii = 0 to 64 {
    for %jj = 0 to 9 {
      %a0 = affine.apply (d0, d1) -> (d0 * (9 * 1024) + d1 * 128) (%ii, %jj)
      %a10 = affine.apply (d0) ->
        (d0 floordiv (2 * 3 * 3 * 128 * 128)) (%a0)
      %a11 = affine.apply (d0) ->
        ((d0 mod 294912) floordiv (3 * 3 * 128 * 128)) (%a0)
      %a12 = affine.apply (d0) ->
        ((((d0 mod 294912) mod 147456) floordiv 1152) floordiv 8) (%a0)
      %a13 = affine.apply (d0) ->
        ((((d0 mod 294912) mod 147456) mod 1152) floordiv 384) (%a0)
      %a14 = affine.apply (d0) ->
        (((((d0 mod 294912) mod 147456) mod 1152) mod 384) floordiv 128) (%a0)
      %a15 = affine.apply (d0) ->
        ((((((d0 mod 294912) mod 147456) mod 1152) mod 384) mod 128)
          floordiv 128) (%a0)
      %v0 = load %in[%a10, %a11, %a13, %a14, %a12, %a15]
        : memref<2x2x3x3x16x1xi32>
    }
  }
  return
}

// CHECK-LABEL: func @zero_d_memref
func @zero_d_memref(%arg0: memref<i32>) {
  %c0 = constant 0 : i32
  // A 0-d memref always has in-bound accesses!
  store %c0, %arg0[] : memref<i32>
  return
}

// CHECK-LABEL: func @out_of_bounds
func @out_of_bounds() {
  %in = alloc() : memref<1xi32>
  %c9 = constant 9 : i32

  for %i0 = 10 to 11 {
    %idy = affine.apply (d0) ->  (100 * d0 floordiv 1000) (%i0)
    store %c9, %in[%idy] : memref<1xi32> // expected-error {{'std.store' op memref out of upper bound access along dimension #1}}
  }
  return
}

// -----

// This test case accesses within bounds. Without removal of a certain type of
// trivially redundant constraints (those differing only in their constant
// term), the number of constraints here explodes, and this would return out of
// bounds errors conservatively due to FlatAffineConstraints::kExplosionFactor.
#map3 = (d0, d1) -> ((d0 * 72 + d1) floordiv 2304 + ((((d0 * 72 + d1) mod 2304) mod 1152) mod 9) floordiv 3)
#map4 = (d0, d1) -> ((d0 * 72 + d1) mod 2304 - (((d0 * 72 + d1) mod 2304) floordiv 1152) * 1151 - ((((d0 * 72 + d1) mod 2304) mod 1152) floordiv 9) * 9 - (((((d0 * 72 + d1) mod 2304) mod 1152) mod 9) floordiv 3) * 3)
#map5 = (d0, d1) -> (((((d0 * 72 + d1) mod 2304) mod 1152) floordiv 9) floordiv 8)
// CHECK-LABEL: func @test_complex_mod_floordiv
func @test_complex_mod_floordiv(%arg0: memref<4x4x16x1xf32>) {
  %c0 = constant 0 : index
  %0 = alloc() : memref<1x2x3x3x16x1xf32>
  for %i0 = 0 to 64 {
    for %i1 = 0 to 9 {
      %2 = affine.apply #map3(%i0, %i1)
      %3 = affine.apply #map4(%i0, %i1)
      %4 = affine.apply #map5(%i0, %i1)
      %5 = load %arg0[%2, %c0, %4, %c0] : memref<4x4x16x1xf32>
    }
  }
  return
}

// -----

// The first load is within bounds, but not the second one.
#map0 = (d0) -> (d0 mod 4)
#map1 = (d0) -> (d0 mod 4 + 4)

// CHECK-LABEL: func @test_mod_bound
func @test_mod_bound() {
  %0 = alloc() : memref<7 x f32>
  %1 = alloc() : memref<6 x f32>
  for %i0 = 0 to 4096 {
    for %i1 = #map0(%i0) to #map1(%i0) {
      load %0[%i1] : memref<7 x f32>
      load %1[%i1] : memref<6 x f32>
      // expected-error@-1 {{'std.load' op memref out of upper bound access along dimension #1}}
    }
  }
  return
}

// -----

#map0 = (d0) -> (d0 floordiv 4)
#map1 = (d0) -> (d0 floordiv 4 + 4)
#map2 = (d0) -> (4 * (d0 floordiv 4)  + d0 mod 4)

// CHECK-LABEL: func @test_floordiv_bound
func @test_floordiv_bound() {
  %0 = alloc() : memref<1027 x f32>
  %1 = alloc() : memref<1026 x f32>
  %2 = alloc() : memref<4096 x f32>
  %N = constant 2048 : index
  for %i0 = 0 to 4096 {
    for %i1 = #map0(%i0) to #map1(%i0) {
      load %0[%i1] : memref<1027 x f32>
      load %1[%i1] : memref<1026 x f32>
      // expected-error@-1 {{'std.load' op memref out of upper bound access along dimension #1}}
    }
    for %i2 = 0 to #map2(%N) {
      // Within bounds.
      %v = load %2[%i2] : memref<4096 x f32>
    }
  }
  return
}
