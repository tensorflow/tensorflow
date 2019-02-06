// RUN: mlir-opt %s -memref-bound-check -split-input-file -verify | FileCheck %s

// -----

// CHECK-LABEL: func @test() {
func @test() {
  %zero = constant 0 : index
  %minusone = constant -1 : index
  %sym = constant 111 : index

  %A = alloc() : memref<9 x 9 x i32>
  %B = alloc() : memref<111 x i32>

  affine.for %i = -1 to 10 {
    affine.for %j = -1 to 10 {
      %idx0 = affine.apply (d0, d1) -> (d0)(%i, %j)
      %idx1 = affine.apply (d0, d1) -> (d1)(%i, %j)
      // Out of bound access.
      %x  = load %A[%idx0, %idx1] : memref<9 x 9 x i32>  
      // expected-error@-1 {{'load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'load' op memref out of lower bound access along dimension #1}}
      // expected-error@-3 {{'load' op memref out of upper bound access along dimension #2}}
      // expected-error@-4 {{'load' op memref out of lower bound access along dimension #2}}
      // This will access 0 to 110 - hence an overflow.
      %idy = affine.apply (d0, d1) -> (10*d0 - d1 + 19)(%i, %j)
      %y = load %B[%idy] : memref<111 x i32>
    }
  }

  affine.for %k = 0 to 10 {
      // In bound.
      %u = load %B[%zero] : memref<111 x i32>
      // Out of bounds.
      %v = load %B[%sym] : memref<111 x i32> // expected-error {{'load' op memref out of upper bound access along dimension #1}}
      // Out of bounds.
      store %v, %B[%minusone] : memref<111 x i32>  // expected-error {{'store' op memref out of lower bound access along dimension #1}}
  }
  return
}

// CHECK-LABEL: func @test_mod_floordiv_ceildiv
func @test_mod_floordiv_ceildiv() {
  %zero = constant 0 : index
  %A = alloc() : memref<128 x 64 x 64 x i32>

  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %idx0 = affine.apply (d0, d1, d2) -> (d0 mod 128 + 1)(%i, %j, %j)
      %idx1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4 + 1)(%i, %j, %j)
      %idx2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4)(%i, %j, %j)
      %x  = load %A[%idx0, %idx1, %idx2] : memref<128 x 64 x 64 x i32>
      // expected-error@-1 {{'load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'load' op memref out of upper bound access along dimension #2}}
      // expected-error@-3 {{'load' op memref out of upper bound access along dimension #3}}
      %idy0 = affine.apply (d0, d1, d2) -> (d0 mod 128)(%i, %j, %j)
      %idy1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4)(%i, %j, %j)
      %idy2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4 - 1)(%i, %j, %j)
      store %x, %A[%idy0, %idy1, %idy2] : memref<128 x 64 x 64 x i32> // expected-error {{'store' op memref out of lower bound access along dimension #3}}
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

  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
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

  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %idx0 = affine.apply (d0, d1, d2) -> (d0 mod 128 + 1)(%i, %j, %j)
      %idx1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4 + 1)(%i, %j, %j)
      %idx2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4)(%i, %j, %j)
      %x  = load %A[%idx0, %idx1, %idx2] : memref<128 x 64 x 64 x i32>
      // expected-error@-1 {{'load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'load' op memref out of upper bound access along dimension #2}}
      // expected-error@-3 {{'load' op memref out of upper bound access along dimension #3}}
      %idy0 = affine.apply (d0, d1, d2) -> (d0 mod 128)(%i, %j, %j)
      %idy1 = affine.apply (d0, d1, d2) -> (d1 floordiv 4)(%i, %j, %j)
      %idy2 = affine.apply (d0, d1, d2) -> (d2 ceildiv 4 - 1)(%i, %j, %j)
      store %x, %A[%idy0, %idy1, %idy2] : memref<128 x 64 x 64 x i32> // expected-error {{'store' op memref out of lower bound access along dimension #3}}
    }
  }
  return
}

// Tests with nested mod's and floordiv's.
// CHECK-LABEL: func @mod_floordiv_nested() {
func @mod_floordiv_nested() {
  %A = alloc() : memref<256 x 256 x i32>
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %idx0 = affine.apply (d0, d1) -> ((d0 mod 1024) floordiv 4)(%i, %j)
      %idx1 = affine.apply (d0, d1) -> ((((d1 mod 128) mod 32) ceildiv 4) * 32)(%i, %j)
      load %A[%idx0, %idx1] : memref<256 x 256 x i32> // expected-error {{'load' op memref out of upper bound access along dimension #2}}
    }
  }
  return
}

// CHECK-LABEL: func @test_semi_affine_bailout
func @test_semi_affine_bailout(%N : index) {
  %B = alloc() : memref<10 x i32>
  affine.for %i = 0 to 10 {
    %idx = affine.apply (d0)[s0] -> (d0 * s0)(%i)[%N]
    %y = load %B[%idx] : memref<10 x i32>
  }
  return
}

// CHECK-LABEL: func @multi_mod_floordiv
func @multi_mod_floordiv() {
  %A = alloc() : memref<2x2xi32>
  affine.for %ii = 0 to 64 {
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
  affine.for %ii = 0 to 64 {
    affine.for %jj = 0 to 9 {
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

  affine.for %i0 = 10 to 11 {
    %idy = affine.apply (d0) ->  (100 * d0 floordiv 1000) (%i0)
    store %c9, %in[%idy] : memref<1xi32> // expected-error {{'store' op memref out of upper bound access along dimension #1}}
  }
  return
}
