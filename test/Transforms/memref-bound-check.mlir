// RUN: mlir-opt %s -memref-bound-check -split-input-file -verify | FileCheck %s

// -----

// CHECK-LABEL: mlfunc @test() {
mlfunc @test() {
  %zero = constant 0 : index
  %minusone = constant -1 : index
  %sym = constant 111 : index

  %A = alloc() : memref<9 x 9 x i32>
  %B = alloc() : memref<111 x i32>

  for %i = -1 to 10 {
    for %j = -1 to 10 {
      %idx = affine_apply (d0, d1) -> (d0, d1)(%i, %j)
      // Out of bound access.
      %x  = load %A[%idx#0, %idx#1] : memref<9 x 9 x i32>  
      // expected-error@-1 {{'load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'load' op memref out of lower bound access along dimension #1}}
      // expected-error@-3 {{'load' op memref out of upper bound access along dimension #2}}
      // expected-error@-4 {{'load' op memref out of lower bound access along dimension #2}}
      // This will access 0 to 110 - hence an overflow.
      %idy = affine_apply (d0, d1) -> (10*d0 - d1 + 19)(%i, %j)
      %y = load %B[%idy] : memref<111 x i32>
    }
  }

  for %k = 0 to 10 {
      // In bound.
      %u = load %B[%zero] : memref<111 x i32>
      // Out of bounds.
      %v = load %B[%sym] : memref<111 x i32> // expected-error {{'load' op memref out of upper bound access along dimension #1}}
      // Out of bounds.
      %w = load %B[%minusone] : memref<111 x i32>  // expected-error {{'load' op memref out of lower bound access along dimension #1}}
  }
  return
}

// CHECK-LABEL: mlfunc @test_
mlfunc @test_mod_floordiv_ceildiv() {
  %zero = constant 0 : index
  %A = alloc() : memref<128 x 64 x 64 x i32>

  for %i = 0 to 256 {
    for %j = 0 to 256 {
      %idx = affine_apply (d0, d1, d2) -> (d0 mod 128 + 1, d1 floordiv 4 + 1, d2 ceildiv 4)(%i, %j, %j)
      %x  = load %A[%idx#0, %idx#1, %idx#2] : memref<128 x 64 x 64 x i32>
      // expected-error@-1 {{'load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'load' op memref out of upper bound access along dimension #2}}
      // expected-error@-3 {{'load' op memref out of upper bound access along dimension #3}}
      %idy = affine_apply (d0, d1, d2) -> (d0 mod 128, d1 floordiv 4, d2 ceildiv 4 - 1)(%i, %j, %j)
      %y  = load %A[%idy#0, %idy#1, %idy#2] : memref<128 x 64 x 64 x i32> // expected-error {{'load' op memref out of lower bound access along dimension #3}}
      // CHECK-EMPTY
    } // CHECK }
  } // CHECK }
  return
}

// CHECK-LABEL: mlfunc @test_no_out_of_bounds()
mlfunc @test_no_out_of_bounds() {
  %zero = constant 0 : index
  %A = alloc() : memref<257 x 256 x i32>
  %C = alloc() : memref<257 x i32>
  %B = alloc() : memref<1 x i32>

  for %i = 0 to 256 {
    for %j = 0 to 256 {
      // All of these accesses are in bound; check that no errors are emitted.
      // CHECK: %3 = affine_apply #map4(%i0, %i1)
      // CHECK-NEXT: %4 = load %0[%3#0, %c0] : memref<257x256xi32>
      // CHECK-NEXT: %5 = affine_apply #map5(%i0, %i0)
      // CHECK-NEXT: %6 = load %2[%5] : memref<1xi32>
      %idx = affine_apply (d0, d1) -> ( 64 * (d0 ceildiv 64), d1 floordiv 4 + d1 mod 4)(%i, %j)
      // Without GCDTightenInequalities(), the upper bound on the region
      // accessed along first memref dimension would have come out as d0 <= 318
      // (instead of d0 <= 256), and led to a false positive out of bounds.
      %x  = load %A[%idx#0, %zero] : memref<257 x 256 x i32>
      %idy = affine_apply (d0, d1) -> (d0 floordiv 256)(%i, %i)
      %y  = load %B[%idy] : memref<1 x i32>
    } // CHECK-NEXT }
  }
  return
}

// CHECK-LABEL: mlfunc @test_semi_affine_bailout
mlfunc @test_semi_affine_bailout(%N : index) {
  %B = alloc() : memref<10 x i32>
  for %i = 0 to 10 {
    %idx = affine_apply (d0)[s0] -> (d0 * s0)(%i)[%N]
    %y = load %B[%idx] : memref<10 x i32>
  }
  return
}

// CHECK-LABEL: mlfunc @multi_mod_floordiv
mlfunc @multi_mod_floordiv() {
  %A = alloc() : memref<2x2xi32>
  for %ii = 0 to 64 {
      %idx = affine_apply (d0) -> ((d0 mod 147456) floordiv 1152,
                                  ((d0 mod 147456) mod 1152) floordiv 384) (%ii)
      %v = load %A[%idx#0, %idx#1] : memref<2x2xi32>
  }
  return
}

// CHECK-LABEL: mlfunc @delinearize_mod_floordiv
mlfunc @delinearize_mod_floordiv() {
  %c0 = constant 0 : index
  %in = alloc() : memref<2x2x3x3x16x1xi32>
  %out = alloc() : memref<64x9xi32>

  // Reshape '%in' into '%out'.
  for %ii = 0 to 64 {
    for %jj = 0 to 9 {
      %a0 = affine_apply (d0, d1) -> (d0 * (9 * 1024) + d1 * 128) (%ii, %jj)
      %a1 = affine_apply (d0) ->
        (d0 floordiv (2 * 3 * 3 * 128 * 128),
        (d0 mod 294912) floordiv (3 * 3 * 128 * 128),
        (((d0 mod 294912) mod 147456) floordiv 1152) floordiv 8,
        (((d0 mod 294912) mod 147456) mod 1152) floordiv 384,
        ((((d0 mod 294912) mod 147456) mod 1152) mod 384) floordiv 128,
        (((((d0 mod 294912) mod 147456) mod 1152) mod 384) mod 128)
          floordiv 128) (%a0)
      %v0 = load %in[%a1#0, %a1#1, %a1#3, %a1#4, %a1#2, %a1#5]
        : memref<2x2x3x3x16x1xi32>
    }
  }
  return
}
