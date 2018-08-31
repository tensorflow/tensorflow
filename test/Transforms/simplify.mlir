// RUN: mlir-opt %s -o - -simplify-affine-expr | FileCheck %s

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (0, 0)
#map0 = (d0, d1) -> ((d0 - d0 mod 4) mod 4, (d0 - d0 mod 128 - 64) mod 64)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 + 1, d1 * 5 + 3)
#map1 = (d0, d1) -> (d1 - d0 + (d0 - d1 + 1) * 2 + d1 - 1, 1 + 2*d1 + d1 + d1 + d1 + 2)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (0, 0, 0)
#map2 = (d0, d1) -> (((d0 - d0 mod 2) * 2) mod 4, (5*d1 + 8 - (5*d1 + 4) mod 4) mod 4, 0)

mlfunc @test() {
  for %n0 = 0 to 127 {
    for %n1 = 0 to 7 {
      %x  = affine_apply #map0(%n0, %n1)
      %y  = affine_apply #map1(%n0, %n1)
      %z  = affine_apply #map2(%n0, %n1)
    }
  }
  return
}

