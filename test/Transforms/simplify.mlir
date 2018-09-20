// RUN: mlir-opt %s -simplify-affine-expr | FileCheck %s

// CHECK: #map{{[0-9]+}} = (d0, d1) -> (0, 0)
#map0 = (d0, d1) -> ((d0 - d0 mod 4) mod 4, (d0 - d0 mod 128 - 64) mod 64)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 + 1, d1 * 5 + 3)
#map1 = (d0, d1) -> (d1 - d0 + (d0 - d1 + 1) * 2 + d1 - 1, 1 + 2*d1 + d1 + d1 + d1 + 2)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (0, 0, 0)
#map2 = (d0, d1) -> (((d0 - d0 mod 2) * 2) mod 4, (5*d1 + 8 - (5*d1 + 4) mod 4) mod 4, 0)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 ceildiv 2, d0 + 1, (d1 * 3 + 1) ceildiv 2)
#map3 = (d0, d1) -> (d0 ceildiv 2, (2*d0 + 4 + 2*d0) ceildiv 4, (8*d1 + 3 + d1) ceildiv 6)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 floordiv 2, d0 * 2 + d1, (d1 + 2) floordiv 2)
#map4 = (d0, d1) -> (d0 floordiv 2, (3*d0 + 2*d1 + d0) floordiv 2, (50*d1 + 100) floordiv 100)
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (0, d0 * 5 + 3)
#map5 = (d0, d1) -> ((4*d0 + 8*d1) ceildiv 2 mod 2, (2 + d0 + (8*d0 + 2) floordiv 2))
// The flattening based simplification is currently regressive on modulo
// expression simplification in the simple case (d0 mod 8 would be turn into d0
// - 8 * (d0 floordiv 8); however, in other cases like d1 - d1 mod 8, it
// would be simplified to an arithmetically simpler and more intuitive 8 * (d1
// floordiv 8).  In general, we have a choice of using either mod or floordiv
// to express the same expression in mathematically equivalent ways, and making that
// choice to minimize the number of terms or to simplify arithmetic is a TODO. 
// CHECK: #map{{[0-9]+}} = (d0, d1) -> (d0 - (d0 floordiv 8) * 8, (d1 floordiv 8) * 8)
#map6 = (d0, d1) -> (d0 mod 8, d1 - d1 mod 8)

mlfunc @test() {
  for %n0 = 0 to 127 {
    for %n1 = 0 to 7 {
      %x  = affine_apply #map0(%n0, %n1)
      %y  = affine_apply #map1(%n0, %n1)
      %z  = affine_apply #map2(%n0, %n1)
      %w  = affine_apply #map3(%n0, %n1)
      %u  = affine_apply #map4(%n0, %n1)
      %v  = affine_apply #map5(%n0, %n1)
      %t  = affine_apply #map6(%n0, %n1)
    }
  }
  return
}

