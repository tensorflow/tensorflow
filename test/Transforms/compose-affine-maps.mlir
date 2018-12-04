// RUN: mlir-opt %s -compose-affine-maps | FileCheck %s

// Affine maps for test case: compose_affine_maps_1dto2d_no_symbols
// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0 - 1, d0 - 1)
// CHECK: [[MAP1:#map[0-9]+]] = (d0) -> (d0 + 1, d0 + 1)
// CHECK: [[MAP2:#map[0-9]+]] = (d0) -> (d0 - 1, d0 + 1)
// CHECK: [[MAP3:#map[0-9]+]] = (d0) -> (d0 + 1, d0 - 1)

// Affine maps for test case: compose_affine_maps_1dto2d_with_symbols
// CHECK: [[MAP4:#map[0-9]+]] = (d0)[s0] -> (d0 - s0, d0 - s0)
// CHECK: [[MAP5:#map[0-9]+]] = (d0)[s0] -> (d0 - s0, d0 + 1)
// CHECK: [[MAP6:#map[0-9]+]] = (d0)[s0] -> (d0 + 1, d0 - s0)
// CHECK: [[MAP7:#map[0-9]+]] = (d0)[s0, s1] -> (d0 + s1, d0 - s0)

// Affine map for test case: compose_affine_maps_d2_tile
// CHECK: [[MAP8:#map[0-9]+]] = (d0, d1, d2, d3)[s0, s1] -> ((d0 ceildiv s0) * s0 + d2 mod s0, (d1 ceildiv s1) * s1 + d3 mod s1)

// Affine maps for test case: compose_affine_maps_dependent_loads
// CHECK: [[MAP9:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d0 + s0, d1 - s1)
// CHECK: [[MAP10:#map[0-9]+]] = (d0, d1)[s0] -> (d0 + s0, d1 * s0)
// CHECK: [[MAP11:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d1 - s1, d0 + s0)
// CHECK: [[MAP12:#map[0-9]+]] = (d0, d1)[s0, s1] -> ((d1 - s0) * s1, (d0 + s1) ceildiv s0)

// Affine maps for test case: compose_affine_maps_diamond_dependency
// CHECK: [[MAP13:#map[0-9]+]] = (d0) -> ((d0 + 6) ceildiv 8, ((d0 - 1) * 4) floordiv 3)

// Affine maps for test case: arg_used_as_dim_and_symbol
// CHECK: [[MAP14:#map[0-9]+]] = (d0, d1, d2, d3)[s0, s1] -> (d2, d3 + s0 + s1 - (d0 + d1))

// CHECK-LABEL: mlfunc @compose_affine_maps_1dto2d_no_symbols() {
mlfunc @compose_affine_maps_1dto2d_no_symbols() {
  %0 = alloc() : memref<4x4xf32>

  for %i0 = 0 to 15 {
    // Test load[%x, %x]

    %x0 = affine_apply (d0) -> (d0 - 1) (%i0)
    %x1 = affine_apply (d0, d1) -> (d0, d1) (%x0, %x0)

    // CHECK: [[I0:%[0-9]+]] = affine_apply [[MAP0]](%i0)
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
    %v0 = load %0[%x1#0, %x1#1] : memref<4x4xf32>

    // Test load[%y, %y]
    %y0 = affine_apply (d0) -> (d0 + 1) (%i0)
    %y1 = affine_apply (d0, d1) -> (d0, d1) (%y0, %y0)

    // CHECK-NEXT: [[I1:%[0-9]+]] = affine_apply [[MAP1]](%i0)  
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I1]]#0, [[I1]]#1{{\]}}
    %v1 = load %0[%y1#0, %y1#1] : memref<4x4xf32>

    // Test load[%x, %y]
    %xy = affine_apply (d0, d1) -> (d0, d1) (%x0, %y0)
    // CHECK-NEXT: [[I2:%[0-9]+]] = affine_apply [[MAP2]](%i0)  
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I2]]#0, [[I2]]#1{{\]}}
    %v2 = load %0[%xy#0, %xy#1] : memref<4x4xf32>

    // Test load[%y, %x]
    %yx = affine_apply (d0, d1) -> (d0, d1) (%y0, %x0)
    // CHECK-NEXT: [[I3:%[0-9]+]] = affine_apply [[MAP3]](%i0)  
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I3]]#0, [[I3]]#1{{\]}}
    %v3 = load %0[%yx#0, %yx#1] : memref<4x4xf32>
  }
  return
}

// CHECK-LABEL: mlfunc @compose_affine_maps_1dto2d_with_symbols() {
mlfunc @compose_affine_maps_1dto2d_with_symbols() {
  %0 = alloc() : memref<4x4xf32>
  
  for %i0 = 0 to 15 {
    // Test load[%x0, %x0] with symbol %c4
    %c4 = constant 4 : index
    %x0 = affine_apply (d0)[s0] -> (d0 - s0) (%i0)[%c4]
    %y0 = affine_apply (d0, d1) -> (d0, d1) (%x0, %x0)

    // CHECK: [[I0:%[0-9]+]] = affine_apply [[MAP4]](%i0)[%c4]  
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
    %v0 = load %0[%y0#0, %y0#1] : memref<4x4xf32>

    // Test load[%x0, %x1] with symbol %c4 captured by '%x0' map.
    %x1 = affine_apply (d0) -> (d0 + 1) (%i0)
    %y1 = affine_apply (d0, d1) -> (d0, d1) (%x0, %x1)
    // CHECK-NEXT: [[I1:%[0-9]+]] = affine_apply [[MAP5]](%i0)[%c4]
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I1]]#0, [[I1]]#1{{\]}}
    %v1 = load %0[%y1#0, %y1#1] : memref<4x4xf32>

    // Test load[%x1, %x0] with symbol %c4 captured by '%x0' map.
    %y2 = affine_apply (d0, d1) -> (d0, d1) (%x1, %x0)
    // CHECK-NEXT: [[I2:%[0-9]+]] = affine_apply [[MAP6]](%i0)[%c4]
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I2]]#0, [[I2]]#1{{\]}}
    %v2 = load %0[%y2#0, %y2#1] : memref<4x4xf32>

    // Test load[%x2, %x0] with symbol %c4 from '%x0' and %c5 from '%x2'
    %c5 = constant 5 : index
    %x2 = affine_apply (d0)[s0] -> (d0 + s0) (%i0)[%c5]
    %y3 = affine_apply (d0, d1) -> (d0, d1) (%x2, %x0)
    // CHECK: [[I3:%[0-9]+]] = affine_apply [[MAP7]](%i0)[%c4, %c5]
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I3]]#0, [[I3]]#1{{\]}}
    %v3 = load %0[%y3#0, %y3#1] : memref<4x4xf32> 
  }
  return
}

// CHECK-LABEL: mlfunc @compose_affine_maps_2d_tile() {
mlfunc @compose_affine_maps_2d_tile() {
  %0 = alloc() : memref<16x32xf32>
  %1 = alloc() : memref<16x32xf32>

  %c4 = constant 4 : index
  %c8 = constant 8 : index

  for %i0 = 0 to 3 {
    %x0 = affine_apply (d0)[s0] -> (d0 ceildiv s0) (%i0)[%c4]
    for %i1 = 0 to 3 {
      %x1 = affine_apply (d0)[s0] -> (d0 ceildiv s0) (%i1)[%c8]
      for %i2 = 0 to 3 {
        %x2 = affine_apply (d0)[s0] -> (d0 mod s0) (%i2)[%c4]
        for %i3 = 0 to 3 {
          %x3 = affine_apply (d0)[s0] -> (d0 mod s0) (%i3)[%c8]

          %x4 = affine_apply (d0, d1, d2, d3)[s0, s1] ->
	    ((d0 * s0) + d2, (d1 * s1) + d3) (%x0, %x1, %x2, %x3)[%c4, %c8]
          // CHECK: [[I0:%[0-9]+]] = affine_apply [[MAP8]](%i0, %i1, %i2, %i3)[%c4, %c8]
	  // CHECK-NEXT: [[L0:%[0-9]+]] = load %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
          %v0 = load %0[%x4#0, %x4#1] : memref<16x32xf32> 

          // CHECK-NEXT: store [[L0]], %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
          store %v0, %1[%x4#0, %x4#1] : memref<16x32xf32> 
        }
      }  
    }
  }
  return
}

// CHECK-LABEL: mlfunc @compose_affine_maps_dependent_loads() {
mlfunc @compose_affine_maps_dependent_loads() {
  %0 = alloc() : memref<16x32xf32>
  %1 = alloc() : memref<16x32xf32>

  for %i0 = 0 to 3 {
    for %i1 = 0 to 3 {
      for %i2 = 0 to 3 {
	%c3 = constant 3 : index
        %c7 = constant 7 : index

	%x0 = affine_apply (d0, d1, d2)[s0, s1] -> (d0 + s0, d1 - s1, d2 * s0)
	  (%i0, %i1, %i2)[%c3, %c7]
	%y0 = affine_apply (d0, d1) -> (d0, d1) (%x0#0, %x0#1)

        // CHECK: [[I0:%[0-9]+]] = affine_apply [[MAP9]](%i0, %i1)[%c3, %c7]
        // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}	  
        %v0 = load %0[%y0#0, %y0#1] : memref<16x32xf32>
	
	%y1 = affine_apply (d0, d1) -> (d0, d1) (%x0#0, %x0#2)
        // CHECK-NEXT: [[I1:%[0-9]+]] = affine_apply [[MAP10]](%i0, %i2)[%c3]
	// CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I1]]#0, [[I1]]#1{{\]}}	  
        %v1 = load %0[%y1#0, %y1#1] : memref<16x32xf32> 

        // Swizzle %i0, %i1
        %y2 = affine_apply (d0, d1) -> (d0, d1) (%x0#1, %x0#0)
        // CHECK-NEXT: [[I2:%[0-9]+]] = affine_apply [[MAP11]](%i0, %i1)[%c3, %c7]
	// CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I2]]#0, [[I2]]#1{{\]}}	
        %v2 = load %0[%y2#0, %y2#1] : memref<16x32xf32> 

        // Swizzle %x0#0, %x0#1 and %c3, %c7
	%x1 = affine_apply (d0, d1)[s0, s1] -> (d0 * s1, d1 ceildiv s0)
	  (%x0#1, %x0#0)[%c7, %c3]

        // CHECK-NEXT: [[I3:%[0-9]+]] = affine_apply [[MAP12]](%i0, %i1)[%c7, %c3]
	// CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I3]]#0, [[I3]]#1{{\]}}
        %v3 = load %0[%x1#0, %x1#1] : memref<16x32xf32> 

      }  
    }
  }
  return
}

// CHECK-LABEL: mlfunc @compose_affine_maps_diamond_dependency() {
mlfunc @compose_affine_maps_diamond_dependency() {
  %0 = alloc() : memref<4x4xf32>

  for %i0 = 0 to 15 {
    %a = affine_apply (d0) -> (d0 - 1) (%i0)
    %b = affine_apply (d0) -> (d0 + 7) (%a)
    %c = affine_apply (d0) -> (d0 * 4) (%a)
    %d = affine_apply (d0, d1) -> (d0 ceildiv 8, d1 floordiv 3) (%b, %c)
    // CHECK: [[I0:%[0-9]+]] = affine_apply [[MAP13]](%i0)
    // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
    %v = load %0[%d#0, %d#1] : memref<4x4xf32>
  }

  return
}

// CHECK-LABEL: mlfunc @arg_used_as_dim_and_symbol(%arg0 : memref<100x100xf32>, %arg1 : index) {
mlfunc @arg_used_as_dim_and_symbol(%arg0 : memref<100x100xf32>, %arg1 : index) {
  %c9 = constant 9 : index
  %1 = alloc() : memref<100x100xf32, 1>
  %2 = alloc() : memref<1xi32>
  for %i0 = 0 to 100 {
    for %i1 = 0 to 100 {
      %3 = affine_apply (d0, d1)[s0, s1] -> (d0, d1 + s0 + s1)
        (%i0, %i1)[%arg1, %c9]
      %4 = affine_apply (d0, d1, d2, d3) -> (d2, d3 - (d0 + d1))
        (%arg1, %c9, %3#0, %3#1)
      // CHECK: [[I0:%[0-9]+]] = affine_apply [[MAP14]](%arg1, %c9, %i0, %i1)[%arg1, %c9]
      // CHECK-NEXT: load %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
      %5 = load %1[%4#0, %4#1] : memref<100x100xf32, 1>
    }
  }
  return
}
