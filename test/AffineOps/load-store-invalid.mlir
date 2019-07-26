// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @load_too_many_subscripts(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  "affine.load"(%arg0, %arg1, %arg2, %arg3) : (memref<?x?xf32>, index, index, index) -> f32
}

// -----

func @load_too_many_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.load"(%arg0, %arg1, %arg2, %arg3)
    {map = (i, j) -> (i, j) } : (memref<?x?xf32>, index, index, index) -> f32
}

// -----

func @load_too_few_subscripts(%arg0: memref<?x?xf32>, %arg1: index) {
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  "affine.load"(%arg0, %arg1) : (memref<?x?xf32>, index) -> f32
}

// -----

func @load_too_few_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.load"(%arg0, %arg1)
    {map = (i, j) -> (i, j) } : (memref<?x?xf32>, index) -> f32
}

// -----

func @store_too_many_subscripts(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index,
                                %arg3: index, %val: f32) {
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  "affine.store"(%val, %arg0, %arg1, %arg2, %arg3) : (f32, memref<?x?xf32>, index, index, index) -> ()
}

// -----

func @store_too_many_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index,
                                    %arg3: index, %val: f32) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.store"(%val, %arg0, %arg1, %arg2, %arg3)
    {map = (i, j) -> (i, j) } : (f32, memref<?x?xf32>, index, index, index) -> ()
}

// -----

func @store_too_few_subscripts(%arg0: memref<?x?xf32>, %arg1: index, %val: f32) {
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  "affine.store"(%val, %arg0, %arg1) : (f32, memref<?x?xf32>, index) -> ()
}

// -----

func @store_too_few_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index, %val: f32) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.store"(%val, %arg0, %arg1)
    {map = (i, j) -> (i, j) } : (f32, memref<?x?xf32>, index) -> ()
}

// -----

func @load_non_affine_index(%arg0 : index) {
  %0 = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %1 = muli %i0, %arg0 : index
    // expected-error@+1 {{op index must be a dimension or symbol identifier}}
    %v = affine.load %0[%1] : memref<10xf32>
  }
  return
}

// -----

func @store_non_affine_index(%arg0 : index) {
  %0 = alloc() : memref<10xf32>
  %1 = constant 11.0 : f32
  affine.for %i0 = 0 to 10 {
    %2 = muli %i0, %arg0 : index
    // expected-error@+1 {{op index must be a dimension or symbol identifier}}
    affine.store %1, %0[%2] : memref<10xf32>
  }
  return
}

// -----

func @dma_start_non_affine_src_index(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = alloc() : memref<100xf32, 2>
  %2 = alloc() : memref<1xi32, 4>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    %3 = muli %i0, %arg0 : index
    // expected-error@+1 {{op src index must be a dimension or symbol identifier}}
    affine.dma_start %0[%3], %1[%i0], %2[%c0], %c64
        : memref<100xf32>, memref<100xf32, 2>, memref<1xi32, 4>
  }
  return
}

// -----

func @dma_start_non_affine_dst_index(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = alloc() : memref<100xf32, 2>
  %2 = alloc() : memref<1xi32, 4>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    %3 = muli %i0, %arg0 : index
    // expected-error@+1 {{op dst index must be a dimension or symbol identifier}}
    affine.dma_start %0[%i0], %1[%3], %2[%c0], %c64
        : memref<100xf32>, memref<100xf32, 2>, memref<1xi32, 4>
  }
  return
}

// -----

func @dma_start_non_affine_tag_index(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = alloc() : memref<100xf32, 2>
  %2 = alloc() : memref<1xi32, 4>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    %3 = muli %i0, %arg0 : index
    // expected-error@+1 {{op tag index must be a dimension or symbol identifier}}
    affine.dma_start %0[%i0], %1[%arg0], %2[%3], %c64
        : memref<100xf32>, memref<100xf32, 2>, memref<1xi32, 4>
  }
  return
}

// -----

func @dma_wait_non_affine_tag_index(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = alloc() : memref<100xf32, 2>
  %2 = alloc() : memref<1xi32, 4>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    %3 = muli %i0, %arg0 : index
    // expected-error@+1 {{op index must be a dimension or symbol identifier}}
    affine.dma_wait %2[%3], %c64 : memref<1xi32, 4>
  }
  return
}
