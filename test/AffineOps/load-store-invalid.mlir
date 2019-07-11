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
