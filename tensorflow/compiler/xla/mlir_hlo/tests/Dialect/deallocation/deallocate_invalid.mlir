// RUN: mlir-hlo-opt -allow-unregistered-dialect %s -split-input-file -hlo-deallocate -verify-diagnostics

func.func @dealloc_invalid(%lb: index, %ub: index, %step: index) {
  %alloc = memref.alloc() : memref<i32>
  scf.for %i = %lb to %ub step %step {  // expected-error {{can't implicitly capture across loop boundaries}}
    memref.dealloc %alloc: memref<i32>
  }
  return
}

// -----

func.func @realloc_no_else(%size: index, %cond: i1) {
  %alloc = memref.alloc(%size) : memref<?xi32>
  scf.if %cond {  // expected-error {{cannot implicitly capture from an if without else}}
    %realloc = memref.realloc %alloc(%size) : memref<?xi32> to memref<?xi32>
  }
  return
}

// -----

func.func @realloc_not_yielded(%size: index, %cond: i1) {
  %alloc = memref.alloc(%size) : memref<?xi32>
  scf.if %cond {  // expected-error {{released value not yielded on other branch}}
    %realloc = memref.realloc %alloc(%size) : memref<?xi32> to memref<?xi32>
  } else {
    "test.dummy"() : () -> ()
  }
  return
}

// -----

func.func @realloc_arg(%arg: memref<?xi32>, %size: index) {
  %realloc = memref.realloc %arg(%size) : memref<?xi32> to memref<?xi32>  // expected-error {{unable to find ownership indicator for operand}}
  return
}

// -----

func.func @realloc_twice(%size: index) {  // expected-error {{invalid realloc of memref}}
  %alloc = memref.alloc(%size) : memref<?xi32>
  %realloc0 = memref.realloc %alloc(%size) : memref<?xi32> to memref<?xi32>
  %realloc1 = memref.realloc %alloc(%size) : memref<?xi32> to memref<?xi32>
  return
}

// -----

func.func @realloc_twice_in_if(%size: index, %cond: i1) {  // expected-error {{invalid realloc of memref}}
  %alloc = memref.alloc(%size) : memref<?xi32>
  scf.if %cond -> memref<?xi32> {
    %realloc = memref.realloc %alloc(%size) : memref<?xi32> to memref<?xi32>
    scf.yield %realloc : memref<?xi32>
  } else {
    scf.yield %alloc : memref<?xi32>
  }
  scf.if %cond -> memref<?xi32> {
    %realloc = memref.realloc %alloc(%size) : memref<?xi32> to memref<?xi32>
    scf.yield %realloc : memref<?xi32>
  } else {
    scf.yield %alloc : memref<?xi32>
  }
  return
}

// -----

func.func @cross_loop_boundary(%size: index, %lb: index, %ub: index, %step: index) {
  %alloc = memref.alloc(%size) : memref<?xi32>
  scf.for %i = %lb to %ub step %step {  // expected-error {{can't implicitly capture across loop boundaries}}
    memref.realloc %alloc(%size) : memref<?xi32> to memref<?xi32>
  }
  return
}
