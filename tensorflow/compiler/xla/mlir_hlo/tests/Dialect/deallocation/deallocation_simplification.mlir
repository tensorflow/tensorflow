// RUN: mlir-hlo-opt %s -allow-unregistered-dialect -hlo-deallocation-simplification | FileCheck %s

func.func @retain_is_dealloc() {
  %alloc = memref.alloc() : memref<2xf32>
  %alloc_owned = deallocation.own %alloc : memref<2xf32>
  "test.use"(%alloc) : (memref<2xf32>) -> ()
  deallocation.retain() of (%alloc_owned) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: test.use
// CHECK-NEXT: memref.dealloc %[[ALLOC]]

// -----

func.func @retain_of_nothing(%arg: memref<2xf32>) -> !deallocation.ownership {
  %ret = deallocation.retain(%arg) of() : (memref<2xf32>) -> (!deallocation.ownership)
  return %ret : !deallocation.ownership
}

// CHECK-LABEL: @retain_of_nothing
// CHECK-SAME: (%[[ARG:.*]]: memref<2xf32>
// CHECK-NEXT: %[[NULL:.*]] = deallocation.null
// CHECK-NEXT: return %[[NULL]]

// -----

func.func @retain_is_dealloc_for(%lb: index, %ub: index, %step: index) {
  %alloc = memref.alloc() : memref<2xf32>
  %alloc_owned = deallocation.own %alloc : memref<2xf32>
  %for:2 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %alloc, %arg1 = %alloc_owned)
      -> (memref<2xf32>, !deallocation.ownership) {
    "some.use"(%arg0) : (memref<2xf32>) -> ()
    scf.yield %arg0, %arg1 : memref<2xf32>, !deallocation.ownership
  }
  deallocation.retain() of(%for#1) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc_for
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: deallocation.null
// CHECK-NEXT: %[[FOR:.*]]:2 = scf.for
// CHECK-NEXT:   some.use
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: memref.dealloc %[[FOR]]#0
// CHECK-NEXT: return

// -----

func.func @retain_is_dealloc_reallocated(%lb: index, %ub: index, %step: index) {
  %alloc = memref.alloc() : memref<2xf32>
  %alloc_owned = deallocation.own %alloc : memref<2xf32>
  %for:2 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %alloc, %arg1 = %alloc_owned)
      -> (memref<2xf32>, !deallocation.ownership) {
    "some.use"(%arg0) : (memref<2xf32>) -> ()
    deallocation.retain() of(%arg1) : (!deallocation.ownership) -> ()
    %alloc0 = memref.alloc() : memref<2xf32>
    %alloc0_owned = deallocation.own %alloc0 : memref<2xf32>
    scf.yield %alloc, %alloc0_owned : memref<2xf32>, !deallocation.ownership
  }
  deallocation.retain() of(%for#1) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc_reallocated
// CHECK-NEXT: memref.alloc
// CHECK-NEXT: deallocation.null
// CHECK-NEXT: %[[FOR:.*]]:2 = scf.for
// CHECK:        memref.dealloc
// CHECK:      }
// CHECK:      memref.dealloc %[[FOR]]

// -----

func.func @retain_is_not_dealloc_for(
    %x: memref<2xf32>, %x_owned: !deallocation.ownership,
    %lb: index, %ub: index, %step: index) {
  %for:2 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %x, %arg1 = %x_owned)
      -> (memref<2xf32>, !deallocation.ownership) {
    "some.use"(%arg0) : (memref<2xf32>) -> ()
    deallocation.retain() of(%arg1) : (!deallocation.ownership) -> ()
    %alloc = memref.alloc() : memref<2xf32>
    %alloc_owned = deallocation.own %alloc : memref<2xf32>
    scf.yield %alloc, %alloc_owned : memref<2xf32>, !deallocation.ownership
  }
  deallocation.retain() of(%for#1) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_is_not_dealloc_for
// CHECK: %[[FOR:.*]]:2 = scf.for
// CHECK: deallocation.retain() of(%[[FOR]]#1)

// -----

func.func @retain_is_dealloc_while() {
  %a = memref.alloc() : memref<2xf32>
  %a_owned = deallocation.own %a : memref<2xf32>
  %while:2 = scf.while (%arg0 = %a, %arg1 = %a_owned)
      : (memref<2xf32>, !deallocation.ownership) -> (memref<2xf32>, !deallocation.ownership) {
    %0 = "test.make_condition"() : () -> i1
    scf.condition(%0) %arg0, %arg1 : memref<2xf32>, !deallocation.ownership
  } do {
  ^bb0(%arg0: memref<2xf32>, %arg1: !deallocation.ownership):
    "some.use"(%arg0) : (memref<2xf32>) -> ()
    deallocation.retain() of(%arg1) : (!deallocation.ownership) -> ()
    %b = memref.alloc() : memref<2xf32>
    %b_owned = deallocation.own %b : memref<2xf32>
    scf.yield %b, %b_owned: memref<2xf32>, !deallocation.ownership
  }
  deallocation.retain() of (%while#1) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc_while
// CHECK: %[[WHILE:.*]]:2 = scf.while
// CHECK: memref.dealloc %[[WHILE]]#0

// -----

func.func @retain_is_dealloc_while_permute() {
  %a = memref.alloc() : memref<f32>
  %a_owned = deallocation.own %a : memref<f32>
  %b = memref.alloc() : memref<f32>
  %b_owned = deallocation.own %b : memref<f32>
  %c = memref.alloc() : memref<f32>
  %c_owned = deallocation.own %c : memref<f32>
  %w:6 = scf.while (%arg0 = %a, %arg1 = %b, %arg2 = %c,
                    %arg3 = %a_owned, %arg4 = %b_owned, %arg5 = %c_owned)
    : (memref<f32>, memref<f32>, memref<f32>, !deallocation.ownership, !deallocation.ownership, !deallocation.ownership) ->
      (memref<f32>, memref<f32>, memref<f32>, !deallocation.ownership, !deallocation.ownership, !deallocation.ownership) {
    %cond = "test.make_condition"() : () -> i1
    scf.condition(%cond) %arg2, %arg1, %arg0, %arg5, %arg4, %arg3
      : memref<f32>, memref<f32>, memref<f32>, !deallocation.ownership, !deallocation.ownership, !deallocation.ownership
  } do {
  ^bb0(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>,
        %arg3: !deallocation.ownership, %arg4: !deallocation.ownership, %arg5: !deallocation.ownership):
    scf.yield %arg1, %arg0, %arg2, %arg4, %arg3, %arg5
      : memref<f32>, memref<f32>, memref<f32>, !deallocation.ownership, !deallocation.ownership, !deallocation.ownership
  }
  "test.use"(%w#1) : (memref<f32>) -> ()
  deallocation.retain() of (%w#3) : (!deallocation.ownership) -> ()
  deallocation.retain() of (%w#4) : (!deallocation.ownership) -> ()
  deallocation.retain() of (%w#5) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc_while_permute
// CHECK: memref.alloc
// CHECK: memref.alloc
// CHECK: memref.alloc
// CHECK: %[[WHILE:.*]]:6 = scf.while
// CHECK: memref.dealloc %[[WHILE]]
// CHECK: memref.dealloc %[[WHILE]]
// CHECK: memref.dealloc %[[WHILE]]

func.func @retain_of_null(%arg0: memref<4xi32>, %arg1: memref<4xi32>,
                          %arg2: index, %arg3: index, %arg4: index) {
  %0 = deallocation.null
  %2:4 = scf.for %arg5 = %arg2 to %arg3 step %arg4
      iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %0, %arg9 = %0) ->
      (memref<4xi32>, memref<4xi32>, !deallocation.ownership, !deallocation.ownership) {
    "test.use"(%arg6, %arg7) : (memref<4xi32>, memref<4xi32>) -> ()
    %3 = deallocation.retain(%arg6) of(%arg8)
      : (memref<4xi32>, !deallocation.ownership) -> !deallocation.ownership
    %4 = deallocation.retain(%arg7) of(%arg9)
      : (memref<4xi32>, !deallocation.ownership) -> !deallocation.ownership
    scf.yield %arg7, %arg6, %4, %3
      : memref<4xi32>, memref<4xi32>, !deallocation.ownership, !deallocation.ownership
  }
  deallocation.retain() of(%2#2) : (!deallocation.ownership) -> ()
  deallocation.retain() of(%2#3) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_of_null
// CHECK-NOT: deallocation.retain()
