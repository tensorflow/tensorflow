// RUN: mlir-hlo-opt %s -split-input-file -allow-unregistered-dialect \
// RUN:                 -hlo-buffer-reuse | \
// RUN:   FileCheck %s

func.func @simple_reuse() {
  %condition = "test.make_condition"() : () -> i1
  scf.if %condition {
    %alloc0 = memref.alloc() : memref<2xf32>
    "test.use"(%alloc0) : (memref<2xf32>) -> ()
    memref.dealloc %alloc0 : memref<2xf32>
    %alloc1 = memref.alloc() : memref<2xf32>
    "test.use"(%alloc1) : (memref<2xf32>) -> ()
    memref.dealloc %alloc1 : memref<2xf32>
  }
  return
}

// CHECK-LABEL: @simple_reuse
//      CHECK: scf.if
// CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT:   "test.use"(%[[ALLOC]])
// CHECK-NEXT:   "test.use"(%[[ALLOC]])
// CHECK-NEXT:   memref.dealloc %[[ALLOC]]
// CHECK-NEXT: }

// -----

func.func @hoist_from_for(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    %alloc = memref.alloc() : memref<f32>
    "test.use"(%alloc) : (memref<f32>) -> ()
    memref.dealloc %alloc : memref<f32>
  }
  return
}

// CHECK-LABEL: @hoist_from_for
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloca() : memref<f32>
// CHECK-NEXT: scf.for
// CHECK-NEXT:   test.use
// CHECK-NEXT: }

// -----

func.func @hoist_from_nested_for(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %alloc = memref.alloc() : memref<f32>
     "test.use"(%alloc) : (memref<f32>) -> ()
       memref.dealloc %alloc : memref<f32>
    }
  }
  return
}

// CHECK-LABEL: @hoist_from_nested_for
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloca() : memref<f32>
// CHECK-NEXT: scf.for
// CHECK-NEXT:   scf.for
// CHECK-NEXT:     test.use
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

func.func @hoist_from_while() {
  scf.while() : () -> () {
    %0 = "test.make_condition"() : () -> i1
    scf.condition(%0)
  } do {
  ^bb0():
    %alloc = memref.alloc() : memref<f32>
    "test.use"(%alloc) : (memref<f32>) -> ()
    memref.dealloc %alloc : memref<f32>
    scf.yield
  }
  return
}

// CHECK-LABEL: @hoist_from_while
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloca() : memref<f32>
// CHECK-NEXT: scf.while
// CHECK-NEXT:   test.make_condition
// CHECK-NEXT:   scf.condition
// CHECK-NEXT: } do {
// CHECK-NEXT:   test.use
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }

// -----

func.func @double_buffer_for(%lb: index, %ub: index, %step: index) {
  %init = memref.alloc() : memref<f32>
  %0 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %init) -> (memref<f32>) {
    %alloc = memref.alloc() : memref<f32>
    "test.use"(%arg0, %alloc) : (memref<f32>, memref<f32>) -> ()
    memref.dealloc %arg0 : memref<f32>
    scf.yield %alloc : memref<f32>
  }
  memref.dealloc %0 : memref<f32>
  return
}

// CHECK-LABEL: @double_buffer_for
// CHECK-NEXT:  %[[ALLOC1:.*]] = memref.alloca
// CHECK-NEXT:  %[[ALLOC2:.*]] = memref.alloca
// CHECK-NEXT:  scf.for
// CHECK-SAME:      iter_args(%[[A:.*]] = %[[ALLOC1]], %[[B:.*]] = %[[ALLOC2]])
// CHECK-NEXT:    "test.use"(%[[A]], %[[B]])
// CHECK-NEXT:    scf.yield %[[B]], %[[A]]

// -----

func.func @double_buffer_while(%lb: index, %ub: index, %step: index) {
  %init = memref.alloc() : memref<f32>
  %0 = scf.while (%arg0 = %init) : (memref<f32>) -> (memref<f32>) {
    %0 = "test.make_condition"() : () -> i1
    scf.condition(%0) %arg0 : memref<f32>
  } do {
  ^bb0(%arg0: memref<f32>):
    %alloc = memref.alloc() : memref<f32>
    "test.use"(%arg0, %alloc) : (memref<f32>, memref<f32>) -> ()
    memref.dealloc %arg0 : memref<f32>
    scf.yield %alloc : memref<f32>
  }
  memref.dealloc %0 : memref<f32>
  return
}

// CHECK-LABEL: @double_buffer_while
// CHECK-NEXT:  %[[ALLOC0:.*]] = memref.alloca
// CHECK-NEXT:  %[[ALLOC1:.*]] = memref.alloca
// CHECK-NEXT:  scf.while (%[[A:.*]] = %[[ALLOC0]], %[[B:.*]] = %[[ALLOC1]])
// CHECK-NEXT:    make_condition
// CHECK-NEXT:    condition{{.*}} %[[A]], %[[B]]
// CHECK-NEXT:  } do {
// CHECK-NEXT:  ^bb0
// CHECK-NEXT:    "test.use"(%[[A]], %[[B]])
// CHECK-NEXT:    scf.yield %[[B]], %[[A]]
// CHECK-NEXT:  }

// -----

func.func @simplify_loop_dealloc() {
  %a = memref.alloc() : memref<f32>
  %a_owned = deallocation.own %a : memref<f32>
  %b = memref.alloc() : memref<f32>
  %b_owned = deallocation.own %b : memref<f32>
  %c = memref.alloc() : memref<f32>
  %c_owned = deallocation.own %c : memref<f32>
  %w:6 = scf.while (%arg0 = %a, %arg1 = %b, %arg2 = %c, %arg3 = %a_owned, %arg4 = %b_owned, %arg5 = %c_owned)
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
  memref.dealloc %w#0 : memref<f32>
  memref.dealloc %w#1 : memref<f32>
  memref.dealloc %w#2 : memref<f32>
  return
}

// CHECK-LABEL: @simplify_loop_dealloc
// CHECK: memref.alloca
// CHECK: memref.alloca
// CHECK: memref.alloca
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.dealloc

// -----

func.func @hoist_always_reallocated() {
  %a = memref.alloc() : memref<f32>
  %b = deallocation.own %a : memref<f32>
  %w:3 = scf.while(%arg0 = %a, %arg1 = %b)
      : (memref<f32>, !deallocation.ownership)
     -> (i32, memref<f32>, !deallocation.ownership) {
    %cond = "test.make_condition"() : () -> i1
    %v = "test.dummy"() : () -> i32
    memref.dealloc %arg0 : memref<f32>
    %0 = memref.alloc() : memref<f32>
    %1 = deallocation.own %0 : memref<f32>
    scf.condition (%cond) %v, %0, %1 : i32, memref<f32>, !deallocation.ownership
  } do {
  ^bb0(%_: i32, %arg0: memref<f32>, %arg1 : !deallocation.ownership):
    memref.dealloc %arg0 : memref<f32>
    %0 = memref.alloc() : memref<f32>
    %1 = deallocation.own %0 : memref<f32>
    scf.yield %0, %1 : memref<f32>, !deallocation.ownership
  }
  memref.dealloc %w#1 : memref<f32>
  return
}

// CHECK-LABEL: @hoist_always_reallocated
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: deallocation.null
// CHECK-NEXT: scf.while
// CHECK-NOT: memref.alloc

// -----

func.func @hoist_passthrough() {
  %a = memref.alloc() : memref<f32>
  %b = deallocation.own %a : memref<f32>
  %w:3 = scf.while(%arg0 = %a, %arg1 = %b)
      : (memref<f32>, !deallocation.ownership)
     -> (i32, memref<f32>, !deallocation.ownership) {
    %cond = "test.make_condition"() : () -> i1
    %v = "test.dummy"() : () -> i32
    memref.dealloc %arg0 : memref<f32>
    %0 = memref.alloc() : memref<f32>
    %1 = deallocation.own %0 : memref<f32>
    scf.condition (%cond) %v, %0, %1 : i32, memref<f32>, !deallocation.ownership
  } do {
  ^bb0(%_: i32, %arg0: memref<f32>, %arg1: !deallocation.ownership):
    scf.yield %arg0, %arg1 : memref<f32>, !deallocation.ownership
  }
  memref.dealloc %w#1 : memref<f32>
  return
}

// CHECK-LABEL: @hoist_passthrough
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: deallocation.null
// CHECK-NEXT: scf.while
// CHECK-NOT: memref.alloc

// -----

func.func @allocs_in_different_scopes_with_no_overlap() {
  %alloc0 = memref.alloc() : memref<4xi32>
  "test.use"(%alloc0) : (memref<4xi32>) -> ()
  memref.dealloc %alloc0 : memref<4xi32>
  scf.while() : () -> () {
    %cond = "test.make_condition"() : () -> i1
    scf.condition(%cond)
  } do {
    %alloc1 = memref.alloc() : memref<4xi32>
    "test.use"(%alloc1) : (memref<4xi32>) -> ()
    memref.dealloc %alloc1 : memref<4xi32>
    scf.yield
  }
  %alloc2 = memref.alloc() : memref<4xi32>
  "test.use"(%alloc2) : (memref<4xi32>) -> ()
  memref.dealloc %alloc2 : memref<4xi32>
  return
}

// CHECK-LABEL: @allocs_in_different_scopes_with_no_overlap
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: test.use
// CHECK-NEXT: while
// CHECK-NOT: memref.alloc

// -----

func.func @allocs_in_different_scopes_with_no_overlap_2() {
  %alloc0 = memref.alloc() : memref<4xi32>
  %first0 = "first_op"(%alloc0) : (memref<4xi32>) -> (i32)
  memref.dealloc %alloc0 : memref<4xi32>
  scf.while() : () -> () {
    %cond = "test.make_condition"() : () -> i1
    scf.condition(%cond)
  } do {
    %alloc1 = memref.alloc() : memref<4xi32>
    %first1 = "first_op"(%alloc1) : (memref<4xi32>) -> (i32)
    memref.dealloc %alloc1 : memref<4xi32>
    %alloc2 = memref.alloc() : memref<4xi32>
    %first2 = "first_op"(%alloc2) : (memref<4xi32>) -> (i32)
    memref.dealloc %alloc2 : memref<4xi32>
    scf.yield
  }
  %alloc3 = memref.alloc() : memref<4xi32>
  %first3 = "first_op"(%alloc3) : (memref<4xi32>) -> (i32)
  memref.dealloc %alloc3 : memref<4xi32>
  return
}

// CHECK-LABEL: allocs_in_different_scopes_with_no_overlap_2
// CHECK: memref.alloca
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.dealloc

// -----

func.func @empty_region() {
  %cond = "test.make_condition"() : () -> i1
  scf.if %cond {
    "test.dummy"() : () -> ()
  }
  return
}

// Regression test. Just make sure this doesn't crash.
// CHECK-LABEL: @empty_region

// -----

func.func @copy_to_out_param(
    %arg0: memref<i32> { deallocation.restrict = true }) {
  %foo = memref.alloc() : memref<i32>
  "some.op"(%foo) : (memref<i32>) -> ()
  memref.copy %foo, %arg0 : memref<i32> to memref<i32>
  memref.dealloc %foo : memref<i32>
  return
}

// CHECK-LABEL: @copy_to_out_param(
// CHECK-SAME: %[[ARG:.*]]: memref<i32>
// CHECK-NEXT: "some.op"(%[[ARG]])
// CHECK-NEXT: return

// -----

func.func @copy_to_out_param_no_restrict(
    %arg0: memref<i32> { deallocation.restrict = false }) {
  %foo = memref.alloc() : memref<i32>
  "some.op"(%foo) : (memref<i32>) -> ()
  memref.copy %foo, %arg0 : memref<i32> to memref<i32>
  memref.dealloc %foo : memref<i32>
  return
}

// CHECK-LABEL: @copy_to_out_param_no_restrict(
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: some.op
// CHECK-NEXT: memref.copy
// CHECK-NEXT: return

// -----

func.func @copy_to_out_param_and_change_param(
    %arg0: memref<2xindex> { deallocation.restrict = true }) {
  %foo = memref.alloc() : memref<2xindex>
  "some.op"(%foo) : (memref<2xindex>) -> ()
  memref.copy %foo, %arg0 : memref<2xindex> to memref<2xindex>
  memref.dealloc %foo : memref<2xindex>
  %c1 = arith.constant 1 : index
  memref.store %c1, %arg0[%c1] : memref<2xindex>
  return
}

// CHECK-LABEL: @copy_to_out_param_and_change_param(
// CHECK-SAME: %[[ARG:.*]]: memref<2xindex>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: "some.op"(%[[ARG]])
// CHECK: memref.store %[[C1]], %[[ARG]]
// CHECK-NEXT: return

// -----

func.func @copy_to_out_param_and_change_src(
    %arg0: memref<2xindex> { deallocation.restrict = true }) {
  %c1 = arith.constant 1 : index
  %foo = memref.alloc() : memref<2xindex>
  "some.op"(%foo) : (memref<2xindex>) -> ()
  memref.copy %foo, %arg0 : memref<2xindex> to memref<2xindex>
  memref.store %c1, %foo[%c1] : memref<2xindex>
  "other.op"(%foo) : (memref<2xindex>) -> ()
  memref.dealloc %foo : memref<2xindex>
  return
}

// CHECK-LABEL: @copy_to_out_param_and_change_src(
// CHECK-NEXT: arith.constant
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: some.op
// CHECK-NEXT: memref.copy
// CHECK-NEXT: memref.store
// CHECK-NEXT: other.op
// CHECK-NEXT: return

// -----

func.func @copy_to_out_param_and_change_src_and_copy(
    %arg0: memref<2xindex> { deallocation.restrict = true },
    %arg1: memref<2xindex> { deallocation.restrict = true }) {
  %c1 = arith.constant 1 : index
  %foo = memref.alloc() : memref<2xindex>
  "some.op"(%foo) : (memref<2xindex>) -> ()
  memref.copy %foo, %arg0 : memref<2xindex> to memref<2xindex>
  memref.store %c1, %foo[%c1] : memref<2xindex>
  "other.op"(%foo) : (memref<2xindex>) -> ()
  memref.copy %foo, %arg1 : memref<2xindex> to memref<2xindex>
  memref.dealloc %foo : memref<2xindex>
  return
}

// CHECK-LABEL: @copy_to_out_param_and_change_src_and_copy
// CHECK-SAME:    %[[ARG0:.*]]: memref<2xindex> {{{.*}}},
// CHECK-SAME:    %[[ARG1:.*]]: memref<2xindex>
// CHECK-NEXT: arith.constant
// CHECK-NEXT: "some.op"(%[[ARG1]])
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ARG0]]
// CHECK-NEXT: memref.store
// CHECK-NEXT: "other.op"(%[[ARG1]])
// CHECK-NEXT: return

// -----

func.func @copy_from_param_to_param(
    %arg0: memref<i32>, %arg1: memref<i32> { deallocation.restrict = true }) {
  memref.copy %arg0, %arg1 : memref<i32> to memref<i32>
  return
}

// CHECK-LABEL: @copy_from_param_to_param(
// CHECK-NEXT: memref.copy
// CHECK-NEXT: return