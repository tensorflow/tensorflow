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

func.func @simplify_loop_dealloc() {
  %a = memref.alloc() : memref<f32>
  %b = memref.alloc() : memref<f32>
  %c = memref.alloc() : memref<f32>
  %null = deallocation.null : memref<f32>
  %w:6 = scf.while (%arg0 = %a, %arg1 = %b, %arg2 = %c, %arg3 = %a, %arg4 = %b, %arg5 = %c)
    : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) ->
      (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) {
    %cond = "test.make_condition"() : () -> i1
    scf.condition(%cond) %arg2, %arg1, %arg0, %arg5, %arg4, %arg3
      : memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>
  } do {
  ^bb0(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>,
        %arg3: memref<f32>, %arg4: memref<f32>, %arg5: memref<f32>):
    scf.yield %arg1, %arg0, %arg2, %arg4, %arg3, %arg5
      : memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>
  }
  memref.dealloc %w#5 : memref<f32>
  memref.dealloc %w#4 : memref<f32>
  memref.dealloc %w#3 : memref<f32>
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
  %b = memref.cast %a : memref<f32> to memref<*xf32>
  %w:3 = scf.while(%arg0 = %a, %arg1 = %b)
    : (memref<f32>, memref<*xf32>) -> (i32, memref<f32>, memref<*xf32>) {
    %cond = "test.make_condition"() : () -> i1
    %v = "test.dummy"() : () -> i32
    memref.dealloc %arg1 : memref<*xf32>
    %0 = memref.alloc() : memref<f32>
    %1 = memref.cast %0 : memref<f32> to memref<*xf32>
    scf.condition (%cond) %v, %0, %1 : i32, memref<f32>, memref<*xf32>
  } do {
  ^bb0(%_: i32, %arg0: memref<f32>, %arg1 : memref<*xf32>):
    memref.dealloc %arg1 : memref<*xf32>
    %0 = memref.alloc() : memref<f32>
    %1 = memref.cast %0 : memref<f32> to memref<*xf32>
    scf.yield %0, %1 : memref<f32>, memref<*xf32>
  }
  memref.dealloc %w#2 : memref<*xf32>
  return
}

// CHECK-LABEL: @hoist_always_reallocated
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: scf.while
// CHECK-NOT: memref.alloc

// -----

func.func @hoist_passthrough() {
  %a = memref.alloc() : memref<f32>
  %b = memref.cast %a : memref<f32> to memref<*xf32>
  %w:3 = scf.while(%arg0 = %a, %arg1 = %b)
    : (memref<f32>, memref<*xf32>) -> (i32, memref<f32>, memref<*xf32>) {
    %cond = "test.make_condition"() : () -> i1
    %v = "test.dummy"() : () -> i32
    memref.dealloc %arg1 : memref<*xf32>
    %0 = memref.alloc() : memref<f32>
    %1 = memref.cast %0 : memref<f32> to memref<*xf32>
    scf.condition (%cond) %v, %0, %1 : i32, memref<f32>, memref<*xf32>
  } do {
  ^bb0(%_: i32, %arg0: memref<f32>, %arg1: memref<*xf32>):
    scf.yield %arg0, %arg1 : memref<f32>, memref<*xf32>
  }
  memref.dealloc %w#2 : memref<*xf32>
  return
}

// CHECK-LABEL: @hoist_passthrough
// CHECK-NEXT: memref.alloca
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

func.func @elide_for_ownership() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc_0 = memref.alloc() : memref<1xi64>
  %cast_0 = memref.cast %alloc_0 : memref<1xi64> to memref<*xi64>
  %0:2 = scf.for %arg4 = %c0 to %c1 step %c1 iter_args(%arg0 = %alloc_0, %arg1 = %cast_0) -> (memref<1xi64>, memref<*xi64>) {
    memref.dealloc %arg1 : memref<*xi64>
    %alloc_1 = memref.alloc() : memref<1xi64>
    %cast_1 = memref.cast %alloc_1 : memref<1xi64> to memref<*xi64>
    scf.yield %alloc_1, %cast_1 : memref<1xi64>, memref<*xi64>
  }
  memref.dealloc %0#1 : memref<*xi64>
  return
}

// CHECK-LABEL: @elide_for_ownership
// CHECK-NEXT: return

// -----

func.func @elide_while_ownership() {
  %alloc_1 = memref.alloc() : memref<5xi32>
  %alloc_2 = memref.alloc() : memref<5xi32>
  %alloc_3 = memref.alloc() : memref<5xi32>
  %cast_1 = memref.cast %alloc_2 : memref<5xi32> to memref<*xi32>
  %cast_2 = memref.cast %alloc_3 : memref<5xi32> to memref<*xi32>
  %6:4 = scf.while (%arg0 = %alloc_2, %arg1 = %alloc_3, %arg2 = %cast_1, %arg3 = %cast_2)
      : (memref<5xi32>, memref<5xi32>, memref<*xi32>, memref<*xi32>) ->
        (memref<5xi32>, memref<5xi32>, memref<*xi32>, memref<*xi32>) {
    %alloc_4 = memref.alloc() : memref<5xi32>
    memref.dealloc %arg2 : memref<*xi32>
    %alloc_5 = memref.alloc() : memref<5xi32>
    deallocation.retain() of(%arg3) : (memref<*xi32>) -> ()
    %cast_3 = memref.cast %alloc_4 : memref<5xi32> to memref<*xi32>
    %cast_4 = memref.cast %alloc_5 : memref<5xi32> to memref<*xi32>
    %cond = "test.make_condition"() : () -> (i1)
    scf.condition(%cond) %alloc_4, %alloc_5, %cast_3, %cast_4
      : memref<5xi32>, memref<5xi32>, memref<*xi32>, memref<*xi32>
  } do {
  ^bb0(%arg0: memref<5xi32>, %arg1: memref<5xi32>, %arg2: memref<*xi32>, %arg3: memref<*xi32>):
    memref.dealloc %arg2 : memref<*xi32>
    %alloc_55 = memref.alloc() : memref<5xi32>
    memref.dealloc %arg3 : memref<*xi32>
    %null = deallocation.null : memref<*xi32>
    %cast_3 = memref.cast %alloc_55 : memref<5xi32> to memref<*xi32>
    scf.yield %alloc_55, %alloc_1, %cast_3, %null
      : memref<5xi32>, memref<5xi32>, memref<*xi32>, memref<*xi32>
  }
  memref.dealloc %alloc_1 : memref<5xi32>
  memref.dealloc %6#2 : memref<*xi32>
  memref.dealloc %6#3 : memref<*xi32>
  return
}

// CHECK-LABEL: @elide_while_ownership
// CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca()
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: %[[CAST:.*]] = memref.cast %[[ALLOC1]]
// CHECK-NEXT: %[[WHILE:.*]]:2 = scf.while
// CHECK-SAME:     %[[ARG0:.*]] = %[[ALLOC0]],
// CHECK-SAME:     %[[ARG1:.*]] = %[[ALLOC1]],
// CHECK-SAME:     %[[ARG2:.*]] = %[[CAST]]
// TODO(jreiffers): There's no double buffering for the before region yet.
// CHECK-NEXT:   %[[ALLOC2:.*]] = memref.alloc()
// CHECK-NEXT:   deallocation.retain() of(%[[ARG2]])
// CHECK-NEXT:   test.make_condition
// CHECK-NEXT:   scf.condition
// CHECK-SAME:     %[[ALLOC2]], %[[ARG0]]
// CHECK-NEXT: } do {
// CHECK-NEXT:   %[[ARG0:.*]]: memref<5xi32>, %[[ARG1:.*]]: memref<5xi32>
// CHECK-NEXT:   dealloc %[[ARG1]]
// CHECK-NEXT:   %[[NULL:.*]] = deallocation.null
// CHECK-NEXT:   scf.yield %[[ARG0]], %[[ALLOCA]], %[[NULL]]
// CHECK-NEXT: }
// CHECK-NEXT: dealloc %[[WHILE]]#0
// CHECK-NEXT: dealloc %[[WHILE]]#1
// CHECK-NEXT: return

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
