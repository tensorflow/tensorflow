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
