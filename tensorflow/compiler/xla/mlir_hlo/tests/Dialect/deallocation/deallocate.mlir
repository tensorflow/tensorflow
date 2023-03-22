// RUN: mlir-hlo-opt %s -allow-unregistered-dialect -hlo-deallocate | FileCheck %s
// RUN: mlir-hlo-opt %s -allow-unregistered-dialect -hlo-deallocate -canonicalize | FileCheck %s --check-prefix=CHECK-CANON

func.func @loop_nested_alloc(
    %lb: index, %ub: index, %step: index,
    %buf: memref<2xf32>, %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = scf.for %i2 = %lb to %ub step %step
      iter_args(%iterBuf2 = %iterBuf) -> memref<2xf32> {
      %3 = memref.alloc() : memref<2xf32>
      %4 = arith.cmpi eq, %i, %ub : index
      %5 = scf.if %4 -> (memref<2xf32>) {
        %6 = memref.alloc() : memref<2xf32>
        scf.yield %6 : memref<2xf32>
      } else {
        scf.yield %iterBuf2 : memref<2xf32>
      }
      scf.yield %5 : memref<2xf32>
    }
    scf.yield %2 : memref<2xf32>
  }
  memref.copy %1, %res : memref<2xf32> to memref<2xf32>
  return
}

// CHECK-LABEL: func @loop_nested_alloc
// CHECK-SAME:     %[[ARG3:[a-z0-9]*]]: memref<2xf32>, %[[OUT:.*]]: memref<2xf32>)
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2xf32>
// CHECK: deallocation.retain() of(%[[ALLOC]])
// CHECK: %[[ARG3_UNOWNED:.*]] = deallocation.null : memref<*xf32>
// CHECK: %[[FOR1:.*]]:2 = scf.for {{.*}}iter_args(%[[A:.*]] = %[[ARG3]], %[[A_OWNERSHIP:.*]] = %[[ARG3_UNOWNED]])
// CHECK:   %[[A_UNOWNED:.*]] = deallocation.null : memref<*xf32>
// CHECK:   %[[FOR2:.*]]:2 = scf.for {{.*}} iter_args(%[[B:.*]] = %[[A]], %[[B_OWNERSHIP:.*]] = %[[A_UNOWNED]])
// CHECK:     %[[ALLOC2:.*]] = memref.alloc() : memref<2xf32>
// CHECK:     deallocation.retain() of(%[[ALLOC2]])
// CHECK:     %[[IF:.*]]:2 = scf.if
// CHECK:       %[[ALLOC3:.*]] = memref.alloc() : memref<2xf32>
// CHECK:       %[[ALLOC3_RETAINED:.*]] = memref.cast %[[ALLOC3]] : memref<2xf32> to memref<*xf32>
// CHECK:       scf.yield %[[ALLOC3]], %[[ALLOC3_RETAINED]]
// CHECK:     } else {
// CHECK:       %[[NULL:.*]] = deallocation.retain(%[[B]]) of()
// CHECK:       scf.yield %[[B]], %[[NULL]]
// CHECK:     }
// CHECK:     %[[RETAINED_IF:.*]] = deallocation.retain(%[[IF]]#0) of(%[[B_OWNERSHIP]], %[[IF]]#1)
// CHECK:     scf.yield %[[IF]]#0, %[[RETAINED_IF]]
// CHECK:   }
// CHECK:   %[[RETAINED_FOR2:.*]] = deallocation.retain(%[[FOR2]]#0) of(%[[A_OWNERSHIP]], %[[FOR2]]#1)
// CHECK:   scf.yield %[[FOR2]]#0, %[[RETAINED_FOR2]]
// CHECK: }
// CHECK: memref.copy %[[FOR1]]#0, %[[OUT]]
// CHECK: deallocation.retain() of(%[[FOR1]]#1)
// CHECK: return

// -----

func.func @nested_if() -> (memref<2xf32>, memref<2xf32>) {
  %alloc_0 = memref.alloc() : memref<2xf32>
  %alloc_1 = memref.alloc() : memref<2xf32>
  %a = "test.condition"() : () -> i1
  %0 = scf.if %a -> (memref<2xf32>) {
    %2 = memref.alloc() : memref<2xf32>
    scf.yield %2 : memref<2xf32>
  } else {
    %b = "test.condition"() : () -> i1
    %3 = scf.if %b -> (memref<2xf32>) {
      scf.yield %alloc_0 : memref<2xf32>
    } else {
      scf.yield %alloc_1 : memref<2xf32>
    }
    scf.yield %3 : memref<2xf32>
  }
  return %alloc_0, %0 : memref<2xf32>, memref<2xf32>
}

// CHECK-LABEL: func @nested_if
// CHECK:      %[[ALLOC0:.*]] = memref.alloc()
// CHECK:      %[[ALLOC1:.*]] = memref.alloc()
// CHECK:      %[[IF1:.*]]:2 = scf.if
// CHECK-NEXT:   %[[ALLOC2:.*]] = memref.alloc()
// CHECK-NEXT:   %[[ALLOC2_RETAINED:.*]] = memref.cast %[[ALLOC2]] : memref<2xf32> to memref<*xf32>
// CHECK-NEXT:   scf.yield %[[ALLOC2]], %[[ALLOC2_RETAINED]]
// CHECK-NEXT: } else {
// CHECK:        %[[IF2:.*]]:2 = scf.if
// CHECK-NEXT:     %[[NULL:.*]] = deallocation.retain(%[[ALLOC0]]) of()
// CHECK-NEXT:     scf.yield %[[ALLOC0]], %[[NULL]]
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[NULL:.*]] = deallocation.retain(%[[ALLOC1]]) of()
// CHECK-NEXT:     scf.yield %[[ALLOC1]], %[[NULL]]
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[IF2_RETAINED:.*]] = deallocation.retain(%[[IF2]]#0) of(%[[IF2]]#1)
// CHECK-NEXT:   scf.yield %[[IF2]]#0, %[[IF2_RETAINED]]
// CHECK-NEXT: }
// CHECK-NEXT: deallocation.retain(%[[ALLOC0]], %[[IF1]]#0) of(%[[ALLOC0]], %[[ALLOC1]], %[[IF1]]#1)
// CHECK-NEXT: return %[[ALLOC0]], %[[IF1]]#0 : memref<2xf32>, memref<2xf32>

// -----

func.func @while(%arg0: index) -> (memref<?xf32>, memref<?xf32>, memref<?xf32>) {
  %a = memref.alloc(%arg0) : memref<?xf32>
  %w:3 = scf.while (%arg1 = %a, %arg2 = %a, %arg3 = %a) : (memref<?xf32>, memref<?xf32>, memref<?xf32>)
      -> (memref<?xf32>, memref<?xf32>, memref<?xf32>) {
    %0 = "test.make_condition"() : () -> i1
    scf.condition(%0) %arg1, %arg2, %arg3 : memref<?xf32>, memref<?xf32>, memref<?xf32>
  } do {
  ^bb0(%arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>):
    %b = memref.alloc(%arg0) : memref<?xf32>
    %q = memref.alloc(%arg0) : memref<?xf32>
    scf.yield %q, %b, %arg2: memref<?xf32>, memref<?xf32>, memref<?xf32>
  }
  return %w#0, %w#1, %w#2 : memref<?xf32>, memref<?xf32>, memref<?xf32>
}

// CHECK-LABEL: func @while(
// CHECK-SAME:      %[[ARG0:.*]]:
// CHECK-NEXT:    %[[ALLOC:.*]] = memref.alloc(%arg0) : memref<?xf32>
// CHECK-NEXT:    %[[ALLOC_OWNED:.*]] = memref.cast %[[ALLOC]]
// CHECK-NEXT:    %[[NULL1:.*]] = deallocation.null
// CHECK-NEXT:    %[[NULL2:.*]] = deallocation.null
// CHECK-NEXT:    %[[WHILE:.*]]:6 = scf.while (%[[A:[a-z0-9]*]] = %[[ALLOC]], %[[B:[a-z0-9]*]] = %[[ALLOC]], %[[C:[a-z0-9]*]] = %[[ALLOC]],
// CHECK-SAME:       %[[A_OWNERSHIP:.*]] = %[[ALLOC_OWNED]], %[[B_OWNERSHIP:.*]] = %[[NULL1]], %[[C_OWNERSHIP:.*]] = %[[NULL2]])
// CHECK:            %[[A_RETAINED:.*]] = deallocation.retain(%[[A]]) of(%[[A_OWNERSHIP]])
// CHECK:            %[[B_RETAINED:.*]] = deallocation.retain(%[[B]]) of(%[[B_OWNERSHIP]])
// CHECK:            %[[C_RETAINED:.*]] = deallocation.retain(%[[C]]) of(%[[C_OWNERSHIP]])
// CHECK:            scf.condition{{.*}} %[[A]], %[[B]], %[[C]], %[[A_RETAINED]], %[[B_RETAINED]], %[[C_RETAINED]]
// CHECK:         } do {
// CHECK:           deallocation.retain() of(%[[C_OWNERSHIP]])
// CHECK:           deallocation.retain() of(%[[A_OWNERSHIP]])
// CHECK:           %[[ALLOC1:.*]] = memref.alloc(%[[ARG0]])
// CHECK:           %[[ALLOC2:.*]] = memref.alloc(%[[ARG0]])
// CHECK:           %[[B_RETAINED:.*]] = deallocation.retain(%[[B]]) of(%[[B_OWNERSHIP]])
// CHECK:           %[[ALLOC1_RETAINED:.*]] = memref.cast %[[ALLOC1]]
// CHECK:           %[[ALLOC2_RETAINED:.*]] = memref.cast %[[ALLOC2]]
// CHECK:           scf.yield %[[ALLOC2]], %[[ALLOC1]], %[[B]], %[[ALLOC2_RETAINED]], %[[ALLOC1_RETAINED]], %[[B_RETAINED]]
// CHECK:         }
// CHECK:         %[[RESULTS_RETAINED:.*]] = deallocation.retain(%[[WHILE]]#0, %[[WHILE]]#1, %[[WHILE]]#2)
// CHECK-SAME:      of(%[[WHILE]]#3, %[[WHILE]]#4, %[[WHILE]]#5)
// CHECK:         return %[[WHILE]]#0, %[[WHILE]]#1, %[[WHILE]]#2

// -----

func.func @if_without_else() {
  %cond = "test.make_condition"() : () -> i1
  scf.if %cond {
    %x = memref.alloc() : memref<2xf32>
    "test.use"(%x) : (memref<2xf32>) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: @if_without_else
// CHECK: scf.if
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc
// CHECK-NEXT: test.use
// CHECK-NEXT: deallocation.retain() of(%[[ALLOC]])

// CHECK-CANON-LABEL: @if_without_else
// CHECK-CANON:       scf.if
// CHECK-CANON-NEXT:    memref.alloc
// CHECK-CANON-NEXT:    test.use
// CHECK-CANON-NEXT:    memref.dealloc

// -----

func.func @yield_same_alloc_twice() {
  %alloc = memref.alloc() : memref<f32>
  scf.while (%a = %alloc, %b = %alloc) : (memref<f32>, memref<f32>) -> () {
    %cond = "test.make_condition"() : () -> i1
    scf.condition(%cond)
  } do {
  ^bb0():
    scf.yield %alloc, %alloc : memref<f32>, memref<f32>
  }
  return
}

// CHECK-LABEL: @yield_same_alloc_twice
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc
// CHECK-NEXT: %[[NULL1:.*]] = deallocation.null
// CHECK-NEXT: %[[NULL2:.*]] = deallocation.null
// CHECK: scf.while
// CHECK-SAME: %[[ALLOC]]
// CHECK-SAME: %[[ALLOC]]
// CHECK-SAME: %[[NULL1]]
// CHECK-SAME: %[[NULL2]]
// CHECK: } do {
// CHECK-NEXT: %[[RETAIN:.*]]:2 = deallocation.retain(%[[ALLOC]], %[[ALLOC]]) of()
// CHECK-NEXT: %[[NULL:.*]] = deallocation.null
// CHECK-NEXT: scf.yield %[[ALLOC]], %[[ALLOC]], %[[RETAIN]]#1, %[[NULL]]

// -----

func.func @yield_derived(%lb: index, %ub: index, %step: index) {
  %0 = memref.alloc() : memref<2xi32>
  %1 = scf.for %i2 = %lb to %ub step %step iter_args(%arg0 = %0) -> memref<2xi32> {
    %2 = memref.alloc() : memref<2xi32>
    %3 = "test.someop"(%2) : (memref<2xi32>) -> memref<1xi32>
    %4 = "test.someop"(%3) : (memref<1xi32>) -> memref<2xi32>
    scf.yield %4 : memref<2xi32>
  }
  "test.use"(%1) : (memref<2xi32>) -> ()
  return
}

// CHECK-LABEL: @yield_derived
// CHECK-NEXT:  memref.alloc
// CHECK-NEXT:  memref.cast
// CHECK-NEXT:  scf.for
// CHECK-NEXT:    deallocation.retain()
// CHECK-NEXT:    %[[ALLOC:.*]] = memref.alloc
// CHECK-NEXT:    "test.someop"
// CHECK-NEXT:    %[[RESULT:.*]] = "test.someop"
// CHECK-NEXT:    %[[OWNED:.*]] = memref.cast %[[ALLOC]]
// CHECK-NEXT:    scf.yield %[[RESULT]], %[[OWNED]]
// CHECK-NEXT:  }
// CHECK-NEXT:  test.use
// CHECK-NEXT:  retain

// CHECK-CANON-LABEL: @yield_derived
// CHECK-CANON:       test.use
// CHECK-CANON-NEXT:  dealloc

func.func @unknown_op() {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c1, %c8) {
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
    "test.use"(%alloc_14) : (memref<512x512xf32>) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: @unknown_op
// CHECK: scf.parallel
// CHECK-NEXT: alloc
// CHECK-NEXT: test.use
// CHECK-NEXT: dealloc
