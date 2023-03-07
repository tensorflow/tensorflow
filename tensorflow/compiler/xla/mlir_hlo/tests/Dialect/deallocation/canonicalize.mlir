// RUN: mlir-hlo-opt %s -allow-unregistered-dialect -canonicalize | FileCheck %s

func.func @retain_is_dealloc() {
  %alloc = memref.alloc() : memref<2xf32>
  "test.use"(%alloc) : (memref<2xf32>) -> ()
  deallocation.retain() of (%alloc) : (memref<2xf32>) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc()
// CHECK: memref.dealloc %[[ALLOC]]

// -----

func.func @retain_is_noop(%arg: memref<*xf32>) -> memref<*xf32> {
  %ret = deallocation.retain(%arg) of(%arg) :
     (memref<*xf32>, memref<*xf32>) -> (memref<*xf32>)
  return %ret : memref<*xf32>
}

// CHECK-LABEL: @retain_is_noop
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>)
// CHECK-NEXT: return %[[ARG]]

// -----

func.func @retain_of_nothing(%arg: memref<2xf32>) -> memref<*xf32> {
  %ret = deallocation.retain(%arg) of() : (memref<2xf32>) -> (memref<*xf32>)
  return %ret : memref<*xf32>
}

// CHECK-LABEL: @retain_of_nothing
// CHECK-SAME: (%[[ARG:.*]]: memref<2xf32>
// CHECK-NEXT: %[[NULL:.*]] = deallocation.null : memref<*xf32>
// CHECK-NEXT: return %[[NULL]]

// -----

func.func @retain_is_dealloc_for(%x: memref<2xf32>, %lb: index, %ub: index, %step: index) {
  %for = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %x)
      -> (memref<2xf32>) {
    %alloc = memref.alloc() : memref<2xf32>
    scf.yield %alloc : memref<2xf32>
  }
  deallocation.retain() of(%for) : (memref<2xf32>) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc_for
// CHECK: %[[FOR:.*]] = scf.for
// CHECK: memref.dealloc %[[FOR]]

// -----

func.func @retain_is_dealloc_for_cast(%x: memref<*xf32>, %lb: index, %ub: index, %step: index) {
  %for = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %x)
      -> (memref<*xf32>) {
    %alloc = memref.alloc() : memref<2xf32>
    %cast = memref.cast %alloc : memref<2xf32> to memref<*xf32>
    scf.yield %cast : memref<*xf32>
  }
  deallocation.retain() of(%for) : (memref<*xf32>) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc_for_cast
// CHECK: %[[FOR:.*]] = scf.for
// CHECK: memref.dealloc %[[FOR]]

// -----

func.func @retain_is_not_dealloc_for(%x: memref<2xf32>, %lb: index, %ub: index, %step: index) {
  %for = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %x)
      -> (memref<2xf32>) {
    %alloc = memref.alloc() : memref<2xf32>
    %cast = "test.someop"(%alloc) : (memref<2xf32>) -> (memref<2xf32>)
    scf.yield %cast : memref<2xf32>
  }
  deallocation.retain() of(%for) : (memref<2xf32>) -> ()
  return
}

// CHECK-LABEL: @retain_is_not_dealloc_for
// CHECK: %[[FOR:.*]] = scf.for
// CHECK: deallocation.retain() of(%[[FOR]])

// -----

func.func @retain_is_dealloc_while() {
  %a = memref.alloc() : memref<2xf32>
  %while = scf.while (%arg0 = %a) : (memref<2xf32>) -> (memref<2xf32>) {
    %0 = "test.make_condition"() : () -> i1
    scf.condition(%0) %arg0 : memref<2xf32>
  } do {
  ^bb0(%arg0: memref<2xf32>):
    %b = memref.alloc() : memref<2xf32>
    scf.yield %b: memref<2xf32>
  }
  deallocation.retain() of (%while) : (memref<2xf32>) -> ()
  return
}

// CHECK-LABEL: @retain_is_dealloc_while
// CHECK: %[[WHILE:.*]] = scf.while
// CHECK: memref.dealloc %[[WHILE]]

func.func @retain_is_dealloc_while_permute() {
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
  "test.use"(%w#1) : (memref<f32>) -> ()
  deallocation.retain() of (%w#3, %w#4, %w#5) : (memref<f32>, memref<f32>, memref<f32>) -> ()
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

func.func @split_retain(%arg0: memref<*xi32>, %arg1: memref<*xf32>) {
  deallocation.retain() of(%arg0, %arg1) : (memref<*xi32>, memref<*xf32>) -> ()
  return
}

// CHECK-LABEL: @split_retain
// CHECK-DAG: deallocation.retain() of(%{{.*}}) : (memref<*xi32>) -> ()
// CHECK-DAG: deallocation.retain() of(%{{.*}}) : (memref<*xf32>) -> ()

func.func @retain_of_null(%arg0: memref<4xi32>, %arg1: memref<4xi32>,
                          %arg2: index, %arg3: index, %arg4: index) {
  %0 = deallocation.null : memref<*xi32>
  %1 = deallocation.null : memref<*xi32>
  %2:4 = scf.for %arg5 = %arg2 to %arg3 step %arg4
      iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %0, %arg9 = %1) ->
      (memref<4xi32>, memref<4xi32>, memref<*xi32>, memref<*xi32>) {
    "test.use"(%arg6, %arg7) : (memref<4xi32>, memref<4xi32>) -> ()
    %3 = deallocation.retain(%arg6) of(%arg8)
      : (memref<4xi32>, memref<*xi32>) -> memref<*xi32>
    %4 = deallocation.retain(%arg7) of(%arg9)
      : (memref<4xi32>, memref<*xi32>) -> memref<*xi32>
    scf.yield %arg7, %arg6, %4, %3
      : memref<4xi32>, memref<4xi32>, memref<*xi32>, memref<*xi32>
  }
  deallocation.retain() of(%2#2) : (memref<*xi32>) -> ()
  deallocation.retain() of(%2#3) : (memref<*xi32>) -> ()
  return
}

// CHECK-LABEL: @retain_of_null
// CHECK-NOT: deallocation.null
// CHECK-NOT: deallocation.retain()
