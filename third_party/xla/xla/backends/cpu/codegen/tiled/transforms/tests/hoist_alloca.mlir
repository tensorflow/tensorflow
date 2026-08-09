// RUN: fusion_compiler_opt --xtile-cpu-hoist-alloca %s | FileCheck %s

// CHECK-LABEL: func.func @hoist_from_for(
// CHECK-SAME:     %[[ARG0:[^:]+]]: index,
// CHECK-SAME:     %[[ARG1:[^:]+]]: index,
// CHECK-SAME:     %[[ARG2:[^:]+]]: index)
func.func @hoist_from_for(%arg0: index, %arg1: index, %arg2: index) {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() : memref<16xf32>
  // CHECK-NEXT: scf.for
  // CHECK-NOT: memref.alloca
  // CHECK: return
  scf.for %iv = %arg0 to %arg1 step %arg2 {
    %alloca = memref.alloca() : memref<16xf32>
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    memref.store %f0, %alloca[%c0] : memref<16xf32>
  }
  return
}

// CHECK-LABEL: func.func @hoist_from_if(
// CHECK-SAME:     %[[COND:[^:]+]]: i1)
func.func @hoist_from_if(%cond: i1) {
  // CHECK-NEXT: %[[ALLOCA:.*]] = memref.alloca() : memref<16xf32>
  // CHECK-NEXT: scf.if
  // CHECK-NOT: memref.alloca
  // CHECK: return
  scf.if %cond {
    %alloca = memref.alloca() : memref<16xf32>
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    memref.store %f0, %alloca[%c0] : memref<16xf32>
  }
  return
}
