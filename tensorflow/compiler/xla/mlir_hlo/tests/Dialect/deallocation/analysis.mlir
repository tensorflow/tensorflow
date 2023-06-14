// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect \
// RUN:     --hlo-deallocation-annotation | \
// RUN: FileCheck %s

func.func @loop_nested_alloc(
    %lb: index, %ub: index, %step: index,
    %buf: memref<2xf32>, %res: memref<2xf32>) {
  // CHECK-LABEL: func.func @loop_nested_alloc
  // CHECK-SAME:    (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index,
  // CHECK-SAME:    %[[BUF:.*]]: memref<2xf32>, %[[RES:.*]]: memref<2xf32>)
  // CHECK-SAME:    attributes {deallocation.region_args_backing_memory = {{\[\[}}
  // CHECK-SAME:      "", "", "", "%[[BUF]], %[[RES]]", "%[[BUF]], %[[RES]]"]]} {
  %0 = memref.alloc() : memref<2xf32>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc()
  // CHECK-SAME:  {deallocation.result_backing_memory = ["%[[ALLOC1]]"]} : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
      iter_args(%iterBuf = %buf) -> memref<2xf32> {
    // CHECK: %[[FOR1:.*]] = scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]]
    // CHECK-SAME: iter_args(%[[ITER_BUF:.*]] = %[[BUF]]) -> (memref<2xf32>)
    %2 = scf.for %j = %lb to %ub step %step
        iter_args(%iterBuf2 = %iterBuf) -> memref<2xf32> {
      // CHECK: %[[FOR2:.*]] = scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]]
      // CHECK-SAME: iter_args(%[[ITER_BUF2:.*]] = %[[ITER_BUF]]) -> (memref<2xf32>)
      %3 = memref.alloc() : memref<2xf32>
      // CHECK: %[[ALLOC2:.*]] = memref.alloc()
      %4 = arith.cmpi eq, %i, %ub : index
      // CHECK: arith.cmpi
      %5 = scf.if %4 -> (memref<2xf32>) {
        // CHECK: %[[IF:.*]] = scf.if
        %6 = memref.alloc() : memref<2xf32>
        // CHECK: %[[ALLOC3:.*]] = memref.alloc()
        scf.yield %6 : memref<2xf32>
        // CHECK: scf.yield %[[ALLOC3]]
      } else {
        scf.yield %iterBuf2 : memref<2xf32>
        // CHECK: scf.yield %[[ITER_BUF2]]
      }
      scf.yield %5 : memref<2xf32>
      // CHECK: scf.yield %[[IF]]
    }
    scf.yield %2 : memref<2xf32>
    // CHECK: scf.yield %[[FOR2]]
  }
  // CHECK: } {deallocation.region_args_backing_memory = {{\[\[}}"", "%[[ITER_BUF]], %[[ITER_BUF2]], %[[BUF]], %[[RES]], %[[ALLOC3]]"]],
  // CHECK-SAME: deallocation.result_backing_memory = ["%[[ITER_BUF]], %[[ITER_BUF2]], %[[BUF]], %[[RES]], %[[ALLOC3]]"]}
  memref.copy %1, %res : memref<2xf32> to memref<2xf32>
  return
}

// -----

func.func @arith_select() -> (memref<i32>, memref<i32>) {
  %cond = "test.make_condition"() : () -> (i1)
  %a = memref.alloc() : memref<i32>
  %b = memref.alloc() : memref<i32>
  %c = arith.select %cond, %a, %b : memref<i32>
  return %a, %c : memref<i32>, memref<i32>
}

// CHECK-LABEL: @arith_select
// CHECK: %[[COND:.*]] = "test.make_condition"
// CHECK: %[[A:.*]] = memref.alloc
// CHECK: %[[B:.*]] = memref.alloc
// CHECK: %[[C:.*]] = arith.select %[[COND]], %[[A]], %[[B]]
// CHECK-SAME: result_backing_memory = ["%[[A]], %[[B]]"]
