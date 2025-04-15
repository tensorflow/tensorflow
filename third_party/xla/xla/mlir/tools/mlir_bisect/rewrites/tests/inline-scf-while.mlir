// RUN: mlir-bisect %s --debug-strategy=InlineScfWhile | FileCheck %s

func.func @main() -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %alloc = memref.alloc() : memref<i64>
  memref.store %c0, %alloc[] : memref<i64>
  %ret = scf.while(%arg0 = %c0): (i64) -> (i64) {
    %cond = arith.cmpi slt, %arg0, %c4 : i64
    scf.condition(%cond) %arg0 : i64
  } do {
  ^bb0(%arg1: i64):
    %add = arith.addi %arg1, %c1 : i64
    scf.yield %add : i64
  }
  return %ret : i64
}

//     CHECK: func @main
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4
//     CHECK:   %[[RET:.*]]:2 = scf.execute_region
//     CHECK:     arith.cmpi slt, %[[C0]], %[[C4]]
//     CHECK:     yield {{.*}}, %[[C0]]
//     CHECK:   return %[[RET]]#1

//     CHECK: func @main
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1
//     CHECK:   %[[BEFORE0:.*]]:2 = scf.execute_region
//     CHECK:     arith.cmpi
//     CHECK:     yield {{.*}}, %[[C0]]
//     CHECK:   %[[AFTER:.*]] = scf.execute_region
//     CHECK:     %[[ADD:.*]] = arith.addi %[[BEFORE0]]#1, %[[C1]]
//     CHECK:     yield %[[ADD]]
//     CHECK:   %[[BEFORE1:.*]]:2 = scf.execute_region
//     CHECK:     arith.cmpi
//     CHECK:     yield {{.*}}, %[[AFTER]]
//     CHECK:   return %[[BEFORE1]]#1