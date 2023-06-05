// RUN: xla-cpu-opt %s -split-input-file -xla-legalize-i1-vector-transfers \
// RUN:   | FileCheck %s

func.func @transfer_read(%in: memref<8xi1>) -> vector<8xi1> {
  %pad = arith.constant true
  %c1 = arith.constant 1 : index
  %ret = vector.transfer_read %in[%c1], %pad : memref<8xi1>, vector<8xi1>
  return %ret : vector<8xi1>
}

// CHECK-LABEL: @transfer_read
//  CHECK-SAME:     %[[IN:.*]]: memref<8xi1>
//   CHECK-DAG:   %[[C1_I8:.*]] = arith.constant 1 : i8
//   CHECK-DAG:   %[[C0_V:.*]] = arith.constant dense<0> : vector<8xi8>
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//       CHECK:   %[[CAST:.*]] = xla_cpu.memref_element_cast %[[IN]]
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[CAST]][%[[C1]]],
//  CHECK-SAME:                    %[[C1_I8]]
//       CHECK:   %[[RET:.*]] = arith.cmpi ne, %[[READ]], %[[C0_V]]
//       CHECK:   return %[[RET]]

func.func @transfer_write(%in: vector<8xi1>, %out: memref<8xi1>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %in, %out[%c0] : vector<8xi1>, memref<8xi1>
  return
}

// CHECK-LABEL: @transfer_write
//  CHECK-SAME:     %[[IN:.*]]: vector<8xi1>
//  CHECK-SAME:     %[[OUT:.*]]: memref<8xi1>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[CAST_IN:.*]] = arith.extui %[[IN]] {{.*}} to vector<8xi8>
//   CHECK-DAG:   %[[CAST_OUT:.*]] = xla_cpu.memref_element_cast %[[OUT]]
//   CHECK-NOT:   vector.transfer_write {{.*}}%[[IN]]
//       CHECK:   vector.transfer_write %[[CAST_IN]], %[[CAST_OUT]][%[[C0]]]
