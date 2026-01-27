// RUN: fusion_compiler_opt %s \
// RUN:   -xtile-cpu-unpack-sub-byte-vector-write -split-input-file \
// RUN:   | FileCheck %s

func.func @unpacks_1d_read(%arg0 : memref<8xi1>) -> vector<2xi1> {
  // CHECK-DAG: %[[MASK:.*]] = ub.poison
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3
  // CHECK: %[[READ0:.*]] = vector.transfer_read %arg0[%[[C2]]], %[[MASK]]
  // CHECK: %[[EXTRACT0:.*]] = vector.extract %[[READ0]][0]
  // CHECK: %[[READ1:.*]] = vector.transfer_read %arg0[%[[C3]]], %[[MASK]]
  // CHECK: %[[EXTRACT1:.*]] = vector.extract %[[READ1]][0]
  // CHECK: %[[RESULT:.*]] = vector.from_elements %[[EXTRACT0]], %[[EXTRACT1]]
  %mask = ub.poison : i1
  %c2 = arith.constant 2 : index
  %result = vector.transfer_read %arg0[%c2], %mask : memref<8xi1>, vector<2xi1>
  // CHECK: return %[[RESULT]]
  return %result : vector<2xi1>
}

//-----

func.func @ignores_trivial_case_read(%arg0 : memref<8xi1>) -> vector<1xi1> {
  // CHECK: %[[MASK:.*]] = ub.poison
  %mask = ub.poison : i1
  // CHECK-NEXT: %[[C2:.*]] = arith.constant 2
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: %[[RESULT:.*]] = vector.transfer_read %arg0[%[[C2]]], %[[MASK]]
  %result = vector.transfer_read %arg0[%c2], %mask : memref<8xi1>, vector<1xi1>
  // CHECK-NEXT: return %[[RESULT]]
  return %result : vector<1xi1>
}

//-----

func.func @ignores_non_sub_byte_type_read(%arg0 : memref<8xi8>) -> vector<2xi8> {
  // CHECK: %[[MASK:.*]] = ub.poison
  %mask = ub.poison : i8
  // CHECK-NEXT: %[[C2:.*]] = arith.constant 2
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: %[[RESULT:.*]] = vector.transfer_read %arg0[%[[C2]]], %[[MASK]]
  %result = vector.transfer_read %arg0[%c2], %mask : memref<8xi8>, vector<2xi8>
  // CHECK-NEXT: return %[[RESULT]]
  return %result : vector<2xi8>
}

//-----

func.func @unpacks_1d_write(%arg0 : vector<2xi1>, %arg1 : memref<8xi1>) {
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3
  // CHECK: %[[ELEMENT0:.*]] = vector.extract %arg0[0]
  // CHECK: %[[VECTOR0:.*]] = vector.from_elements %[[ELEMENT0]]
  // CHECK: vector.transfer_write %[[VECTOR0]], %arg1[%[[C2]]]
  // CHECK: %[[ELEMENT1:.*]] = vector.extract %arg0[1]
  // CHECK: %[[VECTOR1:.*]] = vector.from_elements %[[ELEMENT1]]
  // CHECK: vector.transfer_write %[[VECTOR1]], %arg1[%[[C3]]]
  %c2 = arith.constant 2 : index
  vector.transfer_write %arg0, %arg1[%c2] : vector<2xi1>, memref<8xi1>
  // CHECK-NEXT: return
  return
}

//-----

func.func @ignores_trivial_case_write(%arg0 : vector<1xi1>, %arg1 : memref<8xi1>) {
  // CHECK: %[[C2:.*]] = arith.constant 2
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: vector.transfer_write %arg0, %arg1[%[[C2]]]
  vector.transfer_write %arg0, %arg1[%c2] : vector<1xi1>, memref<8xi1>
  // CHECK-NEXT: return
  return
}

//-----

func.func @ignores_non_sub_byte_type_write(%arg0 : vector<2xi8>, %arg1 : memref<8xi8>) {
  // CHECK: %[[C2:.*]] = arith.constant 2
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: vector.transfer_write %arg0, %arg1[%[[C2]]]
  vector.transfer_write %arg0, %arg1[%c2] : vector<2xi8>, memref<8xi8>
  // CHECK-NEXT: return
  return
}
