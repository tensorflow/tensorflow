// RUN: tf-opt -lhlo-legalize-to-linalg %s -o - | FileCheck %s

// CHECK: #map0 = (d0, d1) -> (d0, d1)
// CHECK-LABEL: func @element_wise
func @element_wise(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>, %arg3: memref<2x2xf32>) {
  "xla_lhlo.add"(%arg1, %arg2, %arg3) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: f32, %[[D1:.*]]: f32, %[[D2:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = addf %[[D0]], %[[D1]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-LABEL: func @minf
func @minf(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
  "xla_lhlo.min"(%arg0, %arg1, %arg2) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: f32, %[[D1:.*]]: f32, %[[D2:.*]]: f32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpf "olt", %[[D0]], %[[D1]] : f32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[D0]], %[[D1]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32


// CHECK-LABEL: func @maxi
func @maxi(%arg0: memref<2x2xi32>, %arg1: memref<2x2xi32>, %arg2: memref<2x2xi32>) {
  "xla_lhlo.max"(%arg0, %arg1, %arg2) : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: i32, %[[D1:.*]]: i32, %[[D2:.*]]: i32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpi "sgt", %[[D0]], %[[D1]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[D0]], %[[D1]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-LABEL: func @and
func @and(%arg0: memref<2x2xi32>, %arg1: memref<2x2xi32>, %arg2: memref<2x2xi32>) {
  "xla_lhlo.and"(%arg0, %arg1, %arg2) : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: i32, %[[D1:.*]]: i32, %[[D2:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = and %[[D0]], %[[D1]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-LABEL: func @exp
func @exp(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  "xla_lhlo.exp"(%arg0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: f32, %[[D1:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = exp %[[D0]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-LABEL: func @float_cmp
func @float_cmp(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xi1>) {
  "xla_lhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "EQ"}: (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_IN:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpf "oeq", %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-LABEL: func @int_cmp
func @int_cmp(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>, %result: memref<2x2xi1>) {
  "xla_lhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "LT"}: (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_IN:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpi "slt", %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
