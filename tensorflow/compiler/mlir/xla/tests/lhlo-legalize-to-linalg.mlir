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
func @minf(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>, %arg3: memref<2x2xf32>) {
  "xla_lhlo.min"(%arg1, %arg2, %arg3) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: f32, %[[D1:.*]]: f32, %[[D2:.*]]: f32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpf "olt", %[[D0]], %[[D1]] : f32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %arg4, %arg5 : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-LABEL: func @maxi
func @maxi(%arg0: memref<2x2xi32>, %arg1: memref<2x2xi32>, %arg2: memref<2x2xi32>, %arg3: memref<2x2xi32>) {
  "xla_lhlo.max"(%arg1, %arg2, %arg3) : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: i32, %[[D1:.*]]: i32, %[[D2:.*]]: i32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpi "sgt", %[[D0]], %[[D1]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %arg4, %arg5 : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-LABEL: func @and
func @and(%arg0: memref<2x2xi32>, %arg1: memref<2x2xi32>, %arg2: memref<2x2xi32>, %arg3: memref<2x2xi32>) {
  "xla_lhlo.and"(%arg1, %arg2, %arg3) : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[D0:.*]]: i32, %[[D1:.*]]: i32, %[[D2:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = and %[[D0]], %[[D1]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32
