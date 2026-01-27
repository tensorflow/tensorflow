// RUN: emitters_opt %s -legalize-unsigned-integers-as-signless -split-input-file -verify-diagnostics

func.func @unsigned_func(%arg0: ui32, %arg1: memref<10xui32>) -> ui32 {
  %c0 = arith.constant 0 : i32
  %c0_as_ui32 = builtin.unrealized_conversion_cast %c0 : i32 to ui32
  return %c0_as_ui32 : ui32
}
// CHECK: func.func @unsigned_func(%arg0: i32, %arg1: memref<10xi32>) -> i32 {
// CHECK:   %[[C0:.*]] = arith.constant 0 : i32
// CHECK:   return %[[C0]] : i32
// CHECK: }

// -----

func.func @unsigned_dense_const() {
  %c = arith.constant dense<[1, 2, 3]> : tensor<3xui16>
  return
}
// CHECK: func.func @unsigned_dense_const() {
// CHECK:   %[[C:.*]] = arith.constant dense<[1, 2, 3]> : tensor<3xi16>
// CHECK:   return
// CHECK: }
