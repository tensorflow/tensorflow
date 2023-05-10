// RUN: xla-cpu-opt %s -xla-lmhlo-to-cpu-runtime | FileCheck %s

func.func @infeed(%arg0 : memref<3x3xi32>, %arg1 : memref<i1>) -> () {
  "xla_cpu.infeed"(%arg0, %arg1) {config = "foobar", layout = [[0, 1], [0]]}
    : (memref<3x3xi32>, memref<i1>) -> ()
  return
}

//      CHECK: func @infeed(
// CHECK-SAME:   %[[ARG0:[a-z0-9]+]]: memref<3x3xi32>
// CHECK-SAME:   %[[ARG1:[a-z0-9]+]]: memref<i1>
// CHECK-SAME: )
//      CHECK:   call @[[INFEED:.*]](%[[ARG0]], %[[ARG1]])
// CHECK SAME:   : (memref<3x3xi32>, memref<i1>) -> ()
//      CHECK:   func private @[[INFEED]](memref<3x3xi32>, memref<i1>)
// CHECK-SAME:   attributes {rt.custom_call = "[[INFEED]]"}
