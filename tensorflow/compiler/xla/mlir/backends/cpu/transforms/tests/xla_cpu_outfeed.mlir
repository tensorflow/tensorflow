// RUN: xla-cpu-opt %s -xla-lmhlo-to-cpu-runtime | FileCheck %s


func.func @cpu_onfeed(%arg0: memref<8xf32>, %arg1: memref<10xui32>) {

  "xla_cpu.outfeed"(%arg0, %arg1) {config = "abc", result_type = [f32, ui32]} : (memref<8xf32>, memref<10xui32>) -> ()
  return
}

//      CHECK: func @cpu_onfeed(
// CHECK-SAME:   %[[ARG0:[a-z0-9]+]]: memref<8xf32>
// CHECK-SAME:   %[[ARG1:[a-z0-9]+]]: memref<10xui32>
// CHECK-SAME: )
//      CHECK:   call @[[OUTFEED:.*]](%[[ARG0]], %[[ARG1]])
// CHECK-SAME:   {result_type = [11 : i32, 8 : i32]} : (memref<8xf32>, memref<10xui32>) -> ()
//      CHECK:   func private @[[OUTFEED]](memref<8xf32>, memref<10xui32>)
// CHECK-SAME:   attributes {rt.custom_call = "xla.cpu.outfeed"}