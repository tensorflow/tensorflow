// RUN: xla-cpu-opt %s -split-input-file -xla-lmhlo-to-cpu-runtime \
// RUN: | FileCheck %s

func.func @cpu_outfeed(%arg0: memref<8xf32>, %arg1: memref<10xui32>) {
  "xla_cpu.outfeed"(%arg0, %arg1) {config = "abc", result_type = [f32, ui32]} : (memref<8xf32>, memref<10xui32>) -> ()
  return
}

//      CHECK: func @cpu_outfeed(
// CHECK-SAME:   %[[ARG0:[a-z0-9]+]]: memref<8xf32>
// CHECK-SAME:   %[[ARG1:[a-z0-9]+]]: memref<10xui32>
// CHECK-SAME: )
//      CHECK:   call @[[OUTFEED:.*]](%[[ARG0]], %[[ARG1]])
// CHECK-SAME:   {result_type = [11 : i32, 8 : i32]} : (memref<8xf32>, memref<10xui32>) -> ()
//      CHECK:   func private @[[OUTFEED]](memref<8xf32>, memref<10xui32>)
// CHECK-SAME:   attributes {rt.custom_call = "xla.cpu.outfeed"}

// -----

func.func @cpu_outfeed_strided(
  %arg0: memref<8x8xf32, strided<[?, 1], offset: ?>>,
  %arg1: memref<10xui32>) {
    "xla_cpu.outfeed"(%arg0, %arg1) {config = "abc", result_type = [f32, ui32]}
      : (memref<8x8xf32, strided<[?, 1], offset: ?>>, memref<10xui32>) -> ()
    return
}

//      CHECK: func @cpu_outfeed_strided(
// CHECK-SAME:   %[[ARG0:[a-z0-9]+]]: memref<8x8xf32, strided<[?, 1], offset: ?>>
// CHECK-SAME:   %[[ARG1:[a-z0-9]+]]: memref<10xui32>
// CHECK-SAME: )
// CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT:   memref.copy %[[ARG0]], %[[ALLOC]]
//      CHECK:   call @[[OUTFEED:.*]](%[[ALLOC]], %[[ARG1]])
// CHECK-SAME:   {result_type = [11 : i32, 8 : i32]} : (memref<8x8xf32>, memref<10xui32>) -> ()
//      CHECK:   func private @[[OUTFEED]](memref<8x8xf32>, memref<10xui32>)
// CHECK-SAME:   attributes {rt.custom_call = "xla.cpu.outfeed"}
