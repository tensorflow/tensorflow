// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// Test the rewrite patterns for lmhlo.infeed/outfeed to xlir.infeed/outfeed.

// TODO(bixia): Make the test pass without %dummy.
// CHECK: func @test_infeed_0
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @test_infeed_0(%dummy: memref<3xf32>) -> () {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute
  // CHECK: [[CHAIN:%[0-9]+]] = xlir.infeed %arg1, %arg0
  "lmhlo.infeed"() { config = "x" } : () -> ()
  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK: func @test_infeed_2
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @test_infeed_2(%output0: memref<3xf32>, %output1: memref<4x5xf32>) -> () {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute
  // CHECK: [[CHAIN:%[0-9]+]] = xlir.infeed %arg1, %arg2, %arg3, %arg0
  "lmhlo.infeed"(%output0, %output1) { config = "x" } : (memref<3xf32>, memref<4x5xf32>) -> ()
  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK: func @test_outfeed_1
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @test_outfeed_1(%input0: memref<3x5xf32>) -> () {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute
  // CHECK:[[CHAIN:%[0-9]+]] = xlir.outfeed %arg1, %arg2, %arg0
  "lmhlo.outfeed"(%input0) { config = "x" } : (memref<3x5xf32>) -> ()
  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
