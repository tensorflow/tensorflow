// RUN: xla-gpu-opt %s --split-input-file -xla-lmhlo-to-gpu-runtime \
// RUN:   | FileCheck %s

module attributes {gpu.container_module} {
  memref.global "private" constant @constant : memref<i32> = dense<0>

  gpu.module @cond attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>, %arg1: memref<i1>) kernel {
      gpu.return
    }
  }

  gpu.module @body attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>) kernel {
      gpu.return
    }
  }

  // CHECK:      @while_loop(
  // CHECK-SAME:   %[[ARG0:.*]]: memref<i32>,
  // CHECK-SAME:   %[[ARG1:.*]]: memref<i1>
  // CHECK-SAME: )
  func.func @while_loop(%arg0: memref<i32>, %arg1: memref<i1>) {
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @constant : memref<i32>
    gpu.memcpy  %arg0, %0 : memref<i32>, memref<i32>

    // CHECK: %[[HOST_PRED:.*]] = memref.alloca() : memref<i1>
    // CHECK: scf.while : () -> ()
    "lmhlo.while"(%arg1) ({
      // CHECK: gpu.launch_func @cond::@fn
      // CHECK: gpu.memcpy %[[HOST_PRED]], %[[ARG1]]
      // CHECK: %[[COND:.*]] = memref.load %[[HOST_PRED]][] : memref<i1>
      // CHECK: scf.condition(%[[COND]])
      gpu.launch_func @cond::@fn blocks in (%c1, %c1, %c1)
                                 threads in (%c1, %c1, %c1)
                                 args(%arg0 : memref<i32>, %arg1 : memref<i1>)
      "lmhlo.terminator"() : () -> ()
    }, {
      // CHECK: gpu.launch_func @body::@fn
      // CHECK: scf.yield
      gpu.launch_func @body::@fn blocks in (%c1, %c1, %c1)
                                 threads in (%c1, %c1, %c1)
                                 args(%arg0 : memref<i32>)
      "lmhlo.terminator"() : () -> ()
    }) : (memref<i1>) -> ()
    "lmhlo.terminator"() : () -> ()
  }
}

// -----
// Check that while loops with known trip counts lower to `scf.for` loops.

module attributes {gpu.container_module} {
  memref.global "private" constant @constant : memref<i32> = dense<0>

  gpu.module @cond attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>, %arg1: memref<i1>) kernel {
      gpu.return
    }
  }

  gpu.module @body attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>) kernel {
      gpu.return
    }
  }

  // CHECK:      @for_loop(
  // CHECK-SAME:   %[[ARG0:.*]]: memref<i32>,
  // CHECK-SAME:   %[[ARG1:.*]]: memref<i1>
  // CHECK-SAME: )
  func.func @for_loop(%arg0: memref<i32>, %arg1: memref<i1>) {
    // CHECK: %[[LB:.*]] = arith.constant 0
    // CHECK: %[[UB:.*]] = arith.constant 3000
    // CHECK: %[[C1:.*]] = arith.constant 1
    %c1 = arith.constant 1 : index

    // CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[C1]]
    // CHECK-NEXT: gpu.launch_func @body::@fn
    // CHECK-NOT: gpu.launch.func

    "lmhlo.while"(%arg1) ({
      gpu.launch_func @cond::@fn blocks in (%c1, %c1, %c1)
                                 threads in (%c1, %c1, %c1)
                                 args(%arg0 : memref<i32>, %arg1 : memref<i1>)
      "lmhlo.terminator"() : () -> ()
    }, {
      gpu.launch_func @body::@fn blocks in (%c1, %c1, %c1)
                                 threads in (%c1, %c1, %c1)
                                 args(%arg0 : memref<i32>)
      "lmhlo.terminator"() : () -> ()
    }) {trip_count = 3000 : i64} : (memref<i1>) -> ()

    "lmhlo.terminator"() : () -> ()
  }
}
