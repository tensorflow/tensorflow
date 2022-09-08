// RUN: xla-gpu-opt %s -xla-lmhlo-to-gpu-runtime | FileCheck %s

module attributes {gpu.container_module} {
  memref.global "private" constant @constant : memref<i32> = dense<0>

  gpu.module @case0 attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>) kernel {
      gpu.return
    }
  }

  gpu.module @case1 attributes {binary = "ptx"} {
    gpu.func @fn(%arg0: memref<i32>) kernel {
      gpu.return
    }
  }

  // CHECK: @case_true_false(
  // CHECK-SAME:   %[[ARG0:.*]]: memref<i32>,
  // CHECK-SAME:   %[[ARG1:.*]]: memref<i1>
  // CHECK-SAME: )
  func.func @case_true_false(%arg0: memref<i32>, %arg1: memref<i1>) {
    %c1 = arith.constant 1 : index

    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[C1:.*]] = arith.constant 1 : i32

    // CHECK: %[[HOST:.*]] = memref.alloca() : memref<i1>
    // CHECK: gpu.memcpy %[[HOST]], %[[ARG1]]

    // CHECK: %[[PRED:.*]] = memref.load %[[HOST]][] : memref<i1>
    // CHECK: %[[IDX:.*]] = arith.select %[[PRED]], %[[C0]], %[[C1]]

    // CHECK: cf.switch %[[IDX]] : i32
    // CHECK:   default: ^[[CONT:.*]],
    // CHECK:         0: ^[[CASE0:.*]],
    // CHECK:         1: ^[[CASE1:.*]]
    "lmhlo.case"(%arg1) ({
      gpu.launch_func @case0::@fn blocks in (%c1, %c1, %c1)
                                  threads in (%c1, %c1, %c1)
                                  args(%arg0 : memref<i32>)
      "lmhlo.terminator"() : () -> ()
    }, {
      gpu.launch_func @case1::@fn blocks in (%c1, %c1, %c1)
                                  threads in (%c1, %c1, %c1)
                                  args(%arg0 : memref<i32>)
      "lmhlo.terminator"() : () -> ()
    }) : (memref<i1>) -> ()

    // CHECK: ^[[CASE0]]:
    // CHECK: gpu.launch_func @case0::@fn
    // CHECK: cf.br ^[[CONT]]

    // CHECK: ^[[CASE1]]:
    // CHECK: gpu.launch_func @case1::@fn
    // CHECK: cf.br ^[[CONT]]

    // CHECK: ^[[CONT]]:
    // CHECK: return
    "lmhlo.terminator"() : () -> ()
  }

  // CHECK: @case_index(
  // CHECK-SAME:   %[[ARG0:.*]]: memref<i32>,
  // CHECK-SAME:   %[[ARG1:.*]]: memref<i32>
  // CHECK-SAME: )
  func.func @case_index(%arg0: memref<i32>, %arg1: memref<i32>) {
    %c1 = arith.constant 1 : index

    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[C1:.*]] = arith.constant 1 : i32

    // CHECK: %[[HOST:.*]] = memref.alloca() : memref<i32>
    // CHECK: gpu.memcpy %[[HOST]], %[[ARG1]]

    // CHECK: %[[PRED:.*]] = memref.load %[[HOST]][] : memref<i32>
    // CHECK: %[[SMALL:.*]] = arith.cmpi slt, %[[PRED]], %[[C0]] : i32
    // CHECK: %[[LARGE:.*]] = arith.cmpi sgt, %[[PRED]], %[[C1]] : i32
    // CHECK: %[[OOR:.*]] = arith.ori %[[SMALL]], %[[LARGE]] : i1
    // CHECK: %[[IDX:.*]] = arith.select %[[OOR]], %[[C1]], %[[PRED]] : i32

    // CHECK: cf.switch %[[IDX]] : i32
    // CHECK:   default: ^[[CONT:.*]],
    // CHECK:         0: ^[[CASE0:.*]],
    // CHECK:         1: ^[[CASE1:.*]]
    "lmhlo.case"(%arg1) ({
      gpu.launch_func @case0::@fn blocks in (%c1, %c1, %c1)
                                  threads in (%c1, %c1, %c1)
                                  args(%arg0 : memref<i32>)
      "lmhlo.terminator"() : () -> ()
    }, {
      gpu.launch_func @case1::@fn blocks in (%c1, %c1, %c1)
                                  threads in (%c1, %c1, %c1)
                                  args(%arg0 : memref<i32>)
      "lmhlo.terminator"() : () -> ()
    }) : (memref<i32>) -> ()

    // CHECK: ^[[CASE0]]:
    // CHECK: gpu.launch_func @case0::@fn
    // CHECK: cf.br ^[[CONT]]

    // CHECK: ^[[CASE1]]:
    // CHECK: gpu.launch_func @case1::@fn
    // CHECK: cf.br ^[[CONT]]

    // CHECK: ^[[CONT]]:
    // CHECK: return
    "lmhlo.terminator"() : () -> ()
  }
}
