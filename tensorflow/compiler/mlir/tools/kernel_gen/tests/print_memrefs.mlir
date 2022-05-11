// RUN: kernel-gen-opt %s --embed-memref-prints | FileCheck %s

func.func @print_memrefs(
    %ctx: !tf_framework.op_kernel_context, %input: memref<*xf32>)
    -> memref<*xf32> attributes {tf_entry} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rank = memref.rank %input : memref<*xf32>
  %shape = memref.alloca(%rank) : memref<?xindex>
  scf.for %i = %c0 to %rank step %c1 {
    %dim = memref.dim %input, %i : memref<*xf32>
    memref.store %dim, %shape[%i] : memref<?xindex>
  }

  %c9000 = arith.constant 9000 : index
  %num_elem = memref.alloca() : memref<1xindex>
  memref.store %c9000, %num_elem[%c0] : memref<1xindex>
  %flat_input = memref.reshape %input(%num_elem)
    : (memref<*xf32>, memref<1xindex>) -> memref<?xf32>

  %flat_output = tf_framework.alloc(%ctx, %c9000) : memref<?xf32>
  %output = memref.reshape %flat_output(%shape)
    : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
  func.return %output : memref<*xf32>
}

// CHECK-DAG: global internal constant @[[STR0:debug_op_[0-9]+]]({{.*}} @print_memrefs
// CHECK-DAG: global internal constant @[[STR1:debug_op_[0-9]+]]({{.*}} -> memref<?xf32>
// CHECK-DAG: global internal constant @[[STR2:debug_op_[0-9]+]]({{.*}} -> memref<*xf32>
// CHECK-DAG: func private @printMemrefF32(memref<*xf32>)
// CHECK-DAG: llvm.func @printCString(!llvm.ptr<i8>)

// CHECK: func @print_memrefs
// CHECK-SAME:     , %[[ARG:.*]]: memref<*xf32>)
// Print debug info for the function arg.
// CHECK:       %[[STR0_ADDR:.*]] = llvm.mlir.addressof @[[STR0]]
// CHECK:       %[[STR0_PTR:.*]] = llvm.getelementptr %[[STR0_ADDR]]
// CHECK:       llvm.call @printCString(%[[STR0_PTR]]) : (!llvm.ptr<i8>)
// CHECK:       call @printMemrefF32(%[[ARG]]) : (memref<*xf32>) -> ()

// Print debug info for reshape from unranked to ranked.
// CHECK:       %[[RESHAPE:.*]] = memref.reshape %[[ARG]]
// CHECK:       %[[STR1_ADDR:.*]] = llvm.mlir.addressof @[[STR1]]
// CHECK:       %[[STR1_PTR:.*]] = llvm.getelementptr %[[STR1_ADDR]]
// CHECK:       llvm.call @printCString(%[[STR1_PTR]]) : (!llvm.ptr<i8>)
// CHECK:       %[[UNRANKED_BUF:.*]] = memref.cast %[[RESHAPE]]
// CHECK:       call @printMemrefF32(%[[UNRANKED_BUF]]) : (memref<*xf32>)

// Print debug info for reshape from ranked to unranked.
// CHECK:       %[[ALLOC:.*]] = tf_framework.alloc
// CHECK:       %[[RESHAPE_2:.*]] = memref.reshape %[[ALLOC]]
// CHECK:       %[[STR2_ADDR:.*]] = llvm.mlir.addressof @[[STR2]]
// CHECK:       %[[STR2_PTR:.*]] = llvm.getelementptr %[[STR2_ADDR]]
// CHECK:       llvm.call @printCString(%[[STR2_PTR]]) : (!llvm.ptr<i8>)
// CHECK:       call @printMemrefF32(%[[RESHAPE_2]]) : (memref<*xf32>)
