// RUN: mlir-hlo-opt %s -hlo-deallocation-to-scf | FileCheck %s

func.func @retain_nothing(%arg0: memref<*xf32>) {
  deallocation.retain() of (%arg0) : (memref<*xf32>) -> ()
  return
}

// CHECK-LABEL: @retain_nothing
// CHECK-SAME:     %[[ARG:.*]]: memref<*xf32>
// CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[BUF:.*]] = deallocation.get_buffer %[[ARG]]
// CHECK-NEXT:  %[[NONNULL:.*]] = arith.cmpi ne, %[[BUF]], %[[ZERO]]
// CHECK-NEXT:  scf.if %[[NONNULL]] {
// CHECK-NEXT:    memref.dealloc %[[ARG]]
// CHECK-NEXT:  }

// -----

func.func @retain_something(%arg0: memref<2xf32>, %arg1: memref<*xf32>)
    -> memref<*xf32> {
  %ret = deallocation.retain(%arg0) of (%arg1) : (memref<2xf32>, memref<*xf32>)
      -> (memref<*xf32>)
  return %ret : memref<*xf32>
}

// CHECK-LABEL: @retain_something
// CHECK-SAME:     %[[ARG0:.*]]: memref<2xf32>
// CHECK-SAME:     %[[ARG1:.*]]: memref<*xf32>
// CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[BUF:.*]] = deallocation.get_buffer %[[ARG1]]
// CHECK-NEXT:  %[[RETAINED_BUF:.*]] = deallocation.get_buffer %[[ARG0]]
// CHECK-NEXT:  %[[SAME:.*]] = arith.cmpi eq, %[[BUF]], %[[RETAINED_BUF]]
// CHECK-NEXT:  %[[RET:.*]] = scf.if %[[SAME]]
// CHECK-NEXT:    scf.yield %[[ARG1]]
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %[[NONNULL:.*]] = arith.cmpi ne, %[[BUF]], %[[ZERO]]
// CHECK-NEXT:    scf.if %[[NONNULL]] {
// CHECK-NEXT:      memref.dealloc %[[ARG1]]
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[NULL:.*]] = deallocation.null
// CHECK-NEXT:    scf.yield %[[NULL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[RET]]
