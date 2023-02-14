// RUN: mlir-hlo-opt %s -hlo-deallocation-to-scf | FileCheck %s

func.func @retain_nothing(%arg0: memref<*xf32>, %arg1: memref<*xi32>) {
  deallocation.retain() of (%arg0, %arg1) : (memref<*xf32>, memref<*xi32>) -> ()
  return
}

// CHECK-LABEL: @retain_nothing
// CHECK-SAME:     %[[ARG0:.*]]: memref<*xf32>, %[[ARG1:.*]]: memref<*xi32>
// CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[BUF0:.*]] = deallocation.get_buffer %[[ARG0]]
// CHECK-NEXT:  %[[NONNULL0:.*]] = arith.cmpi ne, %[[BUF0]], %[[ZERO]]
// CHECK-NEXT:  scf.if %[[NONNULL0]] {
// CHECK-NEXT:    memref.dealloc %[[ARG0]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[BUF1:.*]] = deallocation.get_buffer %[[ARG1]]
// CHECK-NEXT:  %[[NONNULL1:.*]] = arith.cmpi ne, %[[BUF1]], %[[ZERO]]
// CHECK-NEXT:  scf.if %[[NONNULL1]] {
// CHECK-NEXT:    memref.dealloc %[[ARG1]]
// CHECK-NEXT:  }
