// RUN: mlir-hlo-opt %s -hlo-deallocation-to-scf | FileCheck %s

func.func @retain_nothing(%arg0: !deallocation.ownership) {
  deallocation.retain() of (%arg0) : (!deallocation.ownership) -> ()
  return
}

// CHECK-LABEL: @retain_nothing
// CHECK-SAME:     %[[ARG:.*]]:
// CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[BUF:.*]] = deallocation.get_buffer %[[ARG]]
// CHECK-NEXT:  %[[NONNULL:.*]] = arith.cmpi ne, %[[BUF]], %[[ZERO]]
// CHECK-NEXT:  scf.if %[[NONNULL]] {
// CHECK-NEXT:    deallocation.free %[[ARG]]
// CHECK-NEXT:  }

// -----

func.func @retain_something(%arg0: memref<2xf32>, %arg1: !deallocation.ownership)
    -> !deallocation.ownership {
  %ret = deallocation.retain(%arg0) of (%arg1) : (memref<2xf32>, !deallocation.ownership)
      -> (!deallocation.ownership)
  return %ret : !deallocation.ownership
}

// CHECK-LABEL: @retain_something
// CHECK-SAME:     %[[ARG0:.*]]: memref<2xf32>, %[[ARG1:.*]]:
// CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[BUF:.*]] = deallocation.get_buffer %[[ARG1]]
// CHECK-NEXT:  %[[NULL:.*]] = deallocation.null
// CHECK-NEXT:  %[[RETAINED_BUF:.*]] = deallocation.get_buffer %[[ARG0]]
// CHECK-NEXT:  %[[SAME:.*]] = arith.cmpi eq, %[[RETAINED_BUF]], %[[BUF]]
// CHECK-NEXT:  %[[RET:.*]]:3 = scf.if %[[SAME]]
// CHECK-NEXT:    scf.yield %[[NULL]], %[[ZERO]], %[[ARG1]]
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.yield %[[ARG1]], %[[BUF]], %[[NULL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[DEALLOC:.*]] = arith.cmpi ne, %[[RET]]#1, %[[ZERO]]
// CHECK-NEXT:  scf.if %[[DEALLOC]] {
// CHECK-NEXT:    deallocation.free %[[RET]]#0
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[RET]]#2

func.func @retain_multiple(%arg0: memref<?xi32>, %arg1: memref<?xi32>,
        %arg2: !deallocation.ownership, %arg3: !deallocation.ownership)
    -> (!deallocation.ownership, !deallocation.ownership) {
  %ret:2 = deallocation.retain(%arg0, %arg1) of (%arg2, %arg3)
    : (memref<?xi32>, memref<?xi32>, !deallocation.ownership, !deallocation.ownership)
    -> (!deallocation.ownership, !deallocation.ownership)
  return %ret#0, %ret#1 : !deallocation.ownership, !deallocation.ownership
}

// CHECK-LABEL: @retain_multiple
// CHECK-SAME:     %[[ARG0:.*]]: memref<?xi32>, %[[ARG1:.*]]: memref<?xi32>
// CHECK-SAME:     %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]:
// CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[BUF2:.*]] = deallocation.get_buffer %[[ARG2]]
// CHECK-NEXT:  %[[BUF3:.*]] = deallocation.get_buffer %[[ARG3]]
// CHECK-NEXT:  %[[NULL:.*]] = deallocation.null
// CHECK-NEXT:  %[[BUF0:.*]] = deallocation.get_buffer %[[ARG0]]
// CHECK:       %[[CMP:.*]] = arith.cmpi eq, %[[BUF0]], %[[BUF2]] : index
// CHECK:       %[[T0:.*]]:3 = scf.if %[[CMP]]
// CHECK:       %[[CMP:.*]] = arith.cmpi eq, %[[BUF0]], %[[BUF3]] : index
// CHECK:       %[[T1:.*]]:3 = scf.if %[[CMP]]
// CHECK:       %[[BUF1:.*]] = deallocation.get_buffer %[[ARG1]]
// CHECK:       %[[CMP:.*]] = arith.cmpi eq, %[[BUF1]], %[[T0]]#1 : index
// CHECK:       scf.if %[[CMP]]
// CHECK:       %[[CMP:.*]] = arith.cmpi eq, %[[BUF1]], %[[T1]]#1 : index
// CHECK:       scf.if %[[CMP]]
