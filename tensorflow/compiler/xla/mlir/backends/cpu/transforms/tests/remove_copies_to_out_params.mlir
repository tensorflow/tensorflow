// RUN: xla-cpu-opt %s -split-input-file -xla-remove-copies-to-out-params \
// RUN:   | FileCheck %s

func.func @alloca(%arg0: memref<f64>, %arg1: memref<f64>) {
  %0 = memref.load %arg0[] : memref<f64>
  %1 = arith.addf %0, %0 : f64
  %alloca = memref.alloca() : memref<f64>
  memref.store %1, %alloca[] : memref<f64>
  memref.copy %alloca, %arg1 : memref<f64> to memref<f64>
  return
}

// CHECK-LABEL: func.func @alloca(
// CHECK-SAME:                    %[[ARG0:.*]]: memref<f64>,
// CHECK-SAME:                    %[[ARG1:.*]]: memref<f64>) {
// CHECK:         %[[R0:.*]] = memref.load %[[ARG0]][] : memref<f64>
// CHECK:         %[[R1:.*]] = arith.addf %[[R0]], %[[R0]] : f64
// CHECK-NOT      memref.alloca
// CHECK:         memref.store %[[R1]], %[[ARG1]][] : memref<f64>
// CHECK-NOT:     memref.copy
// CHECK-NEXT:    return
// CHECK:       }

// -----

func.func @alloc_vectorized(%arg0: memref<1024xf64>, %arg1: memref<1024xf64>) {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f64
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xf64>
  scf.parallel (%arg2) = (%c0) to (%c1024) step (%c8) {
    %subview = memref.subview %alloc[%arg2] [8] [1] :
        memref<1024xf64> to memref<8xf64, strided<[1], offset: ?>>
    %0 = vector.transfer_read %arg0[%arg2], %cst {in_bounds = [true]} :
        memref<1024xf64>, vector<8xf64>
    %1 = arith.addf %0, %0 : vector<8xf64>
    vector.transfer_write %1, %subview[%c0] {in_bounds = [true]} :
        vector<8xf64>, memref<8xf64, strided<[1], offset: ?>>
    scf.yield
  }
  memref.copy %alloc, %arg1 : memref<1024xf64> to memref<1024xf64>
  memref.dealloc %alloc : memref<1024xf64>
  return
}

// CHECK-LABEL: func.func @alloc_vectorized(
// CHECK-SAME:                              %[[ARG0:.*]]: memref<1024xf64>,
// CHECK-SAME:                              %[[ARG1:.*]]: memref<1024xf64>) {
// CHECK-NOT:     memref.alloc
// CHECK:         scf.parallel
// CHECK:           %[[SUBVIEW:.*]] = memref.subview %[[ARG1]]
// CHECK:           %[[R0:.*]] = vector.transfer_read %[[ARG0]]
// CHECK:           %[[R1:.*]] = arith.addf %[[R0]], %[[R0]] : vector<8xf64>
// CHECK:           vector.transfer_write %[[R1]], %[[SUBVIEW]]
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NOT:     memref.copy
// CHECK-NOT:     memref.dealloc
// CHECK-NEXT:    return
// CHECK:       }

// -----

// Similar to alloc_vectorized, but with two output params (%arg1 and %arg2).
// Note: %arg1 = %arg0 + %arg0, and %arg2 = (%arg0 + %arg0) * %arg0
func.func @alloc2_vectorized(%arg0: memref<256xf64>,
                             %arg1: memref<256xf64>,
                             %arg2: memref<256xf64>) {
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f64
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<256xf64>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256xf64>
  scf.parallel (%arg3) = (%c0) to (%c256) step (%c8) {
    %alloca = memref.alloca() : memref<8xf64>
    %0 = vector.transfer_read %arg0[%arg3], %cst {in_bounds = [true]} : memref<256xf64>, vector<8xf64>
    %1 = arith.addf %0, %0 : vector<8xf64>
    vector.transfer_write %1, %alloca[%c0] {in_bounds = [true]} : vector<8xf64>, memref<8xf64>
    %subview = memref.subview %alloc_0[%arg3] [8] [1] : memref<256xf64> to memref<8xf64, strided<[1], offset: ?>>
    memref.copy %alloca, %subview : memref<8xf64> to memref<8xf64, strided<[1], offset: ?>>
    scf.yield
  }
  scf.parallel (%arg3) = (%c0) to (%c256) step (%c8) {
    %subview = memref.subview %alloc[%arg3] [8] [1] : memref<256xf64> to memref<8xf64, strided<[1], offset: ?>>
    %0 = vector.transfer_read %alloc_0[%arg3], %cst {in_bounds = [true]} : memref<256xf64>, vector<8xf64>
    %1 = vector.transfer_read %arg0[%arg3], %cst {in_bounds = [true]} : memref<256xf64>, vector<8xf64>
    %2 = arith.mulf %0, %1 : vector<8xf64>
    vector.transfer_write %2, %subview[%c0] {in_bounds = [true]} : vector<8xf64>, memref<8xf64, strided<[1], offset: ?>>
    scf.yield
  }
  memref.copy %alloc_0, %arg1 : memref<256xf64> to memref<256xf64>
  memref.dealloc %alloc_0 : memref<256xf64>
  memref.copy %alloc, %arg2 : memref<256xf64> to memref<256xf64>
  memref.dealloc %alloc : memref<256xf64>
  return
}

// CHECK-LABEL: func.func @alloc2_vectorized(
// CHECK-SAME:      %[[ARG0:[0-9a-z]*]]: memref<256xf64>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<256xf64>,
// CHECK-SAME:      %[[ARG2:.*]]: memref<256xf64>) {
// CHECK-NOT:     memref.alloc
// CHECK:         scf.parallel
// CHECK:           %[[ALLOCA:.*]] = memref.alloca()
// CHECK:           %[[R0:.*]] = vector.transfer_read %[[ARG0]]
// CHECK:           %[[R1:.*]] = arith.addf %[[R0]], %[[R0]]
// CHECK:           vector.transfer_write %[[R1]], %[[ALLOCA]]
// CHECK:           %[[SUBVIEW:.*]] = memref.subview %[[ARG1]]
// CHECK:           memref.copy %[[ALLOCA]], %[[SUBVIEW]]
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NOT:     memref.copy
// CHECK-NOT:     memref.dealloc
// CHECK-NEXT:    scf.parallel
// CHECK:           %[[SUBVIEW:.*]] = memref.subview %[[ARG2]]
// CHECK:           %[[R0:.*]] = vector.transfer_read %[[ARG1]]
// CHECK:           %[[R1:.*]] = vector.transfer_read %[[ARG0]]
// CHECK:           %[[R2:.*]] = arith.mulf %[[R0]], %[[R1]]
// CHECK:           vector.transfer_write %[[R2]], %[[SUBVIEW]]
// CHECK:           scf.yield
// CHECK:         }
// CHECK-NOT:     memref.copy
// CHECK-NOT:     memref.dealloc
// CHECK-NEXT:    return
// CHECK:       }
