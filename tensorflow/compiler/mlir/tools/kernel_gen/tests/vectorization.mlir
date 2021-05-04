// RUN: kernel-gen-opt %s --vectorization --cse --vectorization-cleanup --canonicalize | FileCheck %s

func @Abs(%in: memref<?xf64>) {
linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                iterator_types = ["parallel"]}
    ins(%in : memref<?xf64>) outs(%in : memref<?xf64>) {
  ^bb0(%arg1: f64, %arg2: f64):  // no predecessors
    %9 = absf %arg1 : f64
    linalg.yield %9 : f64
  }
  return
}
// CHECK-LABEL:   func @Abs(
// CHECK-SAME:              %[[BUF:.*]]: memref<?xf64>) {
// CHECK-DAG:       %[[CF0:.*]] = constant 0.000000e+00 : f64
// CHECK-DAG:       %[[C4:.*]] = constant 4 : index
// CHECK-DAG:       %[[C0:.*]] = constant 0 : index
// The following alloction and operations should be removed.
// CHECK-DAG:       %[[DEAD_BUF:.*]] = memref.alloca() {alignment = 32 : i64} : memref<4xf64>
// CHECK-NEXT:      %[[SIZE:.*]] = memref.dim %[[BUF]], %[[C0]] : memref<?xf64>
// CHECK-NEXT:      %[[REM:.*]] = remi_unsigned %[[SIZE]], %[[C4]] : index
// CHECK-NEXT:      %[[SPLIT_POINT:.*]] = subi %[[SIZE]], %[[REM]] : index
// CHECK-NEXT:      scf.for %[[IV_MAIN:.*]] = %[[C0]] to %[[SPLIT_POINT]] step %[[C4]] {
// CHECK-NEXT:        %[[OUTPUT_FULL_TILE:.*]] = memref.subview %[[BUF]]{{\[}}%[[IV_MAIN]]] [4] [1] : memref<?xf64> to memref<4xf64, #map0>
// The following code can be removed.
// CHECK-NEXT:        %[[VAL_10:.*]] = memref.alloc() {alignment = 16 : i64} : memref<32xi8>
// CHECK-NEXT:        %[[VAL_12:.*]] = memref.view %[[VAL_10]]{{\[}}%[[C0]]][] : memref<32xi8> to memref<4xf64>
// TODO: Remove the following dead code
// CHECK-NEXT:        linalg.fill
// CHECK-NEXT:        linalg.copy
// CHECK-NEXT:        %[[FULL_TILE:.*]] = vector.transfer_read %[[OUTPUT_FULL_TILE]]{{\[}}%[[C0]]], %[[CF0]] {in_bounds = [true]} : memref<4xf64, #map0>, vector<4xf64>
// CHECK-NEXT:        %[[ABS0:.*]] = absf %[[FULL_TILE]] : vector<4xf64>
// CHECK-NEXT:        vector.transfer_write %[[ABS0]], %[[OUTPUT_FULL_TILE]]{{\[}}%[[C0]]] {in_bounds = [true]} : vector<4xf64>, memref<4xf64, #map0>
// CHECK-NEXT:        memref.dealloc %[[VAL_10]] : memref<32xi8>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.for %[[IV_LAST:.*]] = %[[SPLIT_POINT]] to %[[SIZE]] step %[[C4]] {
// The following min can be replaced with the non-const value
// CHECK-NEXT:        %[[TILE_SIZE:.*]] = affine.min #map1(%[[IV_LAST]]){{\[}}%[[SIZE]]]
// CHECK-NEXT:        %[[SUBVIEW:.*]] = memref.subview %[[BUF]]{{\[}}%[[IV_LAST]]] {{\[}}%[[TILE_SIZE]]] [1] : memref<?xf64> to memref<?xf64, #map0>
// CHECK-NEXT:        %[[COND:.*]] = cmpi sle, %[[C4]], %[[TILE_SIZE]] : index
// The following if can be replaced with the false condition.
// CHECK-NEXT:        %[[MASKED_VIEW:.*]] = scf.if %[[COND]] -> (memref<?xf64, #map0>) {
// CHECK-NEXT:          scf.yield %[[SUBVIEW]] : memref<?xf64, #map0>
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[VAL_20:.*]] = vector.transfer_read %[[SUBVIEW]]{{\[}}%[[C0]]], %[[CF0]] : memref<?xf64, #map0>, vector<4xf64>
// CHECK-NEXT:          %[[VAL_21:.*]] = vector.type_cast %[[DEAD_BUF]] : memref<4xf64> to memref<vector<4xf64>>
// CHECK-NEXT:          memref.store %[[VAL_20]], %[[VAL_21]][] : memref<vector<4xf64>>
// CHECK-NEXT:          %[[VAL_22:.*]] = memref.cast %[[DEAD_BUF]] : memref<4xf64> to memref<?xf64, #map0>
// CHECK-NEXT:          scf.yield %[[VAL_22]] : memref<?xf64, #map0>
// CHECK-NEXT:        }
// The following view code can be removed.
// CHECK-NEXT:        %[[VAL_23:.*]] = memref.alloc() {alignment = 16 : i64} : memref<32xi8>
// CHECK-NEXT:        %[[VAL_25:.*]] = memref.view %[[VAL_23]]{{\[}}%[[C0]]][] : memref<32xi8> to memref<4xf64>
// CHECK-NEXT:        %[[VAL_26:.*]] = memref.subview %[[VAL_25]][0] {{\[}}%[[TILE_SIZE]]] [1] : memref<4xf64> to memref<?xf64>
// TODO: Remove the following dead code
// CHECK-NEXT:        linalg.fill
// CHECK-NEXT:        linalg.copy
// CHECK-NEXT:        %[[INPUT_TILE:.*]] = vector.transfer_read %[[VAL_28:.*]]{{\[}}%[[C0]]], %[[CF0]] {in_bounds = [true]} : memref<?xf64, #map0>, vector<4xf64>
// CHECK-NEXT:        %[[ABS1:.*]] = absf %[[INPUT_TILE]] : vector<4xf64>
// CHECK-NEXT:        vector.transfer_write %[[ABS1]], %[[SUBVIEW]]{{\[}}%[[C0]]] : vector<4xf64>, memref<?xf64, #map0>
// CHECK-NEXT:        memref.dealloc %[[VAL_23]] : memref<32xi8> 
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

