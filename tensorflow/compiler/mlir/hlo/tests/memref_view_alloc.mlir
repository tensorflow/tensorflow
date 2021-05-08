// RUN: mlir-hlo-opt --memref-view-to-alloc --split-input-file %s | FileCheck %s

// Expected behavior: This code transforms the memref view to an alloc.
// CHECK-LABEL: func @viewToAlloc
 func @viewToAlloc(%arg0: memref<4xi8>){
    %c0 = constant 0 : index
    %0 = memref.view %arg0[%c0][] : memref<4xi8> to memref<i32>
    return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
// CHECK-NEXT: %[[C0:.*]] = constant 0 : index
// CHECK-NEXT: %[[V0:.*]] = memref.alloc() : memref<i32>

// -----

// Expected behavior: This code transforms the
// memref view with a dynamic shape to an alloc.
// CHECK-LABEL: func @dynamicTypeViewToAlloc
 func @dynamicTypeViewToAlloc(%arg0: memref<?xi8>){
    %c0 = constant 0 : index
    %0 = memref.view %arg0[%c0][] : memref<?xi8> to memref<i32>
    return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
// CHECK-NEXT: %[[C0:.*]] = constant 0 : index
// CHECK-NEXT: %[[V0:.*]] = memref.alloc() : memref<i32>
