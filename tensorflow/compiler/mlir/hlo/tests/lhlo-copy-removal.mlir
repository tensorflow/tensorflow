// RUN: mlir-hlo-opt -lhlo-copy-removal %s -o - | FileCheck %s

// CHECK-LABEL: func @remove_simple
func @remove_simple(%arg0: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    "lmhlo.copy"(%0, %arg0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    dealloc %0 : memref<2x2xf32>
    // CHECK-NEXT: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @remove_without_dealloc
func @remove_without_dealloc(%arg0: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    "lmhlo.copy"(%0, %arg0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @replace_dependency
func @replace_dependency(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    "lmhlo.exponential"(%arg0, %0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "lmhlo.exponential"(%arg0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.copy"(%0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    dealloc %0 : memref<2x2xf32>
    // CHECK-NEXT: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @keep_copies
func @keep_copies(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    // CHECK-NEXT: "lmhlo.copy"(%arg0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.copy"(%arg0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @must_not_be_removed
func @must_not_be_removed(%arg0: memref<2x2xf32>,
                          %arg1: memref<2x2xf32>,
                          %arg2: memref<2x2xf32>) {
    // CHECK-NEXT: %[[ALLOC:.*]] = alloc() {temp = true} : memref<2x2xf32>
    %0 = alloc() {temp = true} : memref<2x2xf32>
    // CHECK-NEXT: "lmhlo.exponential"(%arg0, %[[ALLOC]]) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.exponential"(%arg0, %0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "lmhlo.exponential"(%arg1, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.exponential"(%arg1, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "lmhlo.copy"(%[[ALLOC]], %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.copy"(%0, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    dealloc %0 : memref<2x2xf32>
    "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @must_be_removed_first
func @must_be_removed_first(%arg0: memref<2x2xf32>,
                            %arg1: memref<2x2xf32>,
                            %arg2: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    // CHECK-NEXT: "lmhlo.exponential"(%arg1, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.exponential"(%arg1, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "lmhlo.exponential"(%arg0, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.exponential"(%arg0, %0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.copy"(%0, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    dealloc %0 : memref<2x2xf32>
    "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @must_be_removed_second
func @must_be_removed_second(%arg0: memref<2x2xf32>,
                             %arg1: memref<2x2xf32>,
                             %arg2: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    // CHECK-NEXT: "lmhlo.exponential"(%arg0, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.exponential"(%arg0, %0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.copy"(%0, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "lmhlo.exponential"(%arg1, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "lmhlo.exponential"(%arg1, %arg2) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    dealloc %0 : memref<2x2xf32>
    "lmhlo.terminator"() : () -> ()
}
