// RUN: tf-opt -lhlo-copy-removal %s -o - | FileCheck %s --dump-input=always

// CHECK-LABEL: func @remove_simple
func @remove_simple(%arg0: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    "xla_lhlo.copy"(%0, %arg0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    dealloc %0 : memref<2x2xf32>
    // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
    "xla_lhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @remove_without_dealloc
func @remove_without_dealloc(%arg0: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    "xla_lhlo.copy"(%0, %arg0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
    "xla_lhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @replace_dependency
func @replace_dependency(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    %0 = alloc() {temp = true} : memref<2x2xf32>
    "xla_lhlo.exp"(%arg0, %0) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "xla_lhlo.exp"(%arg0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "xla_lhlo.copy"(%0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    dealloc %0 : memref<2x2xf32>
    // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
    "xla_lhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @keep_copies
func @keep_copies(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    // CHECK-NEXT: "xla_lhlo.copy"(%arg0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    "xla_lhlo.copy"(%arg0, %arg1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
    // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
    "xla_lhlo.terminator"() : () -> ()
}

