// RUN: tf-opt -buffer-assignment -split-input-file %s | FileCheck %s -dump-input-on-failure

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
func @moving_alloc_and_inserting_missing_dealloc(%cond : i1, %arg0 : memref<2xf32>, %arg1: memref<2xf32>){
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %0 = alloc() : memref<2xf32>
    "xla_lhlo.exponential"(%arg0, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    br ^exit(%0 : memref<2xf32>)
  ^bb2:

    %1 = alloc() : memref<2xf32>
    "xla_lhlo.exponential"(%arg0, %1) : (memref<2xf32>, memref<2xf32>) -> ()
    br ^exit(%1 : memref<2xf32>)
  ^exit(%arg2: memref<2xf32>):
    "xla_lhlo.copy"(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
// CHECK-NEXT: {{.*}} = alloc
// CHECK-NEXT: {{.*}} = alloc
//      CHECK: dealloc
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @moving_invalid_dealloc_op_complex
func @moving_invalid_dealloc_op_complex(%cond : i1, %arg0 : memref<2xf32>, %arg1: memref<2xf32>){
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : memref<2xf32>)
  ^bb2:
    %1 = alloc() : memref<2xf32>
    "xla_lhlo.exponential"(%arg0, %1) : (memref<2xf32>, memref<2xf32>) -> ()
    dealloc %1 : memref<2xf32>
    br ^exit(%1 : memref<2xf32>)
  ^exit(%arg2: memref<2xf32>):
    "xla_lhlo.copy"(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
// CHECK-NEXT: {{.*}} = alloc
//      CHECK: xla_lhlo.copy
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @inserting_missing_dealloc_simple
func @inserting_missing_dealloc_simple(%arg0 : memref<2xf32>, %arg1: memref<2xf32>){
    %0 = alloc() : memref<2xf32>
    "xla_lhlo.exponential"(%arg0, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    "xla_lhlo.copy"(%0, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
//      CHECK: xla_lhlo.copy
// CHECK-NEXT: dealloc

// -----

// CHECK-LABEL: func @moving_invalid_dealloc_op
func @moving_invalid_dealloc_op(%arg0 : memref<2xf32>, %arg1: memref<2xf32>){
    %0 = alloc() : memref<2xf32>
    "xla_lhlo.exponential"(%arg0, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    dealloc %0 : memref<2xf32>
    "xla_lhlo.copy"(%0, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
//      CHECK: xla_lhlo.copy
// CHECK-NEXT: dealloc