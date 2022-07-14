// RUN: lhlo-tfrt-opt %s -lmhlo-to-tfrt-branch -split-input-file | FileCheck %s

// CHECK-DAG: func @[[true_func:[^(]+]]{{.*[[:space:]][ ]+}}gpu.memcpy %arg1, %arg0 : memref<f32>, memref<f32>
// CHECK-DAG: func @[[false_func:[^(]+]]{{.*[[:space:]][ ]+}}gpu.memcpy %arg0, %arg1 : memref<f32>, memref<f32>

// CHECK:       func @case_i1(
// CHECK-SAME:    %arg0: memref<i1>,
// CHECK-SAME:    %arg1: memref<f32>,
// CHECK-SAME:    %arg2: memref<f32>
// CHECK-SAME:  ) {

func.func @case_i1(%index: memref<i1>, %operand_1: memref<f32>, %operand_2: memref<f32>) -> () {
  // CHECK:     %[[cond:.*]] = memref.load %arg0[] : memref<i1>
  // CHECK-DAG: tfrt.cond %[[cond]] @[[true_func]] @[[false_func]](%arg1, %arg2)
  "lmhlo.case"(%index) ({
    ^bb0:
      gpu.memcpy %operand_1, %operand_2 : memref<f32>, memref<f32>
      "lmhlo.terminator"() : () -> ()
    },  {
    ^bb0:
      gpu.memcpy %operand_2, %operand_1 : memref<f32>, memref<f32>
      "lmhlo.terminator"() : () -> ()
    }
  ) : (memref<i1>) -> ()
  func.return
}

// -----

// CHECK-DAG: func @[[branch_func0:[^(]+]]{{.*[[:space:]][ ]+}}gpu.memcpy %arg0, %arg1 : memref<f32>, memref<f32>
// CHECK-DAG: func @[[branch_func1:[^(]+]]{{.*[[:space:]][ ]+}}gpu.memcpy %arg1, %arg0 : memref<f32>, memref<f32>
// CHECK-DAG: func @[[branch_func2:[^(]+]]{{.*[[:space:]][ ]+}}gpu.memcpy %arg1, %arg1 : memref<f32>, memref<f32>

// CHECK:       func @case_i32(
// CHECK-SAME:    %arg0: memref<i32>,
// CHECK-SAME:    %arg1: memref<f32>,
// CHECK-SAME:    %arg2: memref<f32>
// CHECK-SAME:  ) {

func.func @case_i32(%index: memref<i32>, %operand_1: memref<f32>, %operand_2: memref<f32>) -> () {
  // CHECK:     %[[index:.*]] = memref.load %arg0[] : memref<i32>
  // CHECK-DAG: tfrt.case %[[index]] ["[[branch_func0]]", "[[branch_func1]]", "[[branch_func2]]"](%arg1, %arg2)
  "lmhlo.case"(%index) ({
    ^bb0:
      gpu.memcpy %operand_1, %operand_2 : memref<f32>, memref<f32>
      "lmhlo.terminator"() : () -> ()
    },  {
    ^bb0:
      gpu.memcpy %operand_2, %operand_1 : memref<f32>, memref<f32>
      "lmhlo.terminator"() : () -> ()
    },  {
    ^bb0:
      gpu.memcpy %operand_2, %operand_2 : memref<f32>, memref<f32>
      "lmhlo.terminator"() : () -> ()
    }
  ) : (memref<i32>) -> ()
  func.return
}
