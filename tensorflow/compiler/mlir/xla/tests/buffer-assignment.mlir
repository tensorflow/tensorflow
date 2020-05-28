// RUN: tf-opt -test-buffer-assignment -allow-unregistered-dialect -split-input-file %s | FileCheck %s -dump-input-on-failure

// CHECK-LABEL: func @func_signature_conversion
func @func_signature_conversion(%arg0: tensor<4x8xf32>) {
    return
}
// CHECK: ({{.*}}: memref<4x8xf32>) {

// -----

// CHECK-LABEL: func @non_void_to_void_return_op_converter
func @non_void_to_void_return_op_converter(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %arg0 : tensor<4x8xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]<[[RANK:.*]]>, %[[RESULT:.*]]: [[TYPE]]<[[RANK]]>) {
// CHECK-NEXT: "buffer_assignment_test.copy"(%[[ARG0]], %[[RESULT]])
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @func_and_block_signature_conversion
func @func_and_block_signature_conversion(%arg0 : tensor<2xf32>, %cond : i1, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>{
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : tensor<2xf32>)
  ^bb2:
    br ^exit(%arg0 : tensor<2xf32>)
  ^exit(%arg2: tensor<2xf32>):
    return %arg1 : tensor<4x4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[ARG0_TYPE:.*]], %[[COND:.*]]: i1, %[[ARG1:.*]]: [[ARG1_TYPE:.*]], %[[RESULT:.*]]: [[RESULT_TYPE:.*]]) {
//      CHECK: br ^[[EXIT_BLOCK:.*]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: br ^[[EXIT_BLOCK]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: ^[[EXIT_BLOCK]](%{{.*}}: [[ARG0_TYPE]])
// CHECK-NEXT: "buffer_assignment_test.copy"(%[[ARG1]], %[[RESULT]])
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @condBranch
func @condBranch(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : tensor<2xf32>)
  ^bb2:
    %1 = "buffer_assignment_test.unary"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    br ^exit(%1 : tensor<2xf32>)
  ^exit(%arg1: tensor<2xf32>):
    return %arg1 : tensor<2xf32>

}
// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: cond_br
//      CHECK: "buffer_assignment_test.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @emptyUsesValue
func @emptyUsesValue(%arg0: memref<4xf32>) {
  %0 = alloc() : memref<4xf32>
  return
}
// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @criticalEdge
func @criticalEdge(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
    cond_br %cond, ^bb1, ^exit(%arg0 : tensor<2xf32>)
  ^bb1:
    %0 = "buffer_assignment_test.unary"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    br ^exit(%0 : tensor<2xf32>)
  ^exit(%arg1: tensor<2xf32>):
    return %arg1 : tensor<2xf32>
}
// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: cond_br
//      CHECK: "buffer_assignment_test.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @invCriticalEdge
func @invCriticalEdge(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
    %0 = "buffer_assignment_test.unary"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1, ^exit(%arg0 : tensor<2xf32>)
  ^bb1:
    br ^exit(%0 : tensor<2xf32>)
  ^exit(%arg1: tensor<2xf32>):
    return %arg1 : tensor<2xf32>
}
// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
//      CHECK: "buffer_assignment_test.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @ifElse
func @ifElse(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
    %0 = "buffer_assignment_test.unary"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    br ^exit(%arg3, %arg4 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
    %1 = "buffer_assignment_test.unary"(%arg5) : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
}
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
// CHECK-NEXT: dealloc %[[FIRST_ALLOC]]
// CHECK-NEXT: "buffer_assignment_test.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @ifElseNoUsers
func @ifElseNoUsers(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
    %0 = "buffer_assignment_test.unary"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    br ^exit(%arg3, %arg4 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
    return %arg0 : tensor<2xf32>
}
// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
//      CHECK: "buffer_assignment_test.copy
// CHECK-NEXT: dealloc %[[ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @ifElseNested
func @ifElseNested(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
    %0 = "buffer_assignment_test.unary"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    cond_br %cond, ^bb3(%arg3 : tensor<2xf32>), ^bb4(%arg4 : tensor<2xf32>)
  ^bb3(%arg7 : tensor<2xf32>):
    br ^exit(%arg7, %arg3 : tensor<2xf32>, tensor<2xf32>)
  ^bb4(%arg8 : tensor<2xf32>):
    br ^exit(%arg3, %arg8 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
    %1 = "buffer_assignment_test.unary"(%arg5) : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
}
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
// CHECK-NEXT: dealloc %[[FIRST_ALLOC]]
// CHECK-NEXT: "buffer_assignment_test.copy
// CHECK-NEXT: dealloc %[[SECOND_ALLOC]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @redundantOperations
func @redundantOperations(%arg0: tensor<4xf32>) {
  %1 = "buffer_assignment_test.unary"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "buffer_assignment_test.unary"(%1) : (tensor<4xf32>) -> tensor<4xf32>
  return
}
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
// CHECK-NEXT: %[[SECOND_ALLOC:.*]] = alloc()
// CHECK-NEXT: "buffer_assignment_test.unary_lowered"
// CHECK-NEXT: dealloc
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
func @moving_alloc_and_inserting_missing_dealloc(%cond : i1, %arg0 : memref<2xf32>, %arg1: memref<2xf32>){
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %0 = alloc() : memref<2xf32>
    "buffer_assignment_test.unary_lowered"(%arg0, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    br ^exit(%0 : memref<2xf32>)
  ^bb2:

    %1 = alloc() : memref<2xf32>
    "buffer_assignment_test.unary_lowered"(%arg0, %1) : (memref<2xf32>, memref<2xf32>) -> ()
    br ^exit(%1 : memref<2xf32>)
  ^exit(%arg2: memref<2xf32>):
    "buffer_assignment_test.copy"(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
// CHECK-NEXT: %[[FIRST_ALLOC:.*]] = alloc()
// CHECK-NEXT: %[[SECOND_ALLOC:.*]] = alloc()
//      CHECK: "buffer_assignment_test.copy"
// CHECK-NEXT: dealloc
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
    "buffer_assignment_test.unary_lowered"(%arg0, %1) : (memref<2xf32>, memref<2xf32>) -> ()
    dealloc %1 : memref<2xf32>
    br ^exit(%1 : memref<2xf32>)
  ^exit(%arg2: memref<2xf32>):
    "buffer_assignment_test.copy"(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
// CHECK-NEXT: %[[ALLOC:.*]] = alloc()
//      CHECK: buffer_assignment_test.copy
// CHECK-NEXT: dealloc
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @inserting_missing_dealloc_simple
func @inserting_missing_dealloc_simple(%arg0 : memref<2xf32>, %arg1: memref<2xf32>){
    %0 = alloc() : memref<2xf32>
    "buffer_assignment_test.unary_lowered"(%arg0, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    "buffer_assignment_test.copy"(%0, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
//      CHECK: buffer_assignment_test.copy
// CHECK-NEXT: dealloc

// -----

// CHECK-LABEL: func @moving_invalid_dealloc_op
func @moving_invalid_dealloc_op(%arg0 : memref<2xf32>, %arg1: memref<2xf32>){
    %0 = alloc() : memref<2xf32>
    "buffer_assignment_test.unary_lowered"(%arg0, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    dealloc %0 : memref<2xf32>
    "buffer_assignment_test.copy"(%0, %arg1) : (memref<2xf32>, memref<2xf32>) -> ()
    return
}
//      CHECK: buffer_assignment_test.copy
// CHECK-NEXT: dealloc
