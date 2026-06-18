// RUN: emitters_opt %s -split-input-file -xla-cpu-add-loop-unroll-flags | FileCheck %s

func.func @nested_for(%arg : tensor<16x16x8xf32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  scf.for %iter0 = %c0 to %c16 step %c1 iter_args(%res0 = %arg) -> tensor<16x16x8xf32> {
    scf.for %iter1 = %c0 to %c16 step %c1 iter_args(%res1 = %res0) -> tensor<16x16x8xf32> {
      scf.for %iter2 = %c0 to %c8 step %c1 iter_args(%res2 = %res1) -> tensor<16x16x8xf32> {
        %extracted = tensor.extract %res2[%iter0, %iter1, %iter2] : tensor<16x16x8xf32>
        scf.yield %res2 : tensor<16x16x8xf32>
      }
      scf.yield %res1 : tensor<16x16x8xf32>
    }
    scf.for %iter1 = %c0 to %c8 step %c1  iter_args(%res1 = %res0) -> tensor<16x16x8xf32> {
      %extracted = tensor.extract %res1[%iter0, %iter0, %iter1] : tensor<16x16x8xf32>
      scf.yield %res1 : tensor<16x16x8xf32>
    }
    scf.yield %res0 : tensor<16x16x8xf32>
  }
  return
}

// CHECK: #[[LOOP_UNROLL:.*]] = #llvm.loop_unroll<disable = true>
// CHECK: #[[LOOP_ANNOTATION:.*]] = #llvm.loop_annotation<unroll = #[[LOOP_UNROLL]]>
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK: tensor.extract
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NOT: loop_annotation
// CHECK: scf.yield
// CHECK-NEXT: } {loop_annotation = #[[LOOP_ANNOTATION]]}
// CHECK-NEXT: scf.for
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: scf.yield
// CHECK-NEXT }
// CHECK-NOT: loop_annotation
// CHECK: scf.yield
// CHECK-NEXT: } {loop_annotation = #[[LOOP_ANNOTATION]]}
// CHECK-NEXT: return
