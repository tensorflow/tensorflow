// RUN: tf_to_kernel --input=%s --output=%t --unroll_factors=4 --tile_sizes=256 --arch=sm_70,compute_75 --print-ir-after-all

func @tanh(%arg: tensor<*xf32>) -> tensor<*xf32> attributes {tf_entry} {
  %0 = "tf.Tanh"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// Make sure that reuse-analysis found the reuse case.
// CHECK-LABEL: IR Dump After BufferReusePass
// CHECK: func @AddV2_f16
// CHECK:   alloc(%{{[0-9]+}})
// CHECK-SAME: {reuse_input_candidates = [0 : index], reuse_output = -1 : index}
// CHECK: return

// Make sure we do not have stray heap allocations.
// CHECK-LABEL: IR Dump After PromoteBuffersToStack
// CHECK: func @tanh
// CHECK: alloca
// CHECK-NOT: alloca
// CHECK: alloc
// CHECK-NOT: alloc
// CHECK: return

// Check that the kernel has the expected format. This can be done best after
// it is in its final form.
// In particular, check that shape information was propagated and that
// TensorFlow abi information was propagated.
// CHECK-LABEL: IR Dump After PropagateShapeKnowledgeToKernels
// CHECK-LABEL: IR Dump After PropagateTfAbiKnowledgeToKernels
// CHECK-LABEL: IR Dump After StripDebugInfo
// CHECK: gpu.module @tanh_kernel {
// CHECK:   llvm.func @tanh_kernel(
// CHECK-SAME: %[[PRT_ARG0]]: !llvm.ptr<float> {llvm.align = 16 : index}
// CHECK-SAME: %[[PRT_ARG1]]: !llvm.ptr<float> {llvm.align = 16 : index, llvm.noalias = true}
// CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   %[[C1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:   %[[ARG0_0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG0_1:.*]] = llvm.insertvalue %[[PTR_ARG0]], %[[ARG0_0]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG0_2:.*]] = llvm.insertvalue %[[PTR_ARG0]], %[[ARG0_1]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG0_3:.*]] = llvm.insertvalue %[[C0]], %[[ARG0_2]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG0_4:.*]] = llvm.insertvalue %[[SHAPE:.*]], %[[ARG0_3]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG0_5:.*]] = llvm.insertvalue %[[C1]], %[[ARG0_4]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG1_0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG1_1:.*]] = llvm.insertvalue %[[PTR_ARG1]], %[[ARG1_0]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG1_2:.*]] = llvm.insertvalue %[[PTR_ARG1]], %[[ARG1_1]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG1_3:.*]] = llvm.insertvalue %[[C0]], %[[ARG1_2]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG1_4:.*]] = llvm.insertvalue %[[SHAPE]], %[[ARG1_3]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:   %[[ARG1_5:.*]] = llvm.insertvalue %[[C1]], %[[ARG1_4]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: llvm.return
