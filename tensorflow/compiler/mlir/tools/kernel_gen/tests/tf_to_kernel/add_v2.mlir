// RUN: tf_to_kernel --input=%s --output=%t --unroll_factors=4 --tile_sizes=256 --arch=sm_70,compute_75 --print-ir-after-all

func @AddV2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
    -> tensor<*xf32> attributes {tf_entry, llvm.emit_c_interface} {
  %0 = "tf.AddV2"(%arg0, %arg1) {T = f32, device = ""}
    : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// Check that the kernel has the expected format. This can be done best after
// it is in its final form.
// In particular, check that shape information was propagated and that
// TensorFlow abi information was propagated.
// CHECK-LABEL: IR Dump After PropagateShapeKnowledgeToKernels
// CHECK-LABEL: IR Dump After PropagateTfAbiKnowledgeToKernels
// CHECK-LABEL: IR Dump After StripDebugInfo

// The expectation is that all memref structures use the same shape, 0 offset
// and inner stride of 1. This is a kernel for 1-d.
// CHECK-LABEL gpu.module @AddV2_kernel_1
// CHECK: llvm.func @AddV2_kernel
// CHECK-DAG: {llvm.align = 16 : index}
// CHECK-DAG: {llvm.align = 16 : index}
// CHECK-DAG: {llvm.align = 16 : index, llvm.noalias = true}
// CHECK: attributes {gpu.kernel}
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK: %[[C1:.*]] llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[PTR0:.*]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[PTR0]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[SHAPE:.*]], %5[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[PTR1:.*]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[PTR1]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[SHAPE]], %5[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[PTR2:.*]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[PTR2]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[SHAPE]], %5[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
