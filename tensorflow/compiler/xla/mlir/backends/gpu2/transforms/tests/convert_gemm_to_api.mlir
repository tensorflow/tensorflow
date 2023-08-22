// RUN: export MSAN_OPTIONS=intercept_strpbrk=0
// RUN: xla-gpu2-opt %s --xla-gpu2-convert-to-runtime=backend=streamexecutor   \
// RUN:                 --split-input-file                                     \
// RUN:   | FileCheck %s

#loc = loc("custom-call")

func.func @gemm(
    %arg0: memref<128xi8>, %arg1: memref<64xi8>,
    %arg2: memref<32xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}
) {
  %c0 = arith.constant 0 : index
  %view0 = memref.view %arg0[%c0][] : memref<128xi8> to memref<4x8xf32>
  %view1 = memref.view %arg1[%c0][] : memref<64xi8> to memref<8x2xf32>
  %view2 = memref.view %arg2[%c0][] : memref<32xi8> to memref<4x2xf32>

  "lmhlo_gpu.gemm"(%view0, %view1, %view2) {
    alpha_imag = 0.000000e+00 : f64,
    alpha_real = 1.000000e+00 : f64,
    beta = 0.000000e+00 : f64,
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0],
                                      rhs_contracting_dimensions = [0]>,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (memref<4x8xf32>, memref<8x2xf32>, memref<4x2xf32>) -> () loc(#loc)

  return
}

// CHECK-LABEL: func @gemm(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<128xi8>,
// CHECK:   %[[ARG1:.*]]: tensor<64xi8>,
// CHECK:   %[[ARG2:.*]]: tensor<32xi8> {lmhlo.output_index = {{.*}}}
// CHECK: ) {

// CHECK-DAG: %[[BUFFER0:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK-DAG: %[[BUFFER1:.*]] = iree_input.tensor.export %[[ARG1]]
// CHECK-DAG: %[[BUFFER2:.*]] = iree_input.tensor.export %[[ARG2]]

// CHECK-DAG: %[[LHS:.*]] = iree_input.tensor.import %[[BUFFER0]]
// CHECK-DAG: %[[RHS:.*]] = iree_input.tensor.import %[[BUFFER1]]
// CHECK-DAG: %[[OUT:.*]] = iree_input.tensor.import %[[BUFFER2]]

// CHECK:   %[[DIMS:.*]] = call @xla_gpu.dot_dimension_numbers.create
// CHECK:   %[[PRECISION:.*]] = call @xla_gpu.dot_precision.create
// CHECK:   %[[CONFIG:.*]] = call @xla_gpu.dot_config.create
// CHECK:   %[[HLO:.*]] = iree_input.byte_buffer.constant {{.*}} = "custom-call"
// CHECK:   %[[TRACE:.*]] = call @xla_gpu.trace.create(%[[HLO]])

// CHECK-DAG: %[[LHS_VIEW:.*]] = iree_input.tensor.export %[[LHS]]
// CHECK-DAG: %[[RHS_VIEW:.*]] = iree_input.tensor.export %[[RHS]]
// CHECK-DAG: %[[OUT_VIEW:.*]] = iree_input.tensor.export %[[OUT]]

// CHECK:   call @xla_gpu.gemm.dispatch(
// CHECK:     %[[CTX]], %[[LHS_VIEW]], %[[RHS_VIEW]], %[[OUT_VIEW]],
// CHECK:     %[[CONFIG]], %[[TRACE]]
// CHECK:   )
// CHECK: }
