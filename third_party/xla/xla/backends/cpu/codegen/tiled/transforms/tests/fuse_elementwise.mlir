// RUN: fusion_compiler_opt %s -split-input-file  \
// RUN:   -linalg-generalize-named-ops -xtile-cpu-fuse-elementwise | FileCheck %s

func.func @elementwise_add_to_vector(
    %lhs : tensor<8x1024xf32>,
    %rhs : tensor<8x1024xf32>) -> tensor<8x1024xf32> {
  %out = tensor.empty() : tensor<8x1024xf32>

  %intermediate = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%lhs, %rhs : tensor<8x1024xf32>, tensor<8x1024xf32>)
    outs(%out : tensor<8x1024xf32>) -> tensor<8x1024xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%intermediate, %rhs : tensor<8x1024xf32>, tensor<8x1024xf32>)
    outs(%out : tensor<8x1024xf32>) -> tensor<8x1024xf32>
  return %result : tensor<8x1024xf32>
}

// CHECK: linalg.generic
// CHECK:    (%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:       %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK:       %[[RES:.*]] = arith.addf %[[MUL]], %[[RHS]] : f32
// CHECK:       linalg.yield %[[RES]] : f32
// CHECK:     }
