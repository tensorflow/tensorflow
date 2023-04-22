// RUN: mlir-hlo-opt %s -lhlo-legalize-roots-to-parallel-loops -split-input-file | FileCheck %s

// CHECK-LABEL: @non_fusion_elemwise_gpu
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?x?xf32, "gpu">, %[[INPUT2:.*]]: memref<?x?x?xf32, "gpu">, %[[OUT:.*]]: memref<?x?x?xf32, "gpu">) -> memref<?x?x?xf32, "gpu">
func @non_fusion_elemwise_gpu(%input1: memref<?x?x?xf32, "gpu">, %input2: memref<?x?x?xf32, "gpu">, %out: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">) {
  // CHECK-NOT: lmhlo
  // CHECK: scf.parallel
  "lmhlo.add"(%input1, %input2, %out) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32, "gpu">
  return %out : memref<?x?x?xf32, "gpu">
}

// CHECK-LABEL: @non_fusion_elemwise_cpu
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?x?x?xf32>, %[[INPUT2:.*]]: memref<?x?x?xf32>, %[[OUT:.*]]: memref<?x?x?xf32>) -> memref<?x?x?xf32>
func @non_fusion_elemwise_cpu(%input1: memref<?x?x?xf32>, %input2: memref<?x?x?xf32>, %out: memref<?x?x?xf32>) -> (memref<?x?x?xf32>) {
  // CHECK-NOT lmhlo
  // CHECK: scf.for
  "lmhlo.add"(%input1, %input2, %out) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32>
  return %out : memref<?x?x?xf32>
}

// CHECK-LABEL: @non_fusion_dynamic_broadcast_in_dim_gpu
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32, "gpu">, %[[INPUT2:.*]]: memref<3xi32>, %[[OUT:.*]]: memref<?x?x?xf32, "gpu">) -> memref<?x?x?xf32, "gpu">
func @non_fusion_dynamic_broadcast_in_dim_gpu(%input1: memref<?xf32, "gpu">, %input2: memref<3xi32>, %out: memref<?x?x?xf32, "gpu">) -> (memref<?x?x?xf32, "gpu">) {
  // CHECK-NOT lmhlo
  // CHECK: scf.parallel
  "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %out) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32, "gpu">, memref<3xi32>, memref<?x?x?xf32, "gpu">) -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32, "gpu">
  return %out : memref<?x?x?xf32, "gpu">
}

// CHECK-LABEL: @basic_loop_fusion_misc_root
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<?xf32>, %[[INPUT3:.*]]: memref<3xi32>, %[[TMP_BUF:.*]]: memref<?xf32>, %[[OUT:.*]]: memref<?x?x?xf32>) -> memref<?x?x?xf32>
func @basic_loop_fusion_misc_root(%input1: memref<?xf32>, %input2: memref<?xf32>, %input3: memref<3xi32>, %tmp: memref<?xf32>, %out: memref<?x?x?xf32>) -> (memref<?x?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  "lmhlo.fusion"() ( {
    // CHECK: lmhlo.add
    // CHECK-NOT lmhlo.dynamic_broadcast_in_dim
    // CHECK: scf.parallel
    "lmhlo.add"(%input1, %input2, %tmp) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%tmp, %input3, %out) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  } ) : () -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32>
  return %out : memref<?x?x?xf32>
}

// CHECK-LABEL: @multioutput_loop_fusion_with_dependency
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<3xi32>, %[[INPUT3:.*]]: memref<?x?x?xf32>, %[[TMP_BUF:.*]]: memref<?x?x?xf32>, %[[OUT1:.*]]: memref<?x?x?xf32>, %[[OUT2:.*]]: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>)
func @multioutput_loop_fusion_with_dependency(%input1: memref<?xf32>, %input2: memref<3xi32>, %input3: memref<?x?x?xf32>, %tmp: memref<?x?x?xf32>, %out_1: memref<?x?x?xf32>, %out_2: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  "lmhlo.fusion"() ( {
    // CHECK: lmhlo.dynamic_broadcast_in_dim
    // CHECK: lmhlo.add
    // CHECK-NOT: lmhlo.multiply
    // CHECK: scf.parallel
    "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %tmp) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%input3, %tmp, %out_1) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    "lmhlo.multiply"(%input3, %out_1, %out_2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?x?xf32>, memref<?x?x?xf32>
  return %out_1, %out_2 : memref<?x?x?xf32>, memref<?x?x?xf32>
}

// CHECK-LABEL: @multioutput_loop_fusion_without_dependency
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<3xi32>, %[[INPUT3:.*]]: memref<?x?x?xf32>, %[[TMP_BUF:.*]]: memref<?x?x?xf32>, %[[OUT1:.*]]: memref<?x?x?xf32>, %[[OUT2:.*]]: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>)
func @multioutput_loop_fusion_without_dependency(%input1: memref<?xf32>, %input2: memref<3xi32>, %input3: memref<?x?x?xf32>, %tmp: memref<?x?x?xf32>, %out_1: memref<?x?x?xf32>, %out_2: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>) {
  // CHECK: "lmhlo.fusion"() ( {
  "lmhlo.fusion"() ( {
    // CHECK: lmhlo.dynamic_broadcast_in_dim
    // CHECK-NOT: lmhlo.add
    // CHECK-NOT: lmhlo.multiply
    // CHECK: scf.parallel
    "lmhlo.dynamic_broadcast_in_dim"(%input1, %input2, %tmp) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%input3, %tmp, %out_1) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    "lmhlo.multiply"(%input3, %tmp, %out_2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?x?xf32>, memref<?x?x?xf32>
  return %out_1, %out_2 : memref<?x?x?xf32>, memref<?x?x?xf32>
}
