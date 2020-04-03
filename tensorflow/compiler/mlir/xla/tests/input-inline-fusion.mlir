// RUN: bazel-bin/tensorflow/compiler/mlir/tf-opt --lhlo-legalize-to-loops --input-inline-fusion %s

func @main() {
  %c4 = constant 1024 : index
  %c5 = constant 1024 : index
  %0 = alloc(%c4, %c5) : memref<?x?xf32>
  %1 = alloc(%c4, %c5) : memref<?x?xf32>
  %2 = alloc() : memref<f32>
  %3:2 = call @lowered_tao_main(%0, %1, %2) : (memref<?x?xf32>, memref<?x?xf32>, memref<f32>) -> (memref<?xf32>, memref<?x?xf32>)
  return
}
func @lowered_dhlo_fusion_xla_hlo_reduce(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<f32>, %arg3: memref<?x?xf32>, %arg4: memref<?xf32>) attributes {xla_dhlo.memref_func} {
  %0 = dim %arg0, 0 : memref<?x?xf32>
  %1 = dim %arg0, 1 : memref<?x?xf32>
  %2 = dim %arg1, 0 : memref<?x?xf32>
  %3 = dim %arg1, 1 : memref<?x?xf32>
  %4 = dim %arg3, 0 : memref<?x?xf32>
  %5 = dim %arg3, 1 : memref<?x?xf32>
  %6 = dim %arg4, 0 : memref<?xf32>
  %tmp0 = alloc(%0, %1) : memref<?x?xf32>
  %tmp1 = alloc(%0, %1) : memref<?x?xf32>
  "xla_lhlo.multiply"(%arg0, %arg1, %tmp0) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  "xla_lhlo.divide"(%arg0, %tmp0, %tmp1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  "xla_lhlo.add"(%tmp0, %tmp1, %arg3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  "xla_lhlo.reduce"(%arg3, %arg2, %arg4) ( {
  ^bb0(%arg5: memref<f32>, %arg6: memref<f32>, %arg7: memref<f32>):	// no predecessors
    "xla_lhlo.add"(%arg5, %arg6, %arg7) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "xla_lhlo.terminator"() : () -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
  return
}
func @lowered_tao_main(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<f32>) -> (memref<?xf32>, memref<?x?xf32>) {
  %0 = dim %arg0, 0 : memref<?x?xf32>
  %1 = dim %arg0, 1 : memref<?x?xf32>
  %2 = dim %arg1, 0 : memref<?x?xf32>
  %3 = dim %arg1, 1 : memref<?x?xf32>
  %4 = alloc(%0, %1) : memref<?x?xf32>
  %5 = alloc(%1) : memref<?xf32>
  call @lowered_dhlo_fusion_xla_hlo_reduce(%arg0, %arg1, %arg2, %4, %5) : (memref<?x?xf32>, memref<?x?xf32>, memref<f32>, memref<?x?xf32>, memref<?xf32>) -> ()
  return %5, %4 : memref<?xf32>, memref<?x?xf32>
}
