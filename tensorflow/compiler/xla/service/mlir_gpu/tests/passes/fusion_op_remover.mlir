// RUN: xla-mlir-gpu-opt --mlir-gpu-fusion-op-remover %s | FileCheck %s

// CHECK-LABEL: func @fusion_memref
func @fusion_memref(%input1: memref<10xf32>, %input2: memref<10xf32>,
                   %input3: memref<10xf32>, %out: memref<10xf32>) -> () {
  // CHECK-NOT: lmhlo.fusion
  "lmhlo.fusion"() ( {
    %0 = tensor_load %input1 : memref<10xf32>
    %1 = tensor_load %input2 : memref<10xf32>
    %2 = "mhlo.add"(%0, %1) {name = "add"}
      : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %3 = tensor_load %input3 : memref<10xf32>
    %4 = "mhlo.multiply"(%2, %3) {name = "multiply"}
      : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    tensor_store %4, %out : memref<10xf32>
  // CHECK-NOT: lmhlo.terminator
    "lmhlo.terminator"() : () -> ()
  } ) : () -> ()
  return
}
