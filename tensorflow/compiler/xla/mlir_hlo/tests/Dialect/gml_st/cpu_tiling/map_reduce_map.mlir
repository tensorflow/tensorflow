// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline \
// RUN: | FileCheck %s

func.func @reduce_map_fuse_map(%arg0: tensor<10x100xf32>,
    %arg1: tensor<10x100xf32>, %output: tensor<10xf32>) -> tensor<10xf32> {
  %map_init = tensor.empty() : tensor<10x100xf32>
  %reduce_init = tensor.empty() : tensor<10xf32>
  %mapped = linalg.map { arith.addf }
              ins(%arg0, %arg1 : tensor<10x100xf32>, tensor<10x100xf32>)
              outs(%map_init : tensor<10x100xf32>)

  %reduce = linalg.reduce { arith.addf }
              ins(%mapped: tensor<10x100xf32>)
              outs(%reduce_init: tensor<10xf32>)
              dimensions = [1]

  %res = linalg.map { math.absf }
           ins(%reduce: tensor<10xf32>)
           outs(%output : tensor<10xf32>)
  return %res : tensor<10xf32>
}
// CHECK-LABEL: @reduce_map_fuse_map

// TODO(pifon): The lowering is severely broken. Fixing it in a follow-up.
