// RUN: mlir-hlo-opt %s \
// RUN:     --gml-st-cpu-tiling-pipeline="enable-fusion-clusters=true enable-fusion-cluster-outlining=true"

func.func @reduce_wo_init(%arg0: tensor<2xf64>, %arg1: tensor<f64>)
    -> tensor<f64> {
  %reduced = linalg.reduce { arith.maxf } ins(%arg0 : tensor<2xf64>)
      outs(%arg1 : tensor<f64>) dimensions = [0]
  return %reduced : tensor<f64>
}

// CHECK: @reduce_wo_init_fusion_0
// CHECK: @reduce_wo_init
