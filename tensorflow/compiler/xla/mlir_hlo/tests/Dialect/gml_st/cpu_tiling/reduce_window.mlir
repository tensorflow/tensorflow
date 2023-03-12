// RUN: mlir-hlo-opt %s --gml-st-cpu-tiling-pipeline
// TODO(b/270534416): Re-enable.
// | FileCheck %s

func.func @reduce_window(%input: tensor<1xf32>, %window: tensor<32xf32>,
                  %output: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %bcast_init = tensor.empty() : tensor<1x256xf32>
  %bcast = linalg.broadcast
             ins(%input : tensor<1xf32>)
             outs(%bcast_init : tensor<1x256xf32>)
             dimensions = [1]

  %abs_init = tensor.empty() : tensor<32xf32>
  %abs = linalg.map { math.absf }
           ins(%window: tensor<32xf32>)
           outs(%abs_init: tensor<32xf32>)

  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x8xf32>
  %fill = linalg.fill
    ins(%cst : f32) outs(%init : tensor<1x8xf32>) -> tensor<1x8xf32>

  %reduce_window = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1 * 32 + d2)>,
      affine_map<(d0, d1, d2) -> (d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%bcast, %abs : tensor<1x256xf32>, tensor<32xf32>)
    outs(%fill : tensor<1x8xf32>) {
  ^bb0(%in: f32, %win: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<1x8xf32>


  %exp = linalg.map { math.exp }
           ins(%reduce_window: tensor<1x8xf32>)
           outs(%init: tensor<1x8xf32>)

  func.return  %exp : tensor<1x8xf32>
}
// CHECK-LABEL: @reduce_window

// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           arith.addf {{.*}} : f32
// CHECK:           scf.yield %{{.*}} : f32
// CHECK:         math.exp %{{.*}} : f32
// CHECK:         tensor.parallel_insert_slice
