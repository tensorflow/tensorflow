// RUN: mlir-hlo-opt %s --gml-st-cpu-tiling-pipeline | FileCheck %s

func.func @fuse_reshape_map(%arg0: tensor<10x16xf32>,
    %arg1: tensor<10x16xf32>) -> tensor<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = tensor.empty() : tensor<10x1x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]] : tensor<10x1x1x1xf32> into tensor<10x1xf32>

  %empty= tensor.empty() : tensor<10x1x4x4x1xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3, 4]] :
              tensor<10x16xf32> into tensor<10x1x4x4x1xf32>
  %neg = linalg.map { arith.negf }
         ins(%expanded: tensor<10x1x4x4x1xf32>)
         outs(%empty: tensor<10x1x4x4x1xf32>)
  %collapsed = tensor.collapse_shape %neg [[0, 1], [2, 3, 4]] :
              tensor<10x1x4x4x1xf32> into tensor<10x16xf32>

  %empty_3D = tensor.empty() : tensor<10x1x16xf32>
  %expanded0 = tensor.expand_shape %collapsed [[0], [1, 2]] :
              tensor<10x16xf32> into tensor<10x1x16xf32>
  %abs0 = linalg.map { math.absf }
         ins(%expanded0: tensor<10x1x16xf32>)
         outs(%empty_3D : tensor<10x1x16xf32>)
  %collapsed0 = tensor.collapse_shape %abs0 [[0], [1, 2]] :
               tensor<10x1x16xf32> into tensor<10x16xf32>

  %empty_5D = tensor.empty() : tensor<10x16x1x1x1xf32>
  %expanded1 = tensor.expand_shape %collapsed0 [[0], [1, 2, 3, 4]] :
               tensor<10x16xf32> into tensor<10x16x1x1x1xf32>
  %abs1 = linalg.map { math.absf }
          ins(%expanded1: tensor<10x16x1x1x1xf32>)
          outs(%empty_5D : tensor<10x16x1x1x1xf32>)
  %collapsed1 = tensor.collapse_shape %abs1 [[0], [1, 2, 3, 4]] :
                tensor<10x16x1x1x1xf32> into tensor<10x16xf32>

  %empty_4D = tensor.empty() : tensor<10x1x16x1xf32>
  %expanded2 = tensor.expand_shape %collapsed1 [[0, 1], [2, 3]] :
              tensor<10x16xf32> into tensor<10x1x16x1xf32>
  %abs2 = linalg.map { math.absf }
         ins(%expanded2: tensor<10x1x16x1xf32>)
         outs(%empty_4D : tensor<10x1x16x1xf32>)
  %collapsed2 = tensor.collapse_shape %abs2 [[0, 1], [2, 3]] :
              tensor<10x1x16x1xf32> into tensor<10x16xf32>

  %empty_2D = tensor.empty() : tensor<10x16xf32>
  %add = linalg.map { arith.addf }
              ins(%collapsed2, %arg1 : tensor<10x16xf32>, tensor<10x16xf32>)
              outs(%empty_2D : tensor<10x16xf32>)
  return %add : tensor<10x16xf32>
}

// CHECK:       @fuse_reshape_map(%[[ARG0:.*]]: tensor<10x16xf32>, %[[ARG1:.*]]: tensor<10x16xf32>)
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[ARG0]] {{.*}} tensor<10x16xf32> into tensor<10x1x4x4x1xf32>
// CHECK:         %[[RES:.*]] = scf.for {{.*}} (tensor<10x1x4x4x1xf32>) {
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               %[[EXTRACT:.*]] = tensor.extract_slice %[[EXPAND]]
// CHECK:               arith.negf
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[RES]] {{.*}} tensor<10x1x4x4x1xf32> into tensor<10x16xf32>

// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[EXTRACT0:.*]] = tensor.extract_slice %[[COLLAPSE]]
// CHECK:             %[[EXPAND0:.*]] = tensor.expand_shape %[[EXTRACT0]] {{.*}} tensor<1x8xf32> into tensor<1x8x1xf32>
// CHECK:             %[[READ0:.*]] = vector.transfer_read %[[EXPAND0]]
// CHECK:             %[[ABS0:.*]] = math.absf %[[READ0]]
// CHECK:             %[[WRITE0:.*]] = vector.transfer_write %[[ABS0]]
// CHECK:             %[[COLLAPSE0:.*]] = tensor.collapse_shape %[[WRITE0]] {{.*}} tensor<1x8x1xf32> into tensor<1x8xf32>

// CHECK:             %[[EXPAND1:.*]] = tensor.expand_shape %[[COLLAPSE0]] {{.*}} tensor<1x8xf32> into tensor<1x8x1x1x1xf32>
// CHECK:             %[[READ1:.*]] = vector.transfer_read %[[EXPAND1]]
// CHECK:             %[[ABS1:.*]] = math.absf %[[READ1]]
// CHECK:             %[[WRITE1:.*]] = vector.transfer_write %[[ABS1]]
// CHECK:             %[[COLLAPSE1:.*]] = tensor.collapse_shape %[[WRITE1]] {{.*}} tensor<1x8x1x1x1xf32> into tensor<1x8xf32>

// CHECK:             %[[EXPAND2:.*]] = tensor.expand_shape %[[COLLAPSE1]] {{.*}} tensor<1x8xf32> into tensor<1x1x8x1xf32>
// CHECK:             %[[READ2:.*]] = vector.transfer_read %[[EXPAND2]]
// CHECK:             %[[ABS2:.*]] = math.absf %[[READ2]]
// CHECK:             %[[WRITE2:.*]] = vector.transfer_write %[[ABS2]]
// CHECK:             %[[COLLAPSE2:.*]] = tensor.collapse_shape %[[WRITE2]] {{.*}} tensor<1x1x8x1xf32> into tensor<1x8xf32>

// CHECK:             %[[READ1:.*]] = vector.transfer_read %[[COLLAPSE2]]
// CHECK:             %[[READ2:.*]] = vector.transfer_read %[[ARG1]]
// CHECK:             %[[ADD:.*]] = arith.addf %[[READ1]], %[[READ2]]
// CHECK:             vector.transfer_write %[[ADD]]
