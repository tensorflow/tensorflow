// RUN: mlir-hlo-opt %s -split-input-file -mhlo-expand-ops-simplifier | FileCheck %s

func.func @main(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = "mhlo.compare"(%arg3, %arg4) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %1 : tensor<10x24x24x64xf32>
}

// CHECK-LABEL:   func @main
// CHECK-SAME:        %[[OPERAND:.*]]: tensor<10x24x24x64xf32>,
// CHECK-SAME:        %[[SOURCE:.*]]: tensor<10x12x12x64xf32>
// CHECK-SAME:        -> tensor<10x24x24x64xf32>
// CHECK-DAG:       %[[NEG_1:.*]] = mhlo.constant dense<-1> : tensor<i64>
// CHECK-DAG:       %[[INIT:.*]] = mhlo.constant dense<0.000000e+00> : tensor<10x24x24x64xf32>
// CHECK-DAG:       %[[C0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[IOTA_0:.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<10x24x24x64xi64>
// CHECK:           %[[IOTA_1:.*]] = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<10x24x24x64xi64>
// CHECK:           %[[IOTA_2:.*]] = "mhlo.iota"() {iota_dimension = 2 : i64} : () -> tensor<10x24x24x64xi64>
// CHECK:           %[[IOTA_3:.*]] = "mhlo.iota"() {iota_dimension = 3 : i64} : () -> tensor<10x24x24x64xi64>
// CHECK:           %[[REDUCE_WINDOW:.*]]:5 = "mhlo.reduce_window"(%[[OPERAND]], %[[IOTA_0]], %[[IOTA_1]], %[[IOTA_2]], %[[IOTA_3]], %[[C0]], %[[NEG_1]], %[[NEG_1]], %[[NEG_1]], %[[NEG_1]]) ({
// CHECK:           ^bb0(%[[VAL_10:.*]]: tensor<f32>, %[[VAL_11:.*]]: tensor<i64>, %[[VAL_12:.*]]: tensor<i64>, %[[VAL_13:.*]]: tensor<i64>, %[[VAL_14:.*]]: tensor<i64>, %[[VAL_15:.*]]: tensor<f32>, %[[VAL_16:.*]]: tensor<i64>, %[[VAL_17:.*]]: tensor<i64>, %[[VAL_18:.*]]: tensor<i64>, %[[VAL_19:.*]]: tensor<i64>):
// CHECK:             %[[VAL_20:.*]] = mhlo.compare  NE, %[[VAL_11]], %[[NEG_1]]
// CHECK:             %[[VAL_21:.*]] = mhlo.compare  NE, %[[VAL_16]], %[[NEG_1]]
// CHECK:             %[[VAL_22:.*]] = mhlo.not %[[VAL_21]] : tensor<i1>
// CHECK:             %[[VAL_23:.*]] = mhlo.compare  GE, %[[VAL_10]], %[[VAL_15]]
// CHECK:             %[[VAL_24:.*]] = mhlo.and %[[VAL_23]], %[[VAL_20]] : tensor<i1>
// CHECK:             %[[VAL_25:.*]] = mhlo.or %[[VAL_24]], %[[VAL_22]] : tensor<i1>
// CHECK:             %[[SELECTED_0:.*]] = mhlo.select %[[VAL_25]], %[[VAL_10]], %[[VAL_15]]
// CHECK:             %[[SELECTED_1:.*]] = mhlo.select %[[VAL_25]], %[[VAL_11]], %[[VAL_16]]
// CHECK:             %[[SELECTED_2:.*]] = mhlo.select %[[VAL_25]], %[[VAL_12]], %[[VAL_17]]
// CHECK:             %[[SELECTED_3:.*]] = mhlo.select %[[VAL_25]], %[[VAL_13]], %[[VAL_18]]
// CHECK:             %[[SELECTED_4:.*]] = mhlo.select %[[VAL_25]], %[[VAL_14]], %[[VAL_19]]
// CHECK:             mhlo.return %[[SELECTED_0]], %[[SELECTED_1]], %[[SELECTED_2]], %[[SELECTED_3]], %[[SELECTED_4]]
// CHECK:           }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<10x24x24x64xf32>, tensor<10x24x24x64xi64>, tensor<10x24x24x64xi64>, tensor<10x24x24x64xi64>, tensor<10x24x24x64xi64>, tensor<f32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> (tensor<10x12x12x64xf32>, tensor<10x12x12x64xi64>, tensor<10x12x12x64xi64>, tensor<10x12x12x64xi64>, tensor<10x12x12x64xi64>)
// CHECK:           %[[RESHAPE_0:.*]] = mhlo.reshape %[[REDUCE_WINDOW]]#1 : (tensor<10x12x12x64xi64>) -> tensor<10x12x12x64x1xi64>
// CHECK:           %[[RESHAPE_1:.*]] = mhlo.reshape %[[REDUCE_WINDOW]]#2 : (tensor<10x12x12x64xi64>) -> tensor<10x12x12x64x1xi64>
// CHECK:           %[[RESHAPE_2:.*]] = mhlo.reshape %[[REDUCE_WINDOW]]#3 : (tensor<10x12x12x64xi64>) -> tensor<10x12x12x64x1xi64>
// CHECK:           %[[RESHAPE_3:.*]] = mhlo.reshape %[[REDUCE_WINDOW]]#4 : (tensor<10x12x12x64xi64>) -> tensor<10x12x12x64x1xi64>
// CHECK:           %[[CONCAT:.*]] = "mhlo.concatenate"(%[[RESHAPE_0]], %[[RESHAPE_1]], %[[RESHAPE_2]], %[[RESHAPE_3]]) {dimension = 4 : i64}
// CHECK:           %[[SCATTER:.*]] = "mhlo.scatter"(%[[INIT]], %[[CONCAT]], %[[SOURCE]]) ({
// CHECK:           ^bb0(%[[VAL_38:.*]]: tensor<f32>, %[[VAL_39:.*]]: tensor<f32>):
// CHECK:             %[[UPDATE:.*]] = mhlo.add %[[VAL_38]], %[[VAL_39]] : tensor<f32>
// CHECK:             mhlo.return %[[UPDATE]] : tensor<f32>
// CHECK:           }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false} : (tensor<10x24x24x64xf32>, tensor<10x12x12x64x4xi64>, tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32>
// CHECK:           return %[[SCATTER]] : tensor<10x24x24x64xf32>
// CHECK:         }