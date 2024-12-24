// RUN: tf_tfl_translate --enable-hlo-to-tf-conversion --input-mlir %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s --check-prefix=CHECK-ROUNDTRIP


module {
  // CHECK-LABEL: func.func public @main
  func.func public @main(%arg0: tensor<3x2x4x7x9xi32>, %arg1: tensor<4x3x5x2xi32>, %arg2: tensor<4x3x5x8xi32>) -> tensor<3x2x4x7x9xi32> {
    // CHECK-ROUNDTRIP:       %[[iota_1:.*]] = "tfl.pseudo_const"() <{{.*}}> : () -> tensor<4x3x5x1xi32
    // CHECK-ROUNDTRIP:       %[[iota_2:.*]] = "tfl.pseudo_const"() <{{.*}}> : () -> tensor<4x3x5x1xi32>
    // CHECK-ROUNDTRIP:       %[[concat:.*]] = "tfl.concatenation"(%[[iota_1]], %[[iota_2]], %arg1) <{axis = 3 : i32, fused_activation_function = "NONE"}> :
    // CHECK-ROUNDTRIP-SAME:    (tensor<4x3x5x1xi32>, tensor<4x3x5x1xi32>, tensor<4x3x5x2xi32>) -> tensor<4x3x5x4xi32>
    // CHECK-ROUNDTRIP:       %[[scatter:.*]] = "stablehlo.scatter"(%arg0, %2, %arg2) <{
    // CHECK-ROUNDTRIP-SAME:    scatter_dimension_numbers = #stablehlo.scatter
    // CHECK-ROUNDTRIP-SAME:      update_window_dims = [3], inserted_window_dims = [0, 1, 2, 3],
    // CHECK-ROUNDTRIP-SAME:      scatter_dims_to_operand_dims = [0, 2, 1, 3], index_vector_dim = 3>}>
    // CHECK-ROUNDTRIP:         (tensor<3x2x4x7x9xi32>, tensor<4x3x5x4xi32>, tensor<4x3x5x8xi32>) -> tensor<3x2x4x7x9xi32>
    // CHECK-ROUNDTRIP:       return %[[scatter]]
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [3],
        inserted_window_dims = [1, 3],
        input_batching_dims = [0, 2],
        scatter_indices_batching_dims = [1, 0],
        scatter_dims_to_operand_dims = [1, 3],
        index_vector_dim = 3
      >,
      unique_indices = false
    }> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<3x2x4x7x9xi32>, tensor<4x3x5x2xi32>, tensor<4x3x5x8xi32>) -> tensor<3x2x4x7x9xi32>
    return %0 : tensor<3x2x4x7x9xi32>
  }
}