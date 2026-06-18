// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @bounds_check() -> tensor<4x3xi32> {
  %operand = mhlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  %indices = mhlo.constant dense<[[1], [8], [-3]]> : tensor<3x1xi32>
  %gather = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [0],
      start_index_map = [0]
    >,
    slice_sizes = dense<[4]> : tensor<1xi64>
  } : (tensor<10xi32>, tensor<3x1xi32>) -> tensor<4x3xi32>
  return %gather : tensor<4x3xi32>
}

// CHECK-LABEL: @bounds_check
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 6, 0], [2, 7, 1], [3, 8, 2], [4, 9, 3]]

func.func @gather_2d() -> tensor<4x2xi32> {
  // operand = np.arange(1, 16).reshape([5, 3, 1])
  // indices = np.array([[0, 0], [1, 0], [4, 3], [-1, -1]])
  // lax.gather(operand, indices, lax.GatherDimensionNumbers(offset_dims=(1,),
  //   collapsed_slice_dims=(1,2), start_index_map=(0,1,)), slice_sizes=[2,1,1]))
  %operand = arith.constant dense<[
    [[1], [2], [3]],
    [[4], [5], [6]],
    [[7], [8], [9]],
    [[10], [11], [12]],
    [[13], [14], [15]]
  ]> : tensor<5x3x1xi32>

  %indices = arith.constant dense<[
    [0, 0],
    [1, 0],
    [4, 3],
    [-1, -1]
  ]> : tensor<4x2xi64>

  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [1, 2],
      index_vector_dim = 1,
      offset_dims = [1],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[2, 1, 1]> : tensor<3xi64>
  } : (tensor<5x3x1xi32>, tensor<4x2xi64>) -> tensor<4x2xi32>

  func.return %0 : tensor<4x2xi32>
}

// CHECK-LABEL: @gather_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 4], [4, 7], [12, 15], [1, 4]]
