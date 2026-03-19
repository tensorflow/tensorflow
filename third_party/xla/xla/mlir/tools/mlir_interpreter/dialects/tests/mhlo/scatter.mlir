// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @bounds_check() -> tensor<10xi32> {
  // operand = np.zeros([8], dtype=np.int32)
  // indices = np.array([[1], [8], [-1]])
  // updates = np.array([[4, 5, 6], [6, 7, 8], [8, 9, 10]])
  // lax.scatter_add(operand, indices, updates,
  //   dimension_numbers=lax.ScatterDimensionNumbers(
  //      update_window_dims=(0,), inserted_window_dims=(),
  //      scatter_dims_to_operand_dims=(0,)))
  %operand = mhlo.constant dense<0> : tensor<10xi32>
  %indices = mhlo.constant dense<[[1], [8], [-1]]> : tensor<3x1xi32>
  %updates = mhlo.constant dense<[[4, 5, 6], [6, 7, 8], [8, 9, 10]]> : tensor<3x3xi32>
  %scatter = "mhlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %add = mhlo.add %lhs, %rhs : tensor<i32>
    "mhlo.return"(%add) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [],
      index_vector_dim = 1,
      scatter_dims_to_operand_dims = [0]
    >
  } : (tensor<10xi32>, tensor<3x1xi32>, tensor<3x3xi32>) -> tensor<10xi32>
  return %scatter : tensor<10xi32>
}

// CHECK-LABEL: @bounds_check
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 4, 6, 8, 0, 0, 0, 0, 0, 0]

func.func @update_last_element() -> tensor<2xi32> {
  %operand = mhlo.constant dense<[1, 1]> : tensor<2xi32>
  %indices = mhlo.constant dense<[[1]]> : tensor<1x1xi32>
  %updates = mhlo.constant dense<[[0]]> : tensor<1x1xi32>

  %scatter = "mhlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      "mhlo.return"(%arg4) : (tensor<i32>) -> ()
    }) {
      indices_are_sorted = false,
      scatter_dimension_numbers = #mhlo.scatter<
        update_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >,
      unique_indices = false
    } : (tensor<2xi32>, tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<2xi32>
  return %scatter : tensor<2xi32>
}

// CHECK-LABEL: @update_last_element
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1, 0]
