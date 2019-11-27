// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%input_tensor: tensor<200x100x300xf32>, %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  %0 = "xla_hlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>): // no predecessors
    %add = xla_hlo.add %lhs, %rhs : tensor<f32>
    "xla_hlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = {
      update_window_dims = dense<[1]> : tensor<1xi64>,
      inserted_window_dims = dense<[0, 1]> : tensor<2xi64>,
      scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
      index_vector_dim = 1 : i64
    },
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  return %0 : tensor<200x100x300xf32>
}

// CHECK:  [[COMPUTATION:%.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK-LABEL:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[200,100,300] parameter(0)
// CHECK:  [[VAL_2:%.*]] = s32[10,2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[10,300] parameter(2)
// CHECK-LABEL:  ROOT
// CHECK-SAME:  f32[200,100,300] scatter(f32[200,100,300] [[VAL_1]], s32[10,2] [[VAL_2]], f32[10,300] [[VAL_3]]), update_window_dims={1}, inserted_window_dims={0,1}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, indices_are_sorted=true, unique_indices=true, to_apply=[[COMPUTATION]]
