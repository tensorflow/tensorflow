// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK: func @scatter
func.func @scatter(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

// CHECK: func @scatter_with_batching_dims
func.func @scatter_with_batching_dims(%input_tensor: tensor<5x200x100x300xf32>,
    %scatter_indices: tensor<5x10x2xi32>, %updates: tensor<5x10x300xf32>) ->
      tensor<5x200x100x300xf32> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [1, 2],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1, 2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<5x200x100x300xf32>, tensor<5x10x2xi32>, tensor<5x10x300xf32>) ->
      tensor<5x200x100x300xf32>
  func.return %0 : tensor<5x200x100x300xf32>
}

// -----

// CHECK: func @valid_scatter_dimensions_with_dynamic_index_vector_dim
func.func @valid_scatter_dimensions_with_dynamic_index_vector_dim(
    %input_tensor: tensor<?x?x?xf32>, %scatter_indices: tensor<10x?xi32>,
    %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1, 2],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<10x?xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK: func @scatter_with_promotable_types
func.func @scatter_with_promotable_types(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf64> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f64>, %rhs: tensor<f64>):
    %add = mhlo.add %lhs, %rhs : tensor<f64>
    "mhlo.return"(%add) : (tensor<f64>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf64>
  func.return %0 : tensor<200x100x300xf64>
}

// -----

// CHECK: func @scatter_with_promotable_quantized_types
func.func @scatter_with_promotable_quantized_types(%input_tensor: tensor<200x100x300x!quant.uniform<i8:f32, 2.000000e+00:15>>,
    %scatter_indices: tensor<10x2xi16>, %updates: tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>, %rhs: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>):
    %add = mhlo.add %lhs, %rhs : tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>
    "mhlo.return"(%add) : (tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300x!quant.uniform<i8:f32, 2.000000e+00:15>>, tensor<10x2xi16>,
      tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>>
}

// -----

func.func @scatter_c1(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  // expected-error @+1 {{Not all inputs have compatible shapes.}}
  %0, %1 = "mhlo.scatter"(%arg0, %arg2, %arg1, %arg2, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
    "mhlo.return"(%arg3, %arg5) : (tensor<i32>, tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<1xi32>, tensor<1x1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return %0 : tensor<3xi32>
}

// -----

func.func @scatter_c2(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects rank-of operand to match size-of('update_window_dims') + size-of('inserted_window_dims') + size-of('input_batching_dims') i.e. 4 but got 3.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1, 2],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<?x?xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c2(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects rank-of operand to match size-of('update_window_dims') + size-of('inserted_window_dims') + size-of('input_batching_dims') i.e. 4 but got 3.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [1, 2],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<?x?x?xi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c3(%input_tensor: tensor<3xi32>, %scatter_indices: tensor<1x1xi32>,
                            %updates: tensor<1xi32>) -> tensor<3xi32> {
  // expected-error @+1 {{Not all updates have compatible shapes.}}
  %0, %1 = "mhlo.scatter"(%input_tensor, %input_tensor, %scatter_indices, %input_tensor, %updates) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
    "mhlo.return"(%arg3, %arg5) : (tensor<i32>, tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<3xi32>, tensor<1x1xi32>, tensor<3xi32>, tensor<1xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return %0 : tensor<3xi32>
}

// -----

func.func @scatter_c4(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{expects updates tensor must be of rank 3 ( == rank-of('scatter_indices') - 1 + size-of('update_window_dims'), where 'scatter_indices' is expanded by a trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')), but got 2.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1, 0],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c4(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{expects updates tensor must be of rank 3 ( == rank-of('scatter_indices') - 1 + size-of('update_window_dims'), where 'scatter_indices' is expanded by a trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')), but got 2.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c4(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x301xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{expects bounds of the window dimensions of updates to not exceed the bounds of the corresponding dimensions of operand. For dimension 1, updates bound is 301, operand bound is 300.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x301xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c4(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<11x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{expects bounds of the scatter dimensions of updates to be same as the bounds of the corresponding dimensions of scatter indices. For scatter dimension 0, updates bound is 10 , scatter_indices bound is 11.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<11x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c4(%input_tensor: tensor<5x200x100x300xf32>,
    %scatter_indices: tensor<5x10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<5x200x100x300xf32> {
  // expected-error @+1 {{expects updates tensor must be of rank 3 ( == rank-of('scatter_indices') - 1 + size-of('update_window_dims'), where 'scatter_indices' is expanded by a trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')), but got 2.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1, 2],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1, 2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<5x200x100x300xf32>, tensor<5x10x2xi32>, tensor<10x300xf32>) ->
      tensor<5x200x100x300xf32>
  func.return %0 : tensor<5x200x100x300xf32>
}

// -----

func.func @scatter_c6_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xi32>) ->
      tensor<200x100x300xf32> {
  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs :  tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xi32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c6_c23_c24(%input_tensor: tensor<200x100x300xi32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error@+1 {{The element-type of reduction-region's argument at index 1 is expected to be promotable from 'i32', but got 'f32'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs :  tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xi32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c7() ->  tensor<512x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = mhlo.constant dense<0> : tensor<1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects update_window_dims to be sorted; got: [0, 1, 3, 2].}}
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0, 1, 3, 2],
      scatter_dims_to_operand_dims = [3]>,
      index_vector_dim = 0,
      unique_indices = true} :
    (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<512x1x6400x6400xf32>) ->
      tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

func.func @scatter_c7() ->  tensor<512x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = mhlo.constant dense<0> : tensor<1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects update_window_dims to not repeat; got: [0, 1, 2, 2].}}
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0, 1, 2, 2],
      scatter_dims_to_operand_dims = [3]>,
      index_vector_dim = 0,
      unique_indices = true} :
    (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<512x1x6400x6400xf32>) ->
      tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

func.func @scatter_c8() ->  tensor<512x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = mhlo.constant dense<0> : tensor<1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects each element of update_window_dims to be in range [0, rank-of('updates')) i.e. [0, 4). got: -1.}}
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [-1, 0, 1, 2],
      scatter_dims_to_operand_dims = [3]>,
      index_vector_dim = 0,
      unique_indices = true} :
    (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<512x1x6400x6400xf32>) ->
      tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

func.func @scatter_c8() ->  tensor<512x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = mhlo.constant dense<0> : tensor<1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects each element of update_window_dims to be in range [0, rank-of('updates')) i.e. [0, 4). got: 4.}}
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {
    indices_are_sorted = true,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0, 1, 2, 4],
      scatter_dims_to_operand_dims = [3]>,
      index_vector_dim = 0,
      unique_indices = true} :
    (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<512x1x6400x6400xf32>) ->
      tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

func.func @scatter_c9(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
// expected-error @+1 {{has duplicated dimension from inserted_window_dims and input_batching_dims: 1}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c9(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
// expected-error @+1 {{has duplicated dimension from inserted_window_dims and input_batching_dims: 0}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c10(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects inserted_window_dims to be sorted; got: [1, 0].}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1, 0],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<?x?xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c11(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of inserted_window_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [-1, 3],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c11(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of inserted_window_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 3],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c12(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects input_batching_dims to be sorted; got: [1, 0].}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      input_batching_dims = [1, 0],
      scatter_indices_batching_dims = [0, 1],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<?x?xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c13(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of input_batching_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1],
      input_batching_dims = [-1],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c13(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of input_batching_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1],
      input_batching_dims = [3],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c14(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects scatter_indices_batching_dims to not repeat; got: [1, 0, 1].}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      input_batching_dims = [0, 1],
      scatter_indices_batching_dims = [1, 0, 1],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<?x?x?xi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c15(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_indices_batching_dims to be in range [0, rank-of('scatter_indices')) i.e. [0, 3). got: -1.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      input_batching_dims = [0, 1],
      scatter_indices_batching_dims = [1, -1],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<?x?x?xi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c15(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_indices_batching_dims to be in range [0, rank-of('scatter_indices')) i.e. [0, 3). got: 3.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      input_batching_dims = [0, 1],
      scatter_indices_batching_dims = [1, 3],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<?x?x?xi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c16(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{expects scatter_indices_batching_dims not to include index_vector_dim 2.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      input_batching_dims = [0, 1],
      scatter_indices_batching_dims = [1, 2],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<?x?x?xi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c17(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x5x2xi32>, %updates: tensor<10x5x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{input_batching_dims and scatter_indices_batching_dims should have the same size.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [1],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [0, 1],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x5x2xi32>, tensor<10x5x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c18(%input_tensor: tensor<5x100x300xf32>,
    %scatter_indices: tensor<10x5x2xi32>, %updates: tensor<10x5x300xf32>) ->
      tensor<5x100x300xf32> {
  // expected-error @+1 {{input_batching_dims[1] and scatter_indices_batching_dims[1] must have compatible sizes, but got 100 and 10.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      input_batching_dims = [0, 1],
      scatter_indices_batching_dims = [1, 0],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<5x100x300xf32>, tensor<10x5x2xi32>, tensor<10x5x300xf32>) ->
      tensor<5x100x300xf32>
  func.return %0 : tensor<5x100x300xf32>
}

// -----

func.func @scatter_c19(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Scatter op has 3 elements in scatter_dims_to_operand_dims and the bound of dimension index_vector_dim=1 of scatter_indices is 2. These two numbers must be equal.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1, 2],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<10x2xi32>, tensor<?x?xf32>) ->
      tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c19(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Scatter op has 3 elements in scatter_dims_to_operand_dims and the bound of dimension index_vector_dim=2 of scatter_indices is 1. These two numbers must be equal.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1, 2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<10x2xi32>, tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c20(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{has duplicated dimension from scatter_dims_to_operand_dims and input_batching_dims: 0}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 0],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c20(%input_tensor: tensor<200x10x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x10x300xf32> {
  // expected-error @+1 {{has duplicated dimension from scatter_dims_to_operand_dims and input_batching_dims: 1}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      input_batching_dims = [1],
      scatter_indices_batching_dims = [0],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x10x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x10x300xf32>
  func.return %0 : tensor<200x10x300xf32>
}

// -----

func.func @scatter_c21(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_dims_to_operand_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [-1, 0],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<?x?x?xi32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c21(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_dims_to_operand_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 3],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<?x?xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c22(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects index_vector_dim to be in range [0, rank-of('scatter_indices')] i.e. [0, 2]. got: 3.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 3
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<?x?x?xf32>, tensor<10x2xi32>, tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @scatter_c22(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects index_vector_dim to be in range [0, rank-of('scatter_indices')] i.e. [0, 2]. got: -1.}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = -1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Reduction-region must take 2 parameters, but takes 1 parameter(s)}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>):
    %add = mhlo.add %lhs, %lhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{The reduction-region expected to return some value(s)}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"() : () -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Reduction-region here must produce 1 tensors, but produces 2 instead}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add, %add) : (tensor<f32>, tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {

  // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'f32' instead}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: f32, %rhs: f32):
    %add = "llvm.add" (%lhs, %rhs) : (f32, f32) -> f32
    "mhlo.return"(%add) : (f32) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<f32>>' instead}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs :  tensor<f32>
    %tuple = "mhlo.tuple"(%add, %add) : (tensor<f32>, tensor<f32>) ->
                  tuple<tensor<f32>, tensor<f32>>
    "mhlo.return"(%tuple) : (tuple<tensor<f32>, tensor<f32>>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs :  tensor<f32>
    %cst = arith.constant dense<-1> : tensor<i32>
    "mhlo.return"(%cst) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<i32>):
    %add = mhlo.add %lhs, %lhs :  tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xi32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xi32>) ->
      tensor<200x100x300xi8> {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i8>' vs 'tensor<i32>'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<i8>, %rhs: tensor<i8>):
    %add = mhlo.add %lhs, %rhs : tensor<i8>
    "mhlo.return"(%add) : (tensor<i8>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xi32>, tensor<10x2xi32>, tensor<10x300xi32>) ->
      tensor<200x100x300xi8>
  func.return %0 : tensor<200x100x300xi8>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300xi32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xi8>) ->
      tensor<200x100x300xi8> {

  // expected-error@+1 {{The element-type of reduction-region's argument at index 1 is expected to be promotable from 'i32', but got 'i8'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<i8>, %rhs: tensor<i8>):
    %add = mhlo.add %lhs, %rhs : tensor<i8>
    "mhlo.return"(%add) : (tensor<i8>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xi32>, tensor<10x2xi32>, tensor<10x300xi8>) ->
      tensor<200x100x300xi8>
  func.return %0 : tensor<200x100x300xi8>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300x!quant.uniform<i8:f32, 2.000000e+00:15>>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i32:f64, 2.000000e+00:15>> {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>' vs 'tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>, %rhs: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>):
    %add = mhlo.add %lhs, %rhs : tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>
    "mhlo.return"(%add) : (tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300x!quant.uniform<i8:f32, 2.000000e+00:15>>, tensor<10x2xi32>, tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i32:f64, 2.000000e+00:15>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i32:f64, 2.000000e+00:15>>
}

// -----

func.func @scatter_c23(%input_tensor: tensor<200x100x300x!quant.uniform<i8:f64, 2.000000e+00:15>>,
    %scatter_indices: tensor<10x2xi16>, %updates: tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>> {

  // expected-error@+1 {{The element-type of reduction-region's argument at index 1 is expected to be promotable from '!quant.uniform<i8:f64, 2.000000e+00:15>', but got '!quant.uniform<i16:f32, 2.000000e+00:15>'}}
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>, %rhs: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>):
    %add = mhlo.add %lhs, %rhs : tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>
    "mhlo.return"(%add) : (tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300x!quant.uniform<i8:f64, 2.000000e+00:15>>, tensor<10x2xi16>, tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>>
}
