// RUN: dtensor-opt %s -split-input-file -verify-diagnostics

// Check that a DTensorLayout op with mismatched rank between layout and input
// value is disallowed.
func.func @invalid_rank_disallowed(%arg0: tensor<i32>) {
  // expected-error@+1 {{requires matching rank for layout and input, but got 2}}
  %0 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=2,y=2|*CPU>} : (tensor<i32>) -> tensor<i32>
  func.return
}

// -----

// Check that a DTensorLayout op with sharding configuration that cannot evenly
// divide the dimension of the input value is disallowed.
func.func @invalid_sharding_dim_disallowed(%arg0: tensor<2x2xi32>) {
  // expected-error@+1 {{requires dimension 0 to be divisible by sharding specified in DTensorLayout, but got dimension size=2 is not divisible by number of shards in layout for this dimension=8.}}
  %0 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>} : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
  func.return
}

// -----

func.func @dtensor_layout_with_sharding(%arg0: tensor<16x2xi32>) {
  // CHECK:      "tf.DTensorLayout"
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,y mesh:CPU|x=8,y=2|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3,CPU:4,CPU:5,CPU:6,CPU:7,CPU:8,CPU:9,CPU:10,CPU:11,CPU:12,CPU:13,CPU:14,CPU:15>
  // CHECK-SAME: (tensor<16x2xi32>) -> (tensor<16x2xi32>)
  %0 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<16x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16x2xi32>)
  func.return
}

// -----

func.func @dtensor_all_gather_unequal_tensor_ranks(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{received input and output layouts of unequal ranks 2 and 1}}
  %0 = "tf.DTensorAllGather"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16xi32>)
  func.return
}

// -----

func.func @dtensor_all_gather_bad_layouts(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{dimension 1 of output layout has sharding spec y which is more sharded then the input layout spec unsharded}}
  %0 = "tf.DTensorAllGather"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16x2xi32>)
  func.return
}

// -----

func.func @dtensor_all_gather_bad_input_rank(%arg0: tensor<16xi32>) {
  // expected-error@+1 {{input layout rank 2 is not equal to input rank 1}}
  %0 = "tf.DTensorAllGather"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16xi32>) -> (tensor<16x2xi32>)
  func.return
}

// -----

func.func @dtensor_all_gather_bad_output_rank(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{output layout rank 2 is not equal to output rank 1}}
  %0 = "tf.DTensorAllGather"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16xi32>)
  func.return
}

// -----

func.func @dtensor_all_gather_bad_output_shape(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{computed output shape 4 at dimension 1 is not equal to actual output shape 2}}
  %0 = "tf.DTensorAllGather"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16x2xi32>)
  func.return
}

// -----

func.func @dtensor_all_scatter_unequal_tensor_ranks(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{received input and output layouts of unequal ranks 2 and 1}}
  %0 = "tf.DTensorAllScatter"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16xi32>)
  func.return
}

// -----

func.func @dtensor_all_scatter_bad_layouts(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{dimension 1 of input layout has sharding spec y which is more sharded then the output layout spec unsharded}}
  %0 = "tf.DTensorAllScatter"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16x2xi32>)
  func.return
}

// -----

func.func @dtensor_all_scatter_bad_input_rank(%arg0: tensor<16xi32>) {
  // expected-error@+1 {{input layout rank 2 is not equal to input rank 1}}
  %0 = "tf.DTensorAllScatter"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16xi32>) -> (tensor<16x2xi32>)
  func.return
}

// -----

func.func @dtensor_all_scatter_bad_output_rank(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{output layout rank 2 is not equal to output rank 1}}
  %0 = "tf.DTensorAllScatter"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16xi32>)
  func.return
}

// -----

func.func @dtensor_all_scatter_bad_output_shape(%arg0: tensor<16x2xi32>) {
  // expected-error@+1 {{computed output shape 1 at dimension 1 is not equal to actual output shape 2}}
  %0 = "tf.DTensorAllScatter"(%arg0) {input_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=8,y=2|*CPU>, output_layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=8,y=2|*CPU>} : (tensor<16x2xi32>) -> (tensor<16x2xi32>)
  func.return
}
