// RUN: dtensor-opt %s -split-input-file -dtensor-embedding -verify-diagnostics | FileCheck %s

// Check simple optimizer passes without error.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<784x10xf32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                      tf._mesh = "CPU|x=1,y=1|*CPU"},
           %arg1: tensor<100xi32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                   tf._mesh = "CPU|x=1,y=1|*CPU"}) -> (
      tensor<!tf_type.string> {tf._default_layout = "sharding_specs:x,unsharded, mesh:EPU|x=1,y=1|*EPU"}) {
  // CHECK: %[[CONST:.*]] = "tf.Const"()
  // CHECK: return %[[CONST]]
  %0 = "tf.GetEmbeddingConfiguration"(%arg0, %arg1) {num_scalars = 0 : i64, num_slots = 0 : i64, operand_segment_sizes = array<i32: 1, 0, 1, 0>, optimizer = @sgd_optimizer} : (tensor<784x10xf32>, tensor<100xi32>) -> tensor<!tf_type.string>
  func.return %0 : tensor<!tf_type.string>
}

// CHECK-NOT: sgd_optimizer
func.func @sgd_optimizer(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = "tf.Const"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Mul"(%arg0, %0) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %2 = "tf.Sub"(%arg1, %1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  %3 = "tf.Identity"(%2) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %3 : tensor<1x1xf32>
}

// -----

// Error on non-float input.
func.func @main(%arg0: tensor<784x10xf32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1|*CPU",
                                      tf._mesh = "CPU|x=1|*CPU"},
           %arg1: tensor<100xi32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1|*CPU",
                                   tf._mesh = "CPU|x=1|*CPU"}) -> (
      tensor<!tf_type.string> {tf._default_layout = "sharding_specs:x,unsharded, mesh:EPU|x=2|*EPU"}) {
  // expected-error @+1 {{optimizer has a non-float32 input with type S32 at input 1}}
  %0 = "tf.GetEmbeddingConfiguration"(%arg0, %arg1) {num_scalars = 0 : i64, num_slots = 0 : i64, operand_segment_sizes = array<i32: 1, 0, 1, 0>, optimizer = @optimizer_with_int} : (tensor<784x10xf32>, tensor<100xi32>) -> tensor<!tf_type.string>
  func.return %0 : tensor<!tf_type.string>
}

func.func @optimizer_with_int(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xi32>) -> tensor<1x1xf32> {
  %0 = "tf.Const"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Cast"(%arg1) : (tensor<1x1xi32>) -> tensor<1x1xf32>
  %2 = "tf.Mul"(%arg0, %0) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %3 = "tf.Sub"(%1, %2) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  %4 = "tf.Identity"(%3) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %4 : tensor<1x1xf32>
}

// -----

// Missing function.
func.func @main(%arg0: tensor<784x10xf32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                      tf._mesh = "CPU|x=1,y=1|*CPU"},
           %arg1: tensor<100xi32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                   tf._mesh = "CPU|x=1,y=1|*CPU"}) -> (
      tensor<!tf_type.string> {tf._default_layout = "sharding_specs:x,unsharded, mesh:EPU|x=2,y=1|*EPU"}) {
  // expected-error @+1 {{optimizer function optimizer_with_int not found}}
  %0 = "tf.GetEmbeddingConfiguration"(%arg0, %arg1) {num_scalars = 0 : i64, num_slots = 0 : i64, operand_segment_sizes = array<i32: 1, 0, 1, 0>, optimizer = @optimizer_with_int} : (tensor<784x10xf32>, tensor<100xi32>) -> tensor<!tf_type.string>
  func.return %0 : tensor<!tf_type.string>
}

// -----

// Missing optimizer attribute.
func.func @main(%arg0: tensor<784x10xf32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                      tf._mesh = "CPU|x=1,y=1|*CPU"},
           %arg1: tensor<100xi32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                   tf._mesh = "CPU|x=1,y=1|*CPU"}) -> (
      tensor<!tf_type.string> {tf._default_layout = "sharding_specs:x,unsharded, mesh:EPU|x=1,y=1|*EPU"}) {
  // expected-error @+1 {{op requires attribute 'optimizer'}}
  %0 = "tf.GetEmbeddingConfiguration"(%arg0, %arg1) {num_scalars = 0 : i64, num_slots = 0 : i64, operand_segment_sizes = array<i32: 1, 0, 1, 0>} : (tensor<784x10xf32>, tensor<100xi32>) -> tensor<!tf_type.string>
  func.return %0 : tensor<!tf_type.string>
}

// -----

// Too many GetEmbeddingConfiguration ops.
func.func @main(%arg0: tensor<784x10xf32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                      tf._mesh = "CPU|x=1,y=1|*CPU"},
           %arg1: tensor<100xi32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1,y=1|*CPU",
                                   tf._mesh = "CPU|x=1,y=1|*CPU"}) -> (
      tensor<!tf_type.string> {tf._default_layout = "sharding_specs:x,unsharded, mesh:EPU|x=2,y=1|*EPU"}) {
  %0 = "tf.GetEmbeddingConfiguration"(%arg0, %arg1) {num_scalars = 0 : i64, num_slots = 0 : i64, operand_segment_sizes = array<i32: 1, 0, 1, 0>, optimizer = @sgd_optimizer} : (tensor<784x10xf32>, tensor<100xi32>) -> tensor<!tf_type.string>
  // expected-error @+1 {{second GetEmbeddingConfiguration op found, only 1 supported}}
  %1 = "tf.GetEmbeddingConfiguration"(%arg0, %arg1) {num_scalars = 0 : i64, num_slots = 0 : i64, operand_segment_sizes = array<i32: 1, 0, 1, 0>, optimizer = @sgd_optimizer} : (tensor<784x10xf32>, tensor<100xi32>) -> tensor<!tf_type.string>
  func.return %0 : tensor<!tf_type.string>
}

func.func @sgd_optimizer(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = "tf.Const"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Mul"(%arg0, %0) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %2 = "tf.Sub"(%arg1, %1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  %3 = "tf.Identity"(%2) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %3 : tensor<1x1xf32>
}
