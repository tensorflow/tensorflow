// RUN: tf-opt -xla-legalize-tf-control-flow %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: @conditional
func @conditional(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>)
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  // CHECK: [[VAL0:%.+]] = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>

  // CHECK: [[VAL1:%.+]] = "xla_hlo.tuple"(%arg0, %arg1)
  // CHECK: [[VAL2:%.+]] = "xla_hlo.conditional"([[VAL0]], %1, %1) ( {
  // CHECK: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
  // CHECK:   [[VAL4:%.+]] = "xla_hlo.log"(%arg2)
  // CHECK:   [[VAL5:%.+]] = "xla_hlo.tuple"([[VAL4]])
  // CHECK:   "xla_hlo.return"([[VAL5]])
  // CHECK: },  {
  // CHECK: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>)
  // CHECK:   [[VAL4:%.+]] = "xla_hlo.exp"(%arg3)
  // CHECK:   [[VAL5:%.+]] = "xla_hlo.tuple"([[VAL4]])
  // CHECK:   "xla_hlo.return"([[VAL5]])
  // CHECK: })
  %1 = "tf.If"(%0, %arg0, %arg1) {Tcond = "tfdtype$DT_BOOL", Tin = ["tfdtype$DT_FLOAT", "tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _lower_using_switch_merge = true, _output_shapes = ["tfshape$"], device = "", else_branch = @cond_false, is_stateless = true, name = "cond", output_shapes = ["tfshape$"], then_branch = @cond_true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: [[VAL3:%.+]] = "xla_hlo.get_tuple_element"([[VAL2]]) {index = 0 : i32}
  // CHECK: return [[VAL3]]
  return %1 : tensor<f32>
}

func @cond_false(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  %0 = "xla_hlo.exp"(%arg1) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @cond_true(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  %0 = "xla_hlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
