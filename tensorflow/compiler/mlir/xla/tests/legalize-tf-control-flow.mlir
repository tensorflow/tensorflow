// RUN: tf-opt -xla-legalize-tf-control-flow %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: @if
func @if(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>)
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  // CHECK: [[VAL0:%.+]] = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: [[VAL1:%.+]] = "xla_hlo.tuple"(%arg0, %arg1)
  // CHECK: [[VAL2:%.+]] = "xla_hlo.if"([[VAL0]], [[VAL1]], [[VAL1]]) ( {
  // CHECK: ^bb0(%arg2: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:   [[VAL4:%.+]] = "xla_hlo.get_tuple_element"(%arg2) {index = 0 : i32}
  // CHECK:   [[VAL5:%.+]] = "xla_hlo.get_tuple_element"(%arg2) {index = 1 : i32}
  // CHECK:   [[VAL6:%.+]] = call @cond_true([[VAL4]], [[VAL5]])
  // CHECK:   [[VAL7:%.+]] = "xla_hlo.tuple"([[VAL6]])
  // CHECK:   "xla_hlo.return"([[VAL7]]) : (tuple<tensor<f32>>) -> ()
  // CHECK: },  {
  // CHECK: ^bb0(%arg2: tuple<tensor<f32>, tensor<f32>>)
  // CHECK:   [[VAL4:%.+]] = "xla_hlo.get_tuple_element"(%arg2) {index = 0 : i32}
  // CHECK:   [[VAL5:%.+]] = "xla_hlo.get_tuple_element"(%arg2) {index = 1 : i32}
  // CHECK:   [[VAL6:%.+]] = call @cond_false([[VAL4]], [[VAL5]])
  // CHECK:   [[VAL7:%.+]] = "xla_hlo.tuple"([[VAL6]])
  // CHECK: "xla_hlo.return"([[VAL7]]) : (tuple<tensor<f32>>) -> ()
  // CHECK: })
  %1 = "tf.If"(%0, %arg0, %arg1) {Tcond = "tfdtype$DT_BOOL", Tin = ["tfdtype$DT_FLOAT", "tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _lower_using_switch_merge = true, _output_shapes = ["tfshape$"], device = "", else_branch = @cond_false, is_stateless = true, name = "cond", output_shapes = [#tf.shape<>], then_branch = @cond_true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: [[VAL3:%.+]] = "xla_hlo.get_tuple_element"([[VAL2]]) {index = 0 : i32}
  // CHECK: return [[VAL3]]
  return %1 : tensor<f32>
}

func @cond_false(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  %0 = "xla_hlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @cond_true(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
attributes  {tf._input_shapes = ["tfshape$", "tfshape$"]} {
  %0 = "xla_hlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// CHECK-LABEL: func @case
// CHECK-SAME:  %[[BRANCH_INDEX:.*]]: tensor<i32>, %[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>) -> (tensor<f32>, tensor<f32>)
func @case(%index: tensor<i32>, %arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "tf.Case"(%index, %arg0, %arg1) {branches = [@exponential, @log, @floor]} : (tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK: %[[TUPLE_INPUT:.*]] = "xla_hlo.tuple"(%[[ARG0]], %[[ARG1]]) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
  // CHECK: %[[CASE:.*]]:2 = "xla_hlo.case"(%[[BRANCH_INDEX]], %[[TUPLE_INPUT]], %[[TUPLE_INPUT]], %[[TUPLE_INPUT]]) ( {
  // CHECK:   ^bb0(%[[TUPLE_ARG:.*]]: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:     %[[TUPLE_ELEMENT_0:.*]] = "xla_hlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[TUPLE_ELEMENT_1:.*]] = "xla_hlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[CALL_EXP:.*]]:2 = call @exponential(%[[TUPLE_ELEMENT_0]], %[[TUPLE_ELEMENT_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK:     "xla_hlo.return"(%[[CALL_EXP]]#0, %[[CALL_EXP]]#1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK:   },  {
  // CHECK:   ^bb0(%[[TUPLE_ARG:.*]]: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:     %[[TUPLE_ELEMENT_0:.*]] = "xla_hlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[TUPLE_ELEMENT_1:.*]] = "xla_hlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[CALL_LOG:.*]]:2 = call @log(%[[TUPLE_ELEMENT_0]], %[[TUPLE_ELEMENT_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK:     "xla_hlo.return"(%[[CALL_LOG]]#0, %[[CALL_LOG]]#1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK:   },  {
  // CHECK:   ^bb0(%[[TUPLE_ARG:.*]]: tuple<tensor<f32>, tensor<f32>>):
  // CHECK:     %[[TUPLE_ELEMENT_0:.*]] = "xla_hlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[TUPLE_ELEMENT_1:.*]] = "xla_hlo.get_tuple_element"(%[[TUPLE_ARG]]) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:     %[[CALL_FLOOR:.*]]:2 = call @floor(%[[TUPLE_ELEMENT_0]], %[[TUPLE_ELEMENT_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  // CHECK:     "xla_hlo.return"(%[[CALL_FLOOR]]#0, %[[CALL_FLOOR]]#1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK:   }) : (tensor<i32>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>) -> (tensor<f32>, tensor<f32>)
  return %0#0, %0#1 : tensor<f32>, tensor<f32>
// CHECK:   return %[[CASE]]#0, %[[CASE]]#1 : tensor<f32>, tensor<f32>
}

func @exponential(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "xla_hlo.exponential"(%arg1) : (tensor<f32>) -> tensor<f32>
  return %0, %arg1 : tensor<f32>, tensor<f32>
}

func @log(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "xla_hlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0, %arg1 : tensor<f32>, tensor<f32>
}

func @floor(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "xla_hlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0, %arg1 : tensor<f32>, tensor<f32>
}


// CHECK-LABEL: func @while
func @while(%arg0: tensor<f32> {tf_saved_model.index_path = [0]}) -> (tensor<i32> {tf_saved_model.index_path = []})
attributes  {tf._input_shapes = ["tfshape$"]} {
  // CHECK: [[VAL0:%.+]] = xla_hlo.constant dense<0>
  // CHECK: [[VAL1:%.+]] = xla_hlo.constant dense<-1>
  %0 = xla_hlo.constant dense<0> : tensor<i32>
  %1 = xla_hlo.constant dense<-1> : tensor<i32>
  // CHECK: [[VAL2:%.+]] = "xla_hlo.tuple"([[VAL0]], [[VAL1]], [[VAL0]])
  // CHECK: [[VAL3:%.+]] = "xla_hlo.while"([[VAL2]]) ( {
  // CHECK:   ^bb0(%arg1: tuple<tensor<i32>, tensor<i32>, tensor<i32>>):
  // CHECK:   [[VAL7:%.+]] = "xla_hlo.get_tuple_element"(%arg1) {index = 0 : i32}
  // CHECK:   [[VAL8:%.+]] = "xla_hlo.get_tuple_element"(%arg1) {index = 1 : i32}
  // CHECK:   [[VAL9:%.+]] = "xla_hlo.get_tuple_element"(%arg1) {index = 2 : i32}
  // CHECK:   [[VAL10:%.+]] = call @while_cond([[VAL7]], [[VAL8]], [[VAL9]])
  // CHECK:   "xla_hlo.return"([[VAL10]])
  // CHECK: },  {
  // CHECK: ^bb0(%arg1: tuple<tensor<i32>, tensor<i32>, tensor<i32>>):
  // CHECK:   [[VAL7:%.+]] = "xla_hlo.get_tuple_element"(%arg1) {index = 0 : i32}
  // CHECK:   [[VAL8:%.+]] = "xla_hlo.get_tuple_element"(%arg1) {index = 1 : i32}
  // CHECK:   [[VAL9:%.+]] = "xla_hlo.get_tuple_element"(%arg1) {index = 2 : i32}
  // CHECK:   [[VAL10:%.+]]:3 = call @while_body([[VAL7]], [[VAL8]], [[VAL9]])
  // CHECK:   [[VAL11:%.+]] = "xla_hlo.tuple"([[VAL10]]#0, [[VAL10]]#1, [[VAL10]]#2)
  // CHECK:   "xla_hlo.return"([[VAL11]])
  // CHECK: }) : (tuple<tensor<i32>, tensor<i32>, tensor<i32>>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>>
  // CHECK: [[VAL4:%.+]] = "xla_hlo.get_tuple_element"([[VAL3]]) {index = 0 : i32}
  // CHECK: [[VAL5:%.+]] = "xla_hlo.get_tuple_element"([[VAL3]]) {index = 1 : i32}
  // CHECK: [[VAL6:%.+]] = "xla_hlo.get_tuple_element"([[VAL3]]) {index = 2 : i32}
  // CHECK: return [[VAL6]]
  %2:3 = "tf.While"(%0, %1, %0) {T = ["tfdtype$DT_INT32", "tfdtype$DT_INT32", "tfdtype$DT_INT32"], _lower_using_switch_merge = true, _num_original_outputs = 3 : i64, _output_shapes = ["tfshape$", "tfshape$", "tfshape$"], body = @while_body, cond = @while_cond, device = "", is_stateless = true, name = "while", output_shapes = [#tf.shape<>, #tf.shape<>, #tf.shape<>], parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  return %2#2 : tensor<i32>
}
func @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i1>
attributes  {tf._input_shapes = ["tfshape$", "tfshape$", "tfshape$"]} {
  %0 = xla_hlo.constant dense<10> : tensor<i32>
  %1 = "xla_hlo.compare"(%arg2, %0) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}
func @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
attributes  {tf._input_shapes = ["tfshape$", "tfshape$", "tfshape$"]} {
  %0 = xla_hlo.constant dense<1> : tensor<i32>
  %1 = xla_hlo.add %arg2, %0 : tensor<i32>
  %2 = xla_hlo.add %arg0, %0 : tensor<i32>
  return %2, %arg1, %1 : tensor<i32>, tensor<i32>, tensor<i32>
}
