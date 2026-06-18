// RUN: litert-opt %s -tfl-pin-ops-with-side-effects | FileCheck %s

func.func @id(%arg0: tensor<1xf32>)->tensor<1xf32> {
  func.return %arg0 : tensor<1xf32>
}

func.func @noop()->() {
  func.return
}

// CHECK-LABEL: @tf_if_gets_control_node
func.func @tf_if_gets_control_node(%arg0: tensor<1xi1>)->() {
 "tf.If"(%arg0) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @noop, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @noop} : (tensor<1xi1>) -> ()
 func.return
}
// CHECK-NEXT: %[[CONTROL:.*]] = tfl.control_node controls "tf.If"
// CHECK-NEXT: return

// CHECK-LABEL: @tfl_if_gets_control_node
func.func @tfl_if_gets_control_node(%arg0: tensor<1xi1>)->() {
  "tfl.if"(%arg0) (
  {
    // then
    ^bb0:
     "tfl.yield"() : () -> ()
  },
  {
    // else
   ^bb0:
    "tfl.yield"() : () -> ()
  }) : (tensor<1xi1>) -> ()
  func.return
}
// CHECK-NEXT: %[[CONTROL:.*]] = tfl.control_node controls "tfl.if"
// CHECK-NEXT: "tfl.yield"() : () -> ()
// CHECK-NEXT: }, {
// CHECK-NEXT: "tfl.yield"() : () -> ()
// CHECK-NEXT: }) : (tensor<1xi1>) -> ()
// CHECK-NEXT: return

// CHECK-LABEL: @tfl_while_gets_control_node
func.func @tfl_while_gets_control_node()->() {
  "tfl.while"() (
  {
    // cond
    ^bb0:
     "tfl.yield"() : () -> ()
  },
  {
    //body
   ^bb0:
    "tfl.yield"() : () -> ()
  }) : () -> ()
  func.return
}
// CHECK-NEXT: %[[CONTROL:.*]] = tfl.control_node controls "tfl.while"
// CHECK-NEXT: "tfl.yield"() : () -> ()
// CHECK-NEXT: }, {
// CHECK-NEXT: "tfl.yield"() : () -> ()
// CHECK-NEXT: }) : () -> ()
// CHECK-NEXT: return

// CHECK-LABEL: @resource_input_gets_control_node
func.func @resource_input_gets_control_node(%arg: tensor<!tf_type.resource<tensor<1xf32>>>) -> tensor<1xf32> {
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %out = "tfl.add"(%const, %arg) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<!tf_type.resource<tensor<1xf32>>>) -> (tensor<1xf32>)
  func.return %out : tensor<1xf32>
}
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[OUT:.*]], %[[CONTROL:.*]] = tfl.control_node controls "tfl.add"(%[[CONST]]


// CHECK-LABEL: @resource_output_gets_control_node
func.func @resource_output_gets_control_node() -> tensor<!tf_type.resource<tensor<1xf32>>> {
  %var = "tfl.var_handle"() {container = "", shared_name = "states"} : () -> tensor<!tf_type.resource<tensor<1xf32>>>
  func.return %var : tensor<!tf_type.resource<tensor<1xf32>>>
}
// CHECK-NEXT: %[[VAR:.*]], %[[CONTROL:.*]] = tfl.control_node controls "tfl.var_handle"
// CHECK-NEXT: return %[[VAR]]


// CHECK-LABEL: @tfl_call_once_gets_control_node
func.func @tfl_call_once_gets_control_node() -> () {
  "tfl.call_once"() { session_init_function = "noop" } : () -> ()
  func.return
}
// CHECK-NEXT: control_node controls "tfl.call_once"
// CHECK-NEXT: return

// CHECK-LABEL: @sequence_of_side_effect_ops
func.func @sequence_of_side_effect_ops() -> tensor<1xf32> {
  %var = "tfl.var_handle"() {container = "", shared_name = "states"} : () -> tensor<!tf_type.resource<tensor<1xf32>>>
  %const = "tfl.pseudo_const"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %tmp1 = "tfl.add"(%const, %const) { fused_activation_function = "NONE" } : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
  %tmp2 = "tfl.add"(%var, %tmp1) { fused_activation_function = "NONE" } : (tensor<!tf_type.resource<tensor<1xf32>>>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp3 = "tfl.add"(%tmp2, %tmp1) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  %tmp4 = "tfl.while"(%tmp3) (
  {
    // cond
    ^bb0(%arg_cond: tensor<1xf32>):
    %result_cond = tfl.greater(%arg_cond, %const) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    "tfl.yield"(%result_cond) : (tensor<1xi1>) -> ()
  },
  {
    //body
    ^bb0(%arg_body: tensor<1xf32>):
    %result_body = "tfl.add"(%arg_body, %arg_body) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
    "tfl.yield"(%result_body) : (tensor<1xf32>) -> ()
  }) : (tensor<1xf32>) -> (tensor<1xf32>)
  %tmp5 = "tfl.add"(%tmp4, %tmp2) { fused_activation_function = "NONE" } : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %result = "tf.If"(%tmp5, %tmp5) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @id, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @id} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %result : tensor<1xf32>
}
// CHECK-NEXT: %[[VAR:.*]], %[[SEQ0:.*]] = tfl.control_node controls "tfl.var_handle"()
// CHECK-NEXT: %[[CONST:.*]] = "tfl.pseudo_const"()
// CHECK-NEXT: %[[TMP1:.*]] = tfl.add %[[CONST]], %[[CONST]]
// CHECK-NEXT: %[[TMP2:.*]], %[[SEQ1:.*]] = tfl.control_node(%[[SEQ0]]) controls "tfl.add"(%[[VAR]], %[[TMP1]])
// CHECK-NEXT: %[[TMP3:.*]] = tfl.add %[[TMP2]], %[[TMP1]]
// CHECK-NEXT: %[[TMP4:.*]], %[[SEQ2:.*]] = tfl.control_node(%[[SEQ1]]) controls "tfl.while"(%[[TMP3]]) ({
// CHECK-NEXT:   ^bb0(%[[ARG_COND:.*]]: tensor<1xf32>):
// CHECK-NEXT:   %[[RESULT_COND:.*]] = tfl.greater(%[[ARG_COND]], %[[CONST]])
// CHECK-NEXT:   "tfl.yield"(%[[RESULT_COND]])
// CHECK-NEXT: }, {
// CHECK-NEXT:   ^bb0(%[[ARG_BODY:.*]]: tensor<1xf32>):
// CHECK-NEXT:   %[[RESULT_BODY:.*]] = tfl.add %[[ARG_BODY]], %[[ARG_BODY]]
// CHECK-NEXT:   "tfl.yield"(%[[RESULT_BODY]])
// CHECK-NEXT: })
// CHECK-NEXT: %[[TMP5:.*]] = tfl.add %[[TMP4]], %[[TMP2]]
// CHECK-NEXT: %[[RESULT:.*]], %[[SEQ3:.*]] = tfl.control_node(%[[SEQ2]]) controls "tf.If"(%[[TMP5]], %[[TMP5]])
// CHECK-NEXT: return %[[RESULT]]
// CHECK-NEXT: }
// CHECK-NEXT: }
