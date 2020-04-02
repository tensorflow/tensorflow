// RUN: tf-opt %s -split-input-file -verify-diagnostics

func @invalid_type() -> !tf_executor.foobar
// expected-error@-1 {{unknown tf_executor type: foobar}}

// -----

// Check that tf_executor.graph does not accept any operand.
func @graph_with_invalid_op(%arg0: tensor<*xf32>) {
  "tf_executor.graph" (%arg0) : (tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.graph' op requires zero operands}}
  return
}

// -----

// Check that an empty graph is invalid (it needs a region).
func @empty_graph() {
 "tf_executor.graph" () ({
// expected-error@-1 {{'tf_executor.graph' op region #0 ('body') failed to verify constraint: region with 1 blocks}}
  }) : () -> ()
  return
}

// -----

// Check that an empty graph is invalid (it needs a region).
func @empty_graph() {
 "tf_executor.graph" () ({
// expected-error@-1 {{'tf_executor.graph' op expects a non-empty block}}
 ^entry:
  }) : () -> ()
  return
}

// -----

// Check that only tf_executor operations can be present in a tf_executor.graph.
func @graph_with_invalid_op(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = "tf_executor.graph" () ({
    %val = addf %arg0, %arg0 : tensor<*xf32>
// expected-error@-1 {{'std.addf' op unallowed inside a tf_executor.graph region}}
    tf_executor.fetch %val : tensor<*xf32>
  }) : () -> tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// Check that tf_executor.graph can't be nested directly in a tf_executor.graph.
func @nested_graph() {
  tf_executor.graph {
    tf_executor.graph {}
// expected-error@-1 {{'tf_executor.graph' op unallowed directly inside another tf_executor.graph}}
  }
  return
}

// -----

// Check that a tf_executor.fetch is terminating a tf_executor.graph (custom parser)
func @graph_with_invalid_terminator(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  tf_executor.graph {
// expected-error@-1 {{custom op 'tf_executor.graph' expects a tf_executor.fetch terminator}}
    return
  }
  return %arg0 : tensor<*xf32>
}

// -----

// Check that a tf_executor.fetch parent is a graph.
func @parent_is_graph() {
  "some.op"() ({
    tf_executor.fetch
// expected-error@-1 {{'tf_executor.fetch' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that a tf_executor.fetch is terminating a tf_executor.graph (verifier)
func @graph_with_invalid_terminator(%arg0: tensor<*xf32>) -> tensor<*xf32> {
// expected-error@+2 {{'tf_executor.graph' op expects regions to end with 'tf_executor.fetch', found 'tf_executor.yield'}}
// expected-note@+1 {{in custom textual format, the absence of terminator implies 'tf_executor.fetch'}}
  "tf_executor.graph" () ({
    tf_executor.yield
  }) : () -> ()
  return %arg0 : tensor<*xf32>
}

// -----

// Check that a tf_executor.fetch is terminating a tf_executor.graph.
func @graph_with_invalid_terminator(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = "tf_executor.graph" () ({
    "tf_executor.fetch"() : () -> ()
// expected-error@-1 {{'tf_executor.fetch' op does not have enough operands to cover the graph returned values}}
  }) : () -> tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// Check that a graph with multiple regions issues an error
func @graph_with_multiple_region(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// expected-error@-1 {{custom op 'tf_executor.graph' expects a single block region}}
    br ^bb
  ^bb:
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that a fetch with not enough operands triggers the verifier.
func @invalid_fetch(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = "tf_executor.graph"() ({
    "tf_executor.fetch"() : () -> ()
// expected-error@-1 {{'tf_executor.fetch' op does not have enough operands to cover the graph returned values}}
  }) : () -> tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// Check that a fetch with not enough data-operands but more control inputs triggers the verifier.
func @invalid_fetch(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) -> tensor<*xf32> {
  %result = "tf_executor.graph"() ({
    "tf_executor.fetch"(%ctl) : (!tf_executor.control) -> ()
// expected-error@-1 {{'tf_executor.fetch' op operand #0 is a control type, can't be bound to a graph result}}
  }) : () -> tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// Check that a fetch with not too many operands triggers the verifier.
func @invalid_fetch(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = "tf_executor.graph"() ({
    "tf_executor.fetch"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.fetch' op operand #1 does not have a graph results to bind}}
  }) : () -> tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// Check that a fetch with operands that mistmatch the graph result type triggers the verifier.
func @invalid_fetch(%arg0: tensor<*xf32>) -> i32 {
  %result = "tf_executor.graph"() ({
    "tf_executor.fetch"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.fetch' op operand #0 type mismatch graph results}}
  }) : () -> i32
  return %result : i32
}

// -----

// Check that a fetch with operands after a control input triggers the verifier.
func @invalid_fetch(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) -> tensor<*xf32> {
  %result = "tf_executor.graph"() ({
    "tf_executor.fetch"(%arg0, %ctl, %arg0) : (tensor<*xf32>, !tf_executor.control, tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.fetch' op found non-control operand #2 after control operand}}
// expected-error@-2 {{'tf_executor.fetch' op failed to verify that all control inputs must appear after any non-control input}}
  }) : () -> tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// Check that a tf_executor.island parent is a graph.
func @parent_is_graph() {
  "some.op"() ({
    %ctl = tf_executor.island {}
// expected-error@-1 {{'tf_executor.island' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that an island can't have other operands than controls.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"(%arg0) ({
// expected-error@-1 {{'tf_executor.island' op operand #0 must be control}}
    }) : (tensor<*xf32>) -> (!tf_executor.control)
  }
  return
}

// -----

// Check that an island must have at least a control result.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
// expected-error@-1 {{'tf_executor.island' op expected 1 or more results}}
    }) : () -> ()
  }
  return
}

// -----

// Check that an island region can't be empty.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
// expected-error@-1 {{'tf_executor.island' op region #0 ('body') failed to verify constraint: region with 1 blocks}}
    }) : () -> (!tf_executor.control)
  }
  return
}

// -----

// Check that an island body can't be empty.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
// expected-error@-1 {{'tf_executor.island' op expects a non-empty block}}
 ^entry:
    }) : () -> (!tf_executor.control)
  }
  return
}

// -----

// Check that an island body can't be empty.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
// expected-error@-1 {{'tf_executor.island' op expects regions to end with 'tf_executor.yield', found 'std.return'}}
// expected-note@-2 {{in custom textual format, the absence of terminator implies 'tf_executor.yield'}}
      return
    }) : () -> (!tf_executor.control)
  }
  return
}

// -----

// Check that a tf_executor.yield parent is a tf_executor.island.
func @parent_is_island() {
  "some.op"() ({
    tf_executor.yield
// expected-error@-1 {{'tf_executor.yield' op expects parent op 'tf_executor.island'}}
  }) : () -> ()
  return
}

// -----

// Check that an island yield matches the island results.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
      "tf_executor.yield"(%arg0) : (tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.yield' op has 1 operand, but island returns 0}}
    }) : () -> (!tf_executor.control)
  }
  return
}

// -----

// Check that an island yield matches the island results.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
      "tf_executor.yield"(%arg0) : (tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.yield' op operand #0 type mismatch island results}}
    }) : () -> (i32, !tf_executor.control)
  }
  return
}

// -----

// Check that an island yield matches the island results.
func @invalid_island(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
      "tf_executor.yield"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.yield' op operand #1 type mismatch island results}}
    }) : () -> (tensor<*xf32>, i32, !tf_executor.control)
  }
  return
}

// -----

// Check that an island yield controls are after all non-control inputs.
func @invalid_yield(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
      "tf_executor.yield"(%arg0, %ctl, %arg0) : (tensor<*xf32>, !tf_executor.control, tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.yield' op unexpected control type for operand #1}}
    }) : () -> (tensor<*xf32>, !tf_executor.control, tensor<*xf32>, !tf_executor.control)
  }
  return
}

// -----

// Check that an island yield controls are after all non-control inputs.
func @invalid_yield(%arg0: tensor<*xf32>, %ctl: !tf_executor.control) {
  tf_executor.graph {
    "tf_executor.island"() ({
      "tf_executor.yield"(%arg0, %ctl) : (tensor<*xf32>, !tf_executor.control) -> ()
// expected-error@-1 {{'tf_executor.yield' op unexpected control type for operand #1}}
    }) : () -> (tensor<*xf32>, !tf_executor.control, !tf_executor.control)
  }
  return
}

// -----

// Check that a tf_executor.Switch parent is a graph.
func @parent_is_graph(%arg0: tensor<*xf32>, %arg1: tensor<i1>) {
  "some.op"() ({
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
// expected-error@-1 {{'tf_executor.Switch' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that a switch always takes two arguments.
func @invalid_switch(%arg0: tensor<*xf32>) {
  tf_executor.graph {
    %true, %false, %ctlSwitch = "tf_executor.Switch"(%arg0) : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Switch' op expected 2 or more operands}}
  }
  return
}

// -----

// Check that a switch second argument must be a valid predicate (i1).
func @invalid_switch(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %true, %false, %ctlSwitch = "tf_executor.Switch"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Switch' op operand #1 must be tensor of 1-bit signless integer values}}
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that a switch result type matches the input type.
func @invalid_switch(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %true, %false, %ctlSwitch = "tf_executor.Switch"(%arg1, %arg1) : (tensor<i1>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Switch' op failed to verify that data operand must be broadcastable to true result}}
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that a switch result type matches the input type.
func @invalid_switch(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %true, %false, %ctlSwitch = "tf_executor.Switch"(%arg1, %arg1) : (tensor<i1>, tensor<i1>) -> (tensor<i1>, tensor<*xf32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Switch' op failed to verify that data operand must be broadcastable to false result}}
    tf_executor.fetch %false : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that a tf_executor.SwitchN parent is a graph.
func @parent_is_graph(%arg0: tensor<*xf32>, %arg1: tensor<i32>) {
  "some.op"() ({
     %1:6 = tf_executor.SwitchN %arg0, %arg1 of 5 : tensor<*xf32>
// expected-error@-1 {{'tf_executor.SwitchN' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that switchN result numbers matches the num_out attribute.
func @invalid_switchN(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %fetches = tf_executor.graph {

     %1:3 = "tf_executor.SwitchN"(%arg1, %arg0) {num_outs = 5} : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.SwitchN' op expect `num_outs` (5) results but got 2}}

     tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %fetches : tensor<*xf32>
}

// -----

// Check that switchN result type matches the input type.
func @invalid_switchN(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %fetches = tf_executor.graph {

     %1:3 = "tf_executor.SwitchN"(%arg1, %arg0) {num_outs = 2} : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, i32, !tf_executor.control)
// expected-error@-1 {{'tf_executor.SwitchN' op type mismatch between data operand and result: 'tensor<*xf32>' vs 'i32'}}

     tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %fetches : tensor<*xf32>
}

// -----

// Check that switchN custom type has a single entry.
func @invalid_switchN(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %fetches = tf_executor.graph {

     %1:3 = tf_executor.SwitchN %arg1, %arg0 of 2 : tensor<*xf32>, i32
// expected-error@-1 {{custom op 'tf_executor.SwitchN'  expects only a single data type}}

     tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %fetches : tensor<*xf32>
}

// -----

// Check that a tf_executor.Merge parent is a graph.
func @parent_is_graph(%arg0: tensor<*xf32>) {
  "some.op"() ({
    %value, %idx, %ctlMerge = tf_executor.Merge %arg0, %arg0 : tensor<*xf32>
// expected-error@-1 {{'tf_executor.Merge' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that merge has at least one operand.
func @invalid_merge(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

    %value, %idx, %ctlMerge = "tf_executor.Merge"() : () -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Merge' op expects at least one operand}}
    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that merge has at least one non-control operand.
func @invalid_merge(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

    %value, %idx, %ctlMerge = "tf_executor.Merge"(%ctlSwitch) : (!tf_executor.control) -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Merge' op expects a non-control input}}
    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that merge data inputs are all the same type
func @invalid_merge(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

    %value, %idx, %ctlMerge = "tf_executor.Merge"(%true, %false, %arg1) : (tensor<*xf32>, tensor<*xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Merge' op expects all operands to be broadcastable with output type but got 'tensor<i1>' vs 'tensor<*xf32>'}}
    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that merge data inputs are broadcastable to the output
func @invalid_merge(%arg0: tensor<*xf32>, %arg1: tensor<4xf32>) -> tensor<8xf32> {
  %result = tf_executor.graph {
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<*xf32>, tensor<4xf32>) -> (tensor<8xf32>, tensor<i32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Merge' op expects all operands to be broadcastable with output type but got 'tensor<4xf32>' vs 'tensor<8xf32>'}}
    tf_executor.fetch %value : tensor<8xf32>
  }
  return %result : tensor<8xf32>
}

// -----

// Check that merge data inputs of variant type are broadcastable to the output
func @invalid_merge(%arg0: tensor<*x!tf.variant>, %arg1: tensor<4x!tf.variant>) -> tensor<8x!tf.variant> {
  %result = tf_executor.graph {
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<*x!tf.variant>, tensor<4x!tf.variant>) -> (tensor<8x!tf.variant>, tensor<i32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Merge' op expects all operands to be broadcastable with output type but got 'tensor<4x!tf.variant>' vs 'tensor<8x!tf.variant>'}}
    tf_executor.fetch %value : tensor<8x!tf.variant>
  }
  return %result : tensor<8x!tf.variant>
}

// -----

// Check that merge data inputs of resource type are broadcastable to the output
func @invalid_merge(%arg0: tensor<*x!tf.resource>, %arg1: tensor<4x!tf.resource>) -> tensor<8x!tf.resource> {
  %result = tf_executor.graph {
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<*x!tf.resource>, tensor<4x!tf.resource>) -> (tensor<8x!tf.resource>, tensor<i32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Merge' op expects all operands to be broadcastable with output type but got 'tensor<4x!tf.resource>' vs 'tensor<8x!tf.resource>'}}
    tf_executor.fetch %value : tensor<8x!tf.resource>
  }
  return %result : tensor<8x!tf.resource>
}

// -----

// Check that if result is a ref type, all operands need to be ref too.
func @invalid_merge(%arg0: tensor<4x!tf.f32ref>, %arg1: tensor<4xf32>) -> tensor<4x!tf.f32ref> {
  %result = tf_executor.graph {
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<4x!tf.f32ref>, tensor<4xf32>) -> (tensor<4x!tf.f32ref>, tensor<i32>, !tf_executor.control)
    // expected-error@-1 {{'tf_executor.Merge' op expects same operand and output element type but got 'tensor<4xf32>' vs 'tensor<4x!tf.f32ref>'}}
    tf_executor.fetch %value : tensor<4x!tf.f32ref>
  }
  return %result : tensor<4x!tf.f32ref>
}

// -----

// Check that merge data inputs can't appear after control input.
func @invalid_merge(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

    %value, %idx, %ctlMerge = "tf_executor.Merge"(%true, %ctlSwitch, %false) : (tensor<*xf32>, !tf_executor.control, tensor<*xf32>) -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Merge' op failed to verify that all control inputs must appear after any non-control input}}
// expected-error@-2 {{'tf_executor.Merge' op found non-control operand #2 after control operand}}
    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that a tf_executor.Enter parent is a graph.
func @parent_is_graph(%arg0: tensor<*xf32>) {
  "some.op"() ({
    %res:2 = tf_executor.Enter %arg0 frame "some/fra\"me" : tensor<*xf32>
// expected-error@-1 {{'tf_executor.Enter' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that Enter return value is the same type as the input.
func @invalid_enter(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %res:2 = "tf_executor.Enter"(%arg1) { frame_name = "some/fra\"me"} : (i1) -> (tensor<*xf32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Enter' op failed to verify that data operand must be broadcastable to result}}
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// -----

// Check that a tf_executor.NextIteration.Sink parent is a graph.
func @parent_is_graph(%arg0: tensor<*xf32>, %arg1: !tf_executor.token) {
  "some.op"() ({
    tf_executor.NextIteration.Sink[%arg1] %arg0 : tensor<*xf32>
// expected-error@-1 {{'tf_executor.NextIteration.Sink' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that a tf_executor.NextIteration.Source parent is a graph.
func @parent_is_graph() {
  "some.op"() ({
    %1:3 = tf_executor.NextIteration.Source : tensor<*xf32>
// expected-error@-1 {{'tf_executor.NextIteration.Source' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

func @invalid_nextiteration(%arg0: tensor<*xf32>, %arg1: !tf_executor.token) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : tensor<*xf32>
// expected-error@-1 {{'tf_executor.NextIteration.Source' op expects a single user for produced token}}
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// -----

func @invalid_nextiteration(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : tensor<*xf32>
// expected-error@-1 {{'tf_executor.NextIteration.Source' op token should be consumed by a sink op}}
    tf_executor.island {
      "tf.consume_token"(%1#1) : (!tf_executor.token) -> ()
    }
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// -----

func @invalid_nextiteration(%arg0: tensor<*xf32>, %arg1: !tf_executor.token) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.produce_token"() : () -> (!tf_executor.token)
      tf_executor.yield %2 : !tf_executor.token
    }
    "tf_executor.NextIteration.Sink"(%1#0, %arg0) : (!tf_executor.token, tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.NextIteration.Sink' op expects a token produced by a tf_executor.NextIteration.Source op}}
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// -----

func @invalid_nextiteration(%arg0: tensor<*xf32>, %arg1: !tf_executor.token) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    "tf_executor.NextIteration.Sink"(%arg1, %arg0) : (!tf_executor.token, tensor<*xf32>) -> ()
// expected-error@-1 {{'tf_executor.NextIteration.Sink' op expects a token directly produced by a tf_executor.NextIteration.Source op}}
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// -----

func @invalid_nextiteration(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : tensor<*xf32>
    "tf_executor.NextIteration.Sink"(%1#1, %arg1) : (!tf_executor.token, i1) -> ()
// expected-error@-1 {{'tf_executor.NextIteration.Sink' op input type 'i1' mismatch the tf_executor.NextIteration.Source output type: 'tensor<*xf32>'}}
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// -----

// Check that a tf_executor.Exit parent is a graph.
func @parent_is_graph(%arg0: tensor<*xf32>) {
  "some.op"() ({
    %1:2 = tf_executor.Exit %arg0 : tensor<*xf32>
// expected-error@-1 {{'tf_executor.Exit' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

func @exit(%arg0: tensor<*xi32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:2 = "tf_executor.Exit"(%arg0) : (tensor<*xi32>) -> (tensor<*xf32>, !tf_executor.control)
// expected-error@-1 {{'tf_executor.Exit' op failed to verify that data operand must be broadcastable to result}}
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// -----

// Check that a tf_executor.ControlTrigger parent is a graph.
func @parent_is_graph(%arg0: !tf_executor.control, %arg1: !tf_executor.control) {
  "some.op"() ({
    %0 = tf_executor.ControlTrigger %arg0, %arg1
// expected-error@-1 {{'tf_executor.ControlTrigger' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}

// -----

// Check that a tf_executor.LoopCond parent is a graph.
func @parent_is_graph(%arg0: tensor<i1>, %arg1: !tf_executor.control) {
  "some.op"() ({
    %1:2 = tf_executor.LoopCond %arg0, %arg1 : tensor<i1>
// expected-error@-1 {{'tf_executor.LoopCond' op expects parent op 'tf_executor.graph'}}
  }) : () -> ()
  return
}
