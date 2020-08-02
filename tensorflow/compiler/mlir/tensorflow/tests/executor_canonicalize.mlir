// RUN: tf-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s


// Test single graph with no outputs and one island is folded away.
// CHECK-LABEL: func @graph_with_no_outputs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @graph_with_no_outputs(%arg0 : tensor<i1>) {
  tf_executor.graph {
    %1:2 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"(%3) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    tf_executor.fetch
  }
  return
}

// CHECK-NEXT: %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT: "tf.opB"(%[[OP_A]])
// CHECK-NEXT: return


// Test single graph with some outputs and one island is folded away.
// CHECK-LABEL: func @graph_with_outputs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @graph_with_outputs(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:3 = tf_executor.graph {
    %1:4 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"(%3) : (tensor<i1>) -> tensor<i1>
      %5 = "tf.opC"(%4) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3, %5, %4 : tensor<i1>, tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %1#1, %1#0, %1#2 : tensor<i1>, tensor<i1>, tensor<i1>
  }
  return %0#2, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK-NEXT: %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT: %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT: "tf.opC"(%[[OP_B]])
// CHECK-NEXT: return %[[OP_B]], %[[OP_A]] : tensor<i1>, tensor<i1>


// Test nested graphs and islands.
// CHECK-LABEL: func @nested_graph
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @nested_graph(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:3 = tf_executor.graph {
    %1:4 = tf_executor.island {
      %2:3 = tf_executor.graph {
        %3:4 = tf_executor.island {
          %4 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
          %5 = "tf.opB"(%4) : (tensor<i1>) -> tensor<i1>
          %6 = "tf.opC"(%5) : (tensor<i1>) -> tensor<i1>
          tf_executor.yield %4, %6, %5 : tensor<i1>, tensor<i1>, tensor<i1>
        }
        tf_executor.fetch %3#2, %3#0, %3#1 : tensor<i1>, tensor<i1>, tensor<i1>
      }
      tf_executor.yield %2#1, %2#1, %2#0 : tensor<i1>, tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %1#1, %1#0, %1#2 : tensor<i1>, tensor<i1>, tensor<i1>
  }
  return %0#2, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK-NEXT: %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT: %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT: "tf.opC"(%[[OP_B]])
// CHECK-NEXT: return %[[OP_B]], %[[OP_A]] : tensor<i1>, tensor<i1>


// Test single graph with multiple islands is unmodified.
// CHECK-LABEL: func @graph_with_multiple_islands
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @graph_with_multiple_islands(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:3 = tf_executor.graph {
    %1:4 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"(%3) : (tensor<i1>) -> tensor<i1>
      %5 = "tf.opC"(%4) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3, %5, %4 : tensor<i1>, tensor<i1>, tensor<i1>
    }
    %6:3 = tf_executor.island {
      %7 = "tf.opD"(%arg0) : (tensor<i1>) -> tensor<i1>
      %8 = "tf.opE"(%7) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %8, %7 : tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %1#1, %1#0, %6#0 : tensor<i1>, tensor<i1>, tensor<i1>
  }
  return %0#2, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK-NEXT: %[[GRAPH:[0-9]*]]:3 = tf_executor.graph {
// CHECK-NEXT:   %[[ISLAND_0:.*]]:3, %{{.*}} = tf_executor.island {
// CHECK-NEXT:     %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT:     %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_B]])
// CHECK-NEXT:     tf_executor.yield %[[OP_A]], %[[OP_C]], %[[OP_B]] : tensor<i1>, tensor<i1>, tensor<i1>
// CHECK:        %[[ISLAND_1:.*]]:2, %{{.*}} = tf_executor.island {
// CHECK-NEXT:     %[[OP_D:[0-9]*]] = "tf.opD"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_D]])
// CHECK-NEXT:     tf_executor.yield %[[OP_E]], %[[OP_D]] : tensor<i1>, tensor<i1>
// CHECK:        tf_executor.fetch %[[ISLAND_0]]#1, %[[ISLAND_0]]#0, %[[ISLAND_1]]#0 : tensor<i1>, tensor<i1>, tensor<i1>
// CHECK:      return %[[GRAPH]]#2, %[[GRAPH]]#1 : tensor<i1>, tensor<i1>


// Test single graph with an island and executor ops is unmodified.
// CHECK-LABEL: func @graph_with_island_and_executor_op
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @graph_with_island_and_executor_op(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:3 = tf_executor.graph {
    %1:4 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"(%3) : (tensor<i1>) -> tensor<i1>
      %5 = "tf.opC"(%4) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3, %5, %4 : tensor<i1>, tensor<i1>, tensor<i1>
    }
    %6:2 = tf_executor.LoopCond %1#0 : tensor<i1>
    tf_executor.fetch %1#1, %1#0, %6#0 : tensor<i1>, tensor<i1>, tensor<i1>
  }
  return %0#2, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK-NEXT: %[[GRAPH:[0-9]*]]:3 = tf_executor.graph {
// CHECK-NEXT:   %[[ISLAND:.*]]:3, %{{.*}} = tf_executor.island {
// CHECK-NEXT:     %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT:     %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_B]])
// CHECK-NEXT:     tf_executor.yield %[[OP_A]], %[[OP_C]], %[[OP_B]] : tensor<i1>, tensor<i1>, tensor<i1>
// CHECK:        %[[LOOP_COND:.*]], %[[LOOP_COND_control:.*]] = tf_executor.LoopCond %[[ISLAND]]#0
// CHECK-NEXT:   tf_executor.fetch %[[ISLAND]]#1, %[[ISLAND]]#0, %[[LOOP_COND]] : tensor<i1>, tensor<i1>, tensor<i1>
// CHECK:      return %[[GRAPH]]#2, %[[GRAPH]]#1 : tensor<i1>, tensor<i1>


// Test multiple graphs collapsed.
// CHECK-LABEL: func @multiple_graphs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @multiple_graphs(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %0:4 = tf_executor.graph {
    %2:4 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"(%3) : (tensor<i1>) -> tensor<i1>
      %5 = "tf.opC"(%4) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3, %5, %4 : tensor<i1>, tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %arg0, %2#0, %2#1, %2#2 : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
  }
  %1:3 = tf_executor.graph {
    %6:3 = tf_executor.island {
      %7 = "tf.opD"(%arg0) : (tensor<i1>) -> tensor<i1>
      %8 = "tf.opE"(%7) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %8, %7 : tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %arg0, %6#0, %6#1 : tensor<i1>, tensor<i1>, tensor<i1>
  }
  return %1#1, %1#0, %1#2, %0#1, %0#0, %0#3 : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-NEXT: %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT: %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT: "tf.opC"(%[[OP_B]])
// CHECK-NEXT: %[[OP_D:[0-9]*]] = "tf.opD"(%[[ARG_0]])
// CHECK-NEXT: %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_D]])
// CHECK-NEXT: return %[[OP_E]], %[[ARG_0]], %[[OP_D]], %[[OP_A]], %[[ARG_0]], %[[OP_B]] : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>


// Test empty graph with no outputs.
// CHECK-LABEL: func @empty_graph_with_no_outputs
func @empty_graph_with_no_outputs() {
  tf_executor.graph {
    tf_executor.fetch
  }
  return
}

// CHECK-NEXT: return


// Test empty graph with some outputs.
// CHECK-LABEL: func @empty_graph_with_outputs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i1>)
func @empty_graph_with_outputs(%arg0 : tensor<i1>, %arg1 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:2 = tf_executor.graph {
    tf_executor.fetch %arg1, %arg0 : tensor<i1>, tensor<i1>
  }
  return %0#0, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK-NEXT: return %[[ARG_1]], %[[ARG_0]] : tensor<i1>, tensor<i1>


// Test multiple empty graphs.
// CHECK-LABEL: func @empty_graphs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i1>)
func @empty_graphs(%arg0 : tensor<i1>, %arg1 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0 = tf_executor.graph {
    tf_executor.fetch %arg1 : tensor<i1>
  }
  tf_executor.graph {
    tf_executor.fetch
  }
  %1 = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<i1>
  }
  return %0, %1 : tensor<i1>, tensor<i1>
}

// CHECK-NEXT: return %[[ARG_1]], %[[ARG_0]] : tensor<i1>, tensor<i1>


// Test empty graphs and graphs with a single island.
// CHECK-LABEL: func @empty_and_filled_graphs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @empty_and_filled_graphs(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %0:4 = tf_executor.graph {
    %2:4 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"(%3) : (tensor<i1>) -> tensor<i1>
      %5 = "tf.opC"(%4) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3, %5, %4 : tensor<i1>, tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %arg0, %2#0, %2#1, %2#2 : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
  }
  tf_executor.graph {
    tf_executor.fetch
  }
  %1:3 = tf_executor.graph {
    %6:3 = tf_executor.island {
      %7 = "tf.opD"(%arg0) : (tensor<i1>) -> tensor<i1>
      %8 = "tf.opE"(%7) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %8, %7 : tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %arg0, %6#0, %6#1 : tensor<i1>, tensor<i1>, tensor<i1>
  }
  %9 = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<i1>
  }
  return %1#1, %1#0, %9, %0#1, %0#0, %0#3 : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-NEXT: %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT: %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT: "tf.opC"(%[[OP_B]])
// CHECK-NEXT: %[[OP_D:[0-9]*]] = "tf.opD"(%[[ARG_0]])
// CHECK-NEXT: %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_D]])
// CHECK-NEXT: return %[[OP_E]], %[[ARG_0]], %[[ARG_0]], %[[OP_A]], %[[ARG_0]], %[[OP_B]] : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>


// Test single empty island in graph with control output in graph fetch results
// in graph being removed.
// CHECK-LABEL: func @single_empty_island_single_graph_control
func @single_empty_island_single_graph_control() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch %0 : !tf_executor.control
  }
  return
}

// CHECK-NEXT: return


// Test empty island with no operands and no data result user is removed.
// Control result users should also have their respective operands removed.
// CHECK-LABEL: func @empty_island_no_operand_no_data_result
func @empty_island_no_operand_no_data_result() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    %1 = tf_executor.island(%0) {
      %3 = "tf.opA"() : () -> tensor<i1>
      tf_executor.yield
    }
    %2 = tf_executor.island(%0, %1) {
      %4 = "tf.opB"() : () -> tensor<i1>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[ISLAND_0:.*]] = tf_executor.island
// CHECK-NEXT:     "tf.opA"
// CHECK:        tf_executor.island(%[[ISLAND_0]])
// CHECK-NEXT:     "tf.opB"
// CHECK-NOT:    tf_executor.island


// Test empty island with one operand and no data results is removed and the
// operand is forwarded to its control result users.
// CHECK-LABEL: func @empty_island_one_operand_no_data_result
func @empty_island_one_operand_no_data_result() {
  tf_executor.graph {
    %0 = tf_executor.island {
      %3 = "tf.opA"() : () -> tensor<i1>
      tf_executor.yield
    }
    %1 = tf_executor.island(%0) {
      tf_executor.yield
    }
    %2 = tf_executor.island(%1) {
      %4 = "tf.opB"() : () -> tensor<i1>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[ISLAND_1:.*]] = tf_executor.island
// CHECK-NEXT:     "tf.opA"
// CHECK:        tf_executor.island(%[[ISLAND_1]])
// CHECK-NEXT:     "tf.opB"
// CHECK-NOT:    tf_executor.island


// Test empty island with no operands, one data result and no control result
// users is removed and its data result forwarded to its users.
// CHECK-LABEL: func @empty_island_no_operand_one_data_no_control_result
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @empty_island_no_operand_one_data_no_control_result(%arg0 : tensor<i1>) {
  tf_executor.graph {
    %0:2 = tf_executor.island() {
      tf_executor.yield %arg0 : tensor<i1>
    }
    %1 = tf_executor.island {
      %3 = "tf.opA"(%0#0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield
    }
    %2 = tf_executor.island() {
      %4 = "tf.opB"(%0#0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        tf_executor.island
// CHECK-NEXT:     "tf.opA"(%[[ARG_0]])
// CHECK:        tf_executor.island {
// CHECK-NEXT:     "tf.opB"(%[[ARG_0]])
// CHECK-NOT:    tf_executor.island


// Test empty control trigger with no operands is removed.
// Control result users should also have their respective operands removed.
// CHECK-LABEL: func @empty_control_trigger
func @empty_control_trigger() {
  tf_executor.graph {
    %0 = tf_executor.ControlTrigger {}
    %1 = tf_executor.island(%0) {
      %3 = "tf.opA"() : () -> tensor<i1>
      tf_executor.yield
    }
    %2 = tf_executor.island(%0, %1) {
      %4 = "tf.opB"() : () -> tensor<i1>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[ISLAND_0:.*]] = tf_executor.island
// CHECK-NEXT:     "tf.opA"
// CHECK:        tf_executor.island(%[[ISLAND_0]])
// CHECK-NEXT:     "tf.opB"
// CHECK-NOT:    tf_executor.island
