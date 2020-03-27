// RUN: tf-opt %s -tf-executor-island-coarsening | FileCheck %s --dump-input=fail


// Test that islands linked by a control dependency are merged.
// CHECK-LABEL: func @control_input
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @control_input(%arg0 : tensor<i1>) -> tensor<f32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island(%1#1) {
      %4 = "tf.opB"() : () -> tensor<f32>
      tf_executor.yield %4 : tensor<f32>
    }
    tf_executor.fetch %2#0 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// CHECK:        %[[ISLAND:.*]], %[[ISLAND_control:.*]] = tf_executor.island {
// CHECK-NEXT:     "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"
// CHECK-NEXT:     tf_executor.yield %[[OP_B]] : tensor<f32>
// CHECK:        tf_executor.fetch %[[ISLAND]] : tensor<f32>


// Test that islands linked by a data dependency are merged.
// CHECK-LABEL: func @data_input
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @data_input(%arg0 : tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island {
      %4 = "tf.opB"(%1#0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %4 : tensor<i1>
    }
    tf_executor.fetch %2#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK:        %[[ISLAND:.*]], %[[ISLAND_control:.*]] = tf_executor.island {
// CHECK-NEXT:     %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT:     tf_executor.yield %[[OP_B]] : tensor<i1>
// CHECK:        tf_executor.fetch %[[ISLAND]] : tensor<i1>


// Test empty/trivial islands are merged.
// CHECK-LABEL: func @empty_islands
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i1>)
func @empty_islands(%arg0 : tensor<i1>, %arg1 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:2 = tf_executor.graph {
    %1:2 = tf_executor.island {
      tf_executor.yield %arg1 : tensor<i1>
    }
    %2:2 = tf_executor.island {
      tf_executor.yield %arg0 : tensor<i1>
    }
    %3:2 = tf_executor.island {
      tf_executor.yield %1#0 : tensor<i1>
    }
    %4:2 = tf_executor.island {
      tf_executor.yield %2#0 : tensor<i1>
    }
    %5:3 = tf_executor.island wraps "tf.opA"(%3#0, %4#0) : (tensor<i1>, tensor<i1>) -> (tensor<i1>, tensor<i1>)
    %6:2 = tf_executor.island {
      tf_executor.yield %5#0 : tensor<i1>
    }
    %7:2 = tf_executor.island {
      tf_executor.yield %5#1 : tensor<i1>
    }
    %8:3 = tf_executor.island {
      tf_executor.yield %6#0, %7#0 : tensor<i1>, tensor<i1>
    }
    %9 = tf_executor.island(%8#2) {
      tf_executor.yield
    }
    tf_executor.fetch %8#0, %8#1 : tensor<i1>, tensor<i1>
  }
  return %0#0, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK:        %[[ISLAND:.*]]:2, %{{.*}} = tf_executor.island
// CHECK-NEXT:     "tf.opA"(%[[ARG_1]], %[[ARG_0]])
// CHECK:        tf_executor.fetch %[[ISLAND]]#0, %[[ISLAND]]#1 : tensor<i1>, tensor<i1>


// Test merging islands handle merging results.
// CHECK-LABEL: func @multiple_outputs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i1>)
func @multiple_outputs(%arg0 : tensor<i1>, %arg1 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:2 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
    %2:2 = tf_executor.island(%1#1) wraps "tf.opB"(%arg1) : (tensor<i1>) -> tensor<i1>
    tf_executor.fetch %1#0, %2#0 : tensor<i1>, tensor<i1>
  }
  return %0#0, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK:        %[[ISLAND:.*]]:2, %{{.*}} = tf_executor.island {
// CHECK-NEXT:     %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"(%[[ARG_1]])
// CHECK-NEXT:     tf_executor.yield %[[OP_A]], %[[OP_B]] : tensor<i1>, tensor<i1>
// CHECK:        tf_executor.fetch %[[ISLAND]]#0, %[[ISLAND]]#1 : tensor<i1>, tensor<i1>


// Test merging islands with multiple inner ops.
// CHECK-LABEL: func @multi_op_regions
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i32>, %[[ARG_1:[a-z0-9]*]]: tensor<i32>)
func @multi_op_regions(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.opA"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %3 = "tf.opB"(%2, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %3 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %5 = "tf.opC"(%1#0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %6 = "tf.opD"(%5, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %6 : tensor<i32>
    }
    tf_executor.fetch %4#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK:        %[[ISLAND:.*]], %[[ISLAND_control:.*]] = tf_executor.island {
// CHECK-NEXT:     %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]], %[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]], %[[ARG_0]])
// CHECK-NEXT:     %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_B]], %[[ARG_1]])
// CHECK-NEXT:     %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_C]], %[[ARG_0]])
// CHECK-NEXT:     tf_executor.yield %[[OP_D]] : tensor<i32>
// CHECK:        tf_executor.fetch %[[ISLAND]] : tensor<i32>


// Test merging multiple islands with multiple inner ops preserves order.
// CHECK-LABEL: func @transitive_preserve_order
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i32>, %[[ARG_1:[a-z0-9]*]]: tensor<i32>)
func @transitive_preserve_order(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.opA"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %3 = "tf.opB"(%2, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %3 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %5 = "tf.opC"(%1#0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %6 = "tf.opD"(%5, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %6 : tensor<i32>
    }
    %7:2 = tf_executor.island {
      %8 = "tf.opE"(%4#0, %1#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %9 = "tf.opF"(%8, %8) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %9 : tensor<i32>
    }
    tf_executor.fetch %7#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK:        %[[ISLAND:.*]], %[[ISLAND_control:.*]] = tf_executor.island {
// CHECK-NEXT:     %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]], %[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]], %[[ARG_0]])
// CHECK-NEXT:     %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_B]], %[[ARG_1]])
// CHECK-NEXT:     %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_C]], %[[ARG_0]])
// CHECK-NEXT:     %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_D]], %[[OP_B]])
// CHECK-NEXT:     %[[OP_F:[0-9]*]] = "tf.opF"(%[[OP_E]], %[[OP_E]])
// CHECK-NEXT:     tf_executor.yield %[[OP_F]] : tensor<i32>
// CHECK:        tf_executor.fetch %[[ISLAND]] : tensor<i32>


// Test if islands can be merged when non dependent islands are interleaved.
// CHECK-LABEL: func @islands_interleaved
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i32>, %[[ARG_1:[a-z0-9]*]]: tensor<i32>)
func @islands_interleaved(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.opA"(%arg0) : (tensor<i32>) -> tensor<i32>
    %2:2 = tf_executor.island wraps "tf.opB"(%arg1) : (tensor<i32>) -> tensor<i32>
    %3:2 = tf_executor.island wraps "tf.opC"(%1#0) : (tensor<i32>) -> tensor<i32>
    %4:2 = tf_executor.island wraps "tf.opD"(%2#0) : (tensor<i32>) -> tensor<i32>
    %5:2 = tf_executor.island(%3#1) wraps "tf.opE"(%arg0) : (tensor<i32>) -> tensor<i32>
    %6:2 = tf_executor.island wraps "tf.opF"(%arg1) : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch %4#0, %3#0 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// CHECK:        %[[ISLAND_0:.*]]:2, %{{.*}} = tf_executor.island {
// CHECK-NEXT:     %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:     %{{[0-9]*}} = "tf.opE"(%[[ARG_0]])
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"(%[[ARG_1]])
// CHECK-NEXT:     %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_B]])
// CHECK-NEXT:     tf_executor.yield %[[OP_D]], %[[OP_C]] : tensor<i32>, tensor<i32>
// CHECK:        tf_executor.island wraps "tf.opF"(%[[ARG_1]])
// CHECK:        tf_executor.fetch %[[ISLAND_0]]#0, %[[ISLAND_0]]#1 : tensor<i32>, tensor<i32>


// Test only islands are merged when other tf_executor ops are interleaved.
// CHECK-LABEL: func @merge_islands_only
func @merge_islands_only() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.opA"() : () -> tensor<i32>
    %1:2 = tf_executor.Enter %0#0 frame "while/while_context" : (tensor<i32>) -> (tensor<*xi32>, !tf_executor.control)
    %2 = tf_executor.island wraps "tf.opB"() : () -> ()
    %3:3 = tf_executor.NextIteration.Source : tensor<*xi32>
    %4:3 = tf_executor.Merge %3#0, %1#0 : tensor<*xi32>
    %5:2 = tf_executor.island(%4#2) wraps "tf.opC"() : () -> tensor<i32>
    %6:2 = tf_executor.island wraps "tf.opD"(%4#0, %5#0) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
    %7:2 = tf_executor.LoopCond %6#0 : (tensor<*xi1>) -> (tensor<i1>, !tf_executor.control)
    %8:3 = tf_executor.Switch %4#0, %7#0 : tensor<*xi32>
    %9:2 = tf_executor.Exit %8#0 : tensor<*xi32>
    %10:2 = tf_executor.island wraps "tf.opE"(%8#1) : (tensor<*xi32>) -> tensor<*xi32>
    %11:2 = tf_executor.island(%10#1) wraps "tf.opF"() : () -> tensor<i32>
    %12:2 = tf_executor.island wraps "tf.opG"(%10#0, %11#0) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %13 = tf_executor.ControlTrigger %2, %12#1, %9#1
    tf_executor.NextIteration.Sink [%3#1] %12#0, %13 : tensor<*xi32>
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[ISLAND_0:.*]], %[[ISLAND_0_control:.*]] = tf_executor.island wraps "tf.opA"
// CHECK:        %[[ENTER:.*]], %[[ENTER_control:.*]] = tf_executor.Enter %[[ISLAND_0]]
// CHECK-NEXT:   %[[ISLAND_1:.*]] = tf_executor.island wraps "tf.opB"()
// CHECK:        %[[NEXTIT_SRC:.*]], %[[NEXTIT_SRC_token:.*]], %[[NEXTIT_SRC_control:.*]] = tf_executor.NextIteration.Source
// CHECK-NEXT:   %[[MERGE:.*]], %[[MERGE_index:.*]], %[[MERGE_control:.*]] = tf_executor.Merge %[[NEXTIT_SRC]], %[[ENTER]]
// CHECK-NEXT:   %[[ISLAND_2:.*]], %[[ISLAND_2_control:.*]] = tf_executor.island(%[[MERGE_control]]) {
// CHECK-NEXT:     %[[OP_C:.*]] = "tf.opC"
// CHECK-NEXT:     %[[OP_D:[0-9]*]] = "tf.opD"(%[[MERGE]], %[[OP_C]])
// CHECK-NEXT:     tf_executor.yield %[[OP_D]] : tensor<*xi1>
// CHECK:        %[[COND:.*]], %[[COND_control:.*]] = tf_executor.LoopCond %[[ISLAND_2]]
// CHECK-NEXT:   %[[SWITCH_false:.*]], %[[SWITCH_true:.*]], %[[SWITCH_control:.*]] = tf_executor.Switch %[[MERGE]], %[[COND]]
// CHECK-NEXT:   %[[EXIT:.*]], %[[EXIT_control:.*]] = tf_executor.Exit %[[SWITCH_false]]
// CHECK-NEXT:   %[[ISLAND_3:.*]], %[[ISLAND_3_control:.*]] = tf_executor.island {
// CHECK-NEXT:     %[[OP_E:[0-9]*]] = "tf.opE"(%[[SWITCH_true]])
// CHECK-NEXT:     %[[OP_F:.*]] = "tf.opF"
// CHECK-NEXT:     %[[OP_G:[0-9]*]] = "tf.opG"(%[[OP_E]], %[[OP_F]])
// CHECK-NEXT:     tf_executor.yield %[[OP_G]] : tensor<*xi32>
// CHECK:        %[[CT:.*]] = tf_executor.ControlTrigger %[[ISLAND_1]], %[[ISLAND_3_control]], %[[EXIT_control]]
// CHECK-NEXT:   tf_executor.NextIteration.Sink [%[[NEXTIT_SRC_token]]] %[[ISLAND_3]], %[[CT]]


// Test no merging took place as cycle would be formed otherwise.
// CHECK-LABEL: func @simple_potential_cycle
func @simple_potential_cycle() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.opA"() : () -> tensor<1xf32>
    %1 = tf_executor.ControlTrigger %0#1
    %2:3 = tf_executor.island(%1) {
      %4 = "tf.opB"() : () -> tensor<1xf32>
      tf_executor.yield %0#0, %4 : tensor<1xf32>, tensor<1xf32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[ISLAND:.*]], %[[ISLAND_control:.*]] = tf_executor.island wraps "tf.opA"
// CHECK:        %[[CT:.*]] = tf_executor.ControlTrigger %[[ISLAND_control]]
// CHECK-NEXT:   tf_executor.island(%[[CT]]) {
// CHECK-NEXT:     %[[OP_B:[0-9]*]] = "tf.opB"
// CHECK-NEXT:     tf_executor.yield %[[ISLAND]], %[[OP_B]] : tensor<1xf32>, tensor<1xf32>


// Test if island was merged into its result.
// CHECK-LABEL: func @merge_into_result
func @merge_into_result() {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      %3 = "tf.opA"() : () -> tensor<1xf32>
      tf_executor.yield %3 : tensor<1xf32>
    }
    %1 = tf_executor.ControlTrigger {}
    %2:3 = tf_executor.island(%1) {
      %4 = "tf.opB"() : () -> tensor<1xf32>
      tf_executor.yield %0#0, %4 : tensor<1xf32>, tensor<1xf32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[CT:[0-9]*]] = tf_executor.ControlTrigger
// CHECK-NEXT:   tf_executor.island(%[[CT]]) {
// CHECK-NEXT:     "tf.opA"
// CHECK-NEXT:     "tf.opB"
// CHECK-NEXT:     tf_executor.yield


// Test merging island into data result nested in a graph of another island.
// CHECK-LABEL: func @merge_into_nested_data_result
func @merge_into_nested_data_result() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.opA"() : () -> tensor<1xf32>
    %2:2 = tf_executor.island {
      %3 = tf_executor.graph {
        %4 = tf_executor.ControlTrigger {}
        %5:2 = tf_executor.island(%4) wraps "tf.opB"(%0#0) : (tensor<1xf32>) -> tensor<1xf32>
        tf_executor.fetch %5#0 : tensor<1xf32>
      }
      tf_executor.yield %3 : tensor<1xf32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        tf_executor.island {
// CHECK-NEXT:     [[OP_A:[0-9*]]] = "tf.opA"
// CHECK-NEXT:     [[INNER_GRAPH:[0-9]*]] = tf_executor.graph {
// CHECK-NEXT:       [[CT:[0-9]*]] = tf_executor.ControlTrigger
// CHECK-NEXT:       %[[ISLAND_1:.*]], %{{.*}} = tf_executor.island(%[[CT]])
// CHECK-NEXT:         "tf.opB"(%[[OP_A]])
// CHECK:            tf_executor.fetch %[[ISLAND_1]] : tensor<1xf32>
// CHECK:          tf_executor.yield


// Test merging islands in a nested graph.
// CHECK-LABEL: func @merge_islands_inner_graph
func @merge_islands_inner_graph() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.opA"() : () -> tensor<1xf32>
    %2:2 = tf_executor.island {
      %3 = tf_executor.graph {
        %4:2 = tf_executor.island wraps "tf.opB"() : () -> tensor<1xf32>
        %6:2 = tf_executor.island wraps "tf.opC"() : () -> tensor<1xf32>
        %8:2 = tf_executor.island(%4#1) wraps "tf.opD"(%6#0) : (tensor<1xf32>) -> tensor<1xf32>
        tf_executor.fetch %8#0 : tensor<1xf32>
      }
      tf_executor.yield %3 : tensor<1xf32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        tf_executor.island wraps "tf.opA"
// CHECK:        tf_executor.island
// CHECK:          tf_executor.graph {
// CHECK-NEXT:       %[[ISLAND_1:.*]], %{{.*}} = tf_executor.island {
// CHECK-NEXT:         "tf.opB"
// CHECK-NEXT:         [[OP_C:[0-9]*]] = "tf.opC"
// CHECK-NEXT:         [[OP_D:[0-9]*]] = "tf.opD"(%[[OP_C]])
// CHECK-NEXT:         tf_executor.yield %[[OP_D]] : tensor<1xf32>
// CHECK:            tf_executor.fetch %[[ISLAND_1]] : tensor<1xf32>


// Test merging islands with control island operands and island results only if
// they are the closest ones.
// CHECK-LABEL: func @merge_islands_closest_control
func @merge_islands_closest_control() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    %1 = tf_executor.ControlTrigger %0
    %2 = tf_executor.ControlTrigger {}
    %3 = tf_executor.island(%0, %2) {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: %[[ISLAND:.*]] = tf_executor.island
// CHECK: tf_executor.ControlTrigger %[[ISLAND]]
// CHECK: %[[CT:[0-9]*]] = tf_executor.ControlTrigger
// CHECK: tf_executor.island(%[[ISLAND]], %[[CT]])

// Test that islands which independently feed into a fetch are merged.
// CHECK-LABEL: func @merge_independently_fetched_islands
func @merge_independently_fetched_islands(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i1>) {
  %0:2 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2:2 = tf_executor.island wraps "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    tf_executor.fetch %1#0, %2#0 : tensor<i32>, tensor<i1>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i1>
}

// CHECK:      tf_executor.island
// CHECK-NEXT:   "tf.Const"
// CHECK-NEXT:   "tf.Const"
// CHECK-NEXT:   tf_executor.yield


// Check that we merge two islands with independent fetches, when one is a data
// fetch and the other is a control fetch.
// CHECK-LABEL: func @merge_independently_fetched_islands_data_and_control
func @merge_independently_fetched_islands_data_and_control(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2 = tf_executor.island { tf_executor.yield }
    tf_executor.fetch %1#0, %2 : tensor<i32>, !tf_executor.control
  }
  return %0#0 : tensor<i32>
}

// CHECK:      tf_executor.graph
// CHECK-NEXT:   %[[ISLAND:.*]], %[[ISLAND_control:.*]] = tf_executor.island {
// CHECK-NEXT:     "tf.Const"
// CHECK-NEXT:     tf_executor.yield
// CHECK-NEXT:   }
// CHECK-NEXT:   tf_executor.fetch %[[ISLAND]], %[[ISLAND_control]]

// Check that we merge two islands with independent fetches, when both are
// control fetches.
// CHECK-LABEL: func @merge_independently_fetched_islands_control_and_control
func @merge_independently_fetched_islands_control_and_control(%arg0: tensor<i32>) {
 tf_executor.graph {
    %1 = tf_executor.island { tf_executor.yield }
    %2 = tf_executor.island { tf_executor.yield }
    tf_executor.fetch %1, %2 : !tf_executor.control, !tf_executor.control
  }
  return
}

// CHECK:     %[[ISLAND:.*]] = tf_executor.island
// CHECK-NOT: tf_executor.island
// CHECK:     tf_executor.fetch %[[ISLAND]]

