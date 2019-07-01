// RUN: tf-opt %s | tf-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @control_type() -> !tf_executor.control
func @control_type() -> !tf_executor.control

// CHECK-LABEL: func @empty_graph
func @empty_graph() {
  tf_executor.graph {
  }
  return

// CHECK:      tf_executor.graph {
// CHECK-NEXT:  tf_executor.fetch
// CHECK-NEXT: }

}

// CHECK-LABEL: func @graph_with_fetch(%arg0: tensor<*xf32>)
func @graph_with_fetch(%0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    tf_executor.fetch %0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @graph_with_fetch_attributes(%arg0: tensor<*xf32>)
func @graph_with_fetch_attributes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
  } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}

// CHECK:      tf_executor.graph {
// CHECK-NEXT:    tf_executor.fetch %arg0 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
// CHECK-NEXT: } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @simpleIsland(%arg0: tensor<*xf32>)
func @simpleIsland(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      tf_executor.yield %arg0 : tensor<*xf32>
    }
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
// CHECK: tf_executor.island {
// CHECK:   tf_executor.yield %arg0 : tensor<*xf32>
// CHECK: tf_executor.fetch %1#0 : tensor<*xf32>

  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @simpleIsland_with_attributes(%arg0: tensor<*xf32>)
func @simpleIsland_with_attributes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      tf_executor.yield %arg0 : tensor<*xf32>  {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}

// CHECK: tf_executor.yield %arg0 : tensor<*xf32>  {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
// CHECK-NEXT: } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}

    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @fetchWithControlDep(%arg0: tensor<*xf32>)
func @fetchWithControlDep(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %val, %ctl_id = tf_executor.island {
      %val = addf %arg0, %arg0 : tensor<*xf32>
      tf_executor.yield %arg0 : tensor<*xf32>
    }
// CHECK: tf_executor.fetch %1#0, %1#1 : tensor<*xf32>, !tf_executor.control
    tf_executor.fetch %val, %ctl_id : tensor<*xf32>, !tf_executor.control
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @testAddWithControlDependency(%arg0: tensor<*xf32>)
func @testAddWithControlDependency(%0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    // This identity operation is unused, but the control dependency on the
    // island is consumed by the add right after.
    %unused, %ctl_id = tf_executor.island {
      %id_in = "tf.Identity"(%0) : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.yield %id_in : tensor<*xf32>
    }

    // The control dependency is held by the operand.
// CHECK: tf_executor.island(%1#1)
    %add, %clt_add = tf_executor.island(%ctl_id) {
      %add_in= "tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
      tf_executor.yield %add_in : tensor<*xf32>
    }

    tf_executor.fetch %add : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_broadcast(%arg0: tensor<2xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch_with_broadcast(%arg0: tensor<2xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_broadcast_one_output(%arg0: tensor<2xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch_with_broadcast_one_output(%arg0: tensor<2xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_broadcast_output_only(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch_with_broadcast_output_only(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : (tensor<*xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : (tensor<*xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_attributes(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch_with_attributes(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switchN(%arg0: i32, %arg1: tensor<*xf32>) -> tensor<*xf32> {
func @switchN(%arg0: i32, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %fetches = tf_executor.graph {

// CHECK: %1:6 = tf_executor.SwitchN %arg1, %arg0 of 5 : tensor<*xf32>
     %1:6 = tf_executor.SwitchN %arg1, %arg0 of 5 : tensor<*xf32>

// CHECK: %2:13 = tf_executor.SwitchN %arg1, %arg0 of 12 (%1#5) : tensor<*xf32>
     %2:13 = tf_executor.SwitchN %arg1, %arg0 of 12 (%1#5) : tensor<*xf32>

     tf_executor.fetch %2#0 : tensor<*xf32>
  }
  return %fetches : tensor<*xf32>
}

// CHECK-LABEL: func @switch_merge(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch_merge(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

// CHECK: tf_executor.Merge %1#0, %1#1 : tensor<*xf32>
    %value, %idx, %ctlMerge = tf_executor.Merge %true, %false : tensor<*xf32>

    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_merge_with_ctl(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch_merge_with_ctl(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

// CHECK: tf_executor.Merge %1#0, %1#1, %1#2, %1#2, %1#2 : tensor<*xf32>
    %value, %idx, %ctlMerge = tf_executor.Merge %true, %false, %ctlSwitch, %ctlSwitch, %ctlSwitch : tensor<*xf32>

    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_merge_with_attributes(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @switch_merge_with_attributes(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

// CHECK: tf_executor.Merge %1#0, %1#1 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %value, %idx, %ctlMerge = tf_executor.Merge %true, %false : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}

    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
func @enter(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %arg0 frame "some/fra\22me" : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/fra\"me" : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_broadcast(%arg0: tensor<8xf32>, %arg1: i1) -> tensor<*xf32> {
func @enter_broadcast(%arg0: tensor<8xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %arg0 frame "some/fra\22me" : (tensor<8xf32>) -> (tensor<*xf32>, !tf_executor.control)
    %res:2 = tf_executor.Enter %arg0 frame "some/fra\"me" : (tensor<8xf32>) -> (tensor<*xf32>, !tf_executor.control)
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_parallel_iterations(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
func @enter_parallel_iterations(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %arg0 frame "some/frame" parallel_iterations 42 : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" parallel_iterations 42 : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_constant(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
func @enter_constant(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %arg0 frame "some/frame" constant : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" constant : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_parallel_iterations_constant(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
func @enter_parallel_iterations_constant(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %arg0 frame "some/frame" parallel_iterations 42 constant : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" parallel_iterations 42 constant : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_with_attributes(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
func @enter_with_attributes(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %arg0 frame "some/frame" : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_control(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
func @enter_control(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
// CHECK: tf_executor.Enter %arg0, %1#2, %1#2 frame "some/frame" : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0, %1#2, %1#2 frame "some/frame" : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}
