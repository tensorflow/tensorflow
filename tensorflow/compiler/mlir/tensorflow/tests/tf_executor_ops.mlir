// RUN: tf-opt %s | tf-opt | FileCheck %s

// CHECK-LABEL: func private @control_type() -> !tf_executor.control
func private @control_type() -> !tf_executor.control

// CHECK-LABEL: func private @token_type() -> !tf_executor.token
func private @token_type() -> !tf_executor.token

// CHECK-LABEL: func @empty_graph
func @empty_graph() {
  tf_executor.graph {
  }
  return

// CHECK:      tf_executor.graph {
// CHECK-NEXT:  tf_executor.fetch
// CHECK-NEXT: }

}

// CHECK-LABEL: func @graph_with_fetch(%{{.*}}: tensor<*xf32>)
func @graph_with_fetch(%0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    tf_executor.fetch %0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @graph_with_fetch_attributes(%{{.*}}: tensor<*xf32>)
func @graph_with_fetch_attributes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
  } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}

// CHECK:      tf_executor.graph {
// CHECK-NEXT:    tf_executor.fetch %{{.*}} : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
// CHECK-NEXT: } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @simpleIsland(%{{.*}}: tensor<*xf32>)
func @simpleIsland(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      tf_executor.yield %arg0 : tensor<*xf32>
    }
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
// CHECK: tf_executor.island {
// CHECK:   tf_executor.yield %{{.*}} : tensor<*xf32>
// CHECK: tf_executor.fetch {{.*}} : tensor<*xf32>

  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @simpleIsland_with_attributes(%{{.*}}: tensor<*xf32>)
func @simpleIsland_with_attributes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      tf_executor.yield %arg0 : tensor<*xf32>  {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}

// CHECK: tf_executor.yield %{{.*}} : tensor<*xf32>  {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
// CHECK-NEXT: } {attr2 = 32 : i64, tf_executor.attr1 = "value1"}

    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @simpleIsland_with_multiple_control_inputs(%{{.*}}: tensor<*xf32>)
func @simpleIsland_with_multiple_control_inputs(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1 = tf_executor.island {
      tf_executor.yield
    }
    %2 = tf_executor.island {
      tf_executor.yield
    }
    %3:2 = tf_executor.island(%1, %2) {
      tf_executor.yield %arg0 : tensor<*xf32>
    }
    tf_executor.fetch %3#0 : tensor<*xf32>
  }
// CHECK:      %[[ISLAND0:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:   tf_executor.yield
// CHECK:      %[[ISLAND1:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:   tf_executor.yield
// CHECK:      %[[ISLAND2:.*]], %[[ISLAND2_control:.*]] = tf_executor.island(%[[ISLAND0]], %[[ISLAND1]]) {
// CHECK:      tf_executor.fetch %[[ISLAND2]] : tensor<*xf32>

  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @fetchWithControlDep(%{{.*}}: tensor<*xf32>)
func @fetchWithControlDep(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    %val, %ctl_id = tf_executor.island {
      %val = addf %arg0, %arg0 : tensor<*xf32>
      tf_executor.yield %arg0 : tensor<*xf32>
    }
// CHECK: tf_executor.fetch %{{.*}}, %{{.*}} : tensor<*xf32>, !tf_executor.control
    tf_executor.fetch %val, %ctl_id : tensor<*xf32>, !tf_executor.control
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @testAddWithControlDependency(%{{.*}}: tensor<*xf32>)
func @testAddWithControlDependency(%0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {
    // This identity operation is unused, but the control dependency on the
    // island is consumed by the add right after.
    %unused, %ctl_id = tf_executor.island {
      %id_in = "tf.Identity"(%0) : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.yield %id_in : tensor<*xf32>
    }

    // The control dependency is held by the operand.
// CHECK: tf_executor.island(%{{.*}})
    %add, %clt_add = tf_executor.island(%ctl_id) {
      %add_in= "tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
      tf_executor.yield %add_in : tensor<*xf32>
    }

    tf_executor.fetch %add : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_broadcast(%{{.*}}: tensor<2xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch_with_broadcast(%arg0: tensor<2xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_broadcast_one_output(%{{.*}}: tensor<2xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch_with_broadcast_one_output(%arg0: tensor<2xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : (tensor<2xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_broadcast_output_only(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch_with_broadcast_output_only(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : (tensor<*xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : (tensor<*xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<2xf32>, !tf_executor.control)
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_attributes(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch_with_attributes(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_unranked_pred(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<*xi1>) -> tensor<*xf32> {
func @switch_with_unranked_pred(%arg0: tensor<*xf32>, %arg1: tensor<*xi1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : (tensor<*xf32>, tensor<*xi1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : (tensor<*xf32>, tensor<*xi1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
    tf_executor.fetch %true : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_with_control_inputs(
func @switch_with_control_inputs(%arg0: tensor<i1>, %arg1: !tf_executor.control, %arg2: !tf_executor.control) -> tensor<i1> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{[^%]*}}, %{{[^%]*}}, %{{[^%]*}}, %{{[^%]*}} : tensor<i1>
    %1:3 = tf_executor.Switch %arg0, %arg0, %arg1, %arg2 : tensor<i1>
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %result : tensor<i1>
}

// CHECK-LABEL: func @switch_with_control_inputs_functional(
func @switch_with_control_inputs_functional(%arg0: tensor<i1>, %arg1: !tf_executor.control, %arg2: !tf_executor.control) -> tensor<i1> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{[^%]*}}, %{{[^%]*}}, %{{[^%]*}}, %{{[^%]*}} : tensor<i1>
    %1:3 = tf_executor.Switch %arg0, %arg0, %arg1, %arg2 : (tensor<i1>, tensor<i1>, !tf_executor.control, !tf_executor.control) -> (tensor<i1>, tensor<i1>, !tf_executor.control)
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %result : tensor<i1>
}

// CHECK-LABEL: func @switchN(
func @switchN(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %fetches = tf_executor.graph {

// CHECK: tf_executor._SwitchN %{{.*}}, %{{.*}} of 5 : tensor<*xf32>
     %1:6 = tf_executor._SwitchN %arg1, %arg0 of 5 : tensor<*xf32>

// CHECK: tf_executor._SwitchN %{{.*}}, %{{.*}} of 12 (%{{.*}}) : tensor<*xf32>
     %2:13 = tf_executor._SwitchN %arg1, %arg0 of 12 (%1#5) : tensor<*xf32>

     tf_executor.fetch %2#0 : tensor<*xf32>
  }
  return %fetches : tensor<*xf32>
}

// CHECK-LABEL: func @switch_merge(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch_merge(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

// CHECK: tf_executor.Merge %{{.*}}, %{{.*}} : tensor<*xf32>
    %value, %idx, %ctlMerge = tf_executor.Merge %true, %false : tensor<*xf32>

    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_merge_with_ctl(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch_merge_with_ctl(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

// CHECK: tf_executor.Merge %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<*xf32>
    %value, %idx, %ctlMerge = tf_executor.Merge %true, %false, %ctlSwitch, %ctlSwitch, %ctlSwitch : tensor<*xf32>

    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @switch_merge_with_attributes(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @switch_merge_with_attributes(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Switch %{{.*}}, %{{.*}} : tensor<*xf32>
    %true, %false, %ctlSwitch = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>

// CHECK: tf_executor.Merge %{{.*}}, %{{.*}} : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %value, %idx, %ctlMerge = tf_executor.Merge %true, %false : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}

    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// Verify that long form printing is used when operand types do not match the
// result type and then it can be parsed again correctly.
// CHECK-LABEL: func @merge_different_operand_types
func @merge_different_operand_types(%arg0: tensor<*xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %result = tf_executor.graph {

// CHECK: tf_executor.Merge{{.*}}(tensor<*xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<i32>, !tf_executor.control)
    %value, %idx, %ctlMerge = tf_executor.Merge %arg0, %arg1  : (tensor<*xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<i32>, !tf_executor.control)
    tf_executor.fetch %value : tensor<4xf32>
  }
  return %result : tensor<4xf32>
}

// Verify that long form printing is used when there is only one data operand
// and then it can be parsed again correctly.
// CHECK-LABEL: func @merge_one_data_operand
func @merge_one_data_operand(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {

// CHECK: tf_executor.Merge{{.*}}(tensor<*xf32>) -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
    %value, %idx, %ctlMerge = tf_executor.Merge %arg0  : (tensor<*xf32>) -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @merge_with_variant_type
func @merge_with_variant_type(%arg0: tensor<!tf.variant>, %arg1: tensor<!tf.variant<tensor<4xi32>>>) -> tensor<!tf.variant<tensor<8xf32>>> {
  %result = tf_executor.graph {

// CHECK: tf_executor.Merge{{.*}}(tensor<!tf.variant>, tensor<!tf.variant<tensor<4xi32>>>) -> (tensor<!tf.variant<tensor<8xf32>>>, tensor<i32>, !tf_executor.control)
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<!tf.variant>, tensor<!tf.variant<tensor<4xi32>>>) -> (tensor<!tf.variant<tensor<8xf32>>>, tensor<i32>, !tf_executor.control)
    tf_executor.fetch %value : tensor<!tf.variant<tensor<8xf32>>>
  }
  return %result : tensor<!tf.variant<tensor<8xf32>>>
}

// CHECK-LABEL: func @merge_with_resource_type
func @merge_with_resource_type(%arg0: tensor<!tf.resource>, %arg1: tensor<!tf.resource<tensor<4xi32>>>) -> tensor<!tf.resource<tensor<8xf32>>> {
  %result = tf_executor.graph {

// CHECK: tf_executor.Merge{{.*}}(tensor<!tf.resource>, tensor<!tf.resource<tensor<4xi32>>>) -> (tensor<!tf.resource<tensor<8xf32>>>, tensor<i32>, !tf_executor.control)
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<!tf.resource>, tensor<!tf.resource<tensor<4xi32>>>) -> (tensor<!tf.resource<tensor<8xf32>>>, tensor<i32>, !tf_executor.control)
    tf_executor.fetch %value : tensor<!tf.resource<tensor<8xf32>>>
  }
  return %result : tensor<!tf.resource<tensor<8xf32>>>
}

// CHECK-LABEL: func @merge_with_ref_type
func @merge_with_ref_type(%arg0: tensor<4x!tf.f32ref>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %result = tf_executor.graph {

// CHECK: tf_executor.Merge{{.*}}(tensor<4x!tf.f32ref>, tensor<4xf32>) -> (tensor<4xf32>, tensor<i32>, !tf_executor.control)
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<4x!tf.f32ref>, tensor<4xf32>) -> (tensor<4xf32>, tensor<i32>, !tf_executor.control)
    tf_executor.fetch %value : tensor<4xf32>
  }
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func @merge_with_dynamic_shape
func @merge_with_dynamic_shape(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> tensor<?xf32> {
  %result = tf_executor.graph {

// CHECK: tf_executor.Merge{{.*}}(tensor<2xf32>, tensor<3xf32>) -> (tensor<?xf32>, tensor<i32>, !tf_executor.control)
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<2xf32>, tensor<3xf32>) -> (tensor<?xf32>, tensor<i32>, !tf_executor.control)
    tf_executor.fetch %value : tensor<?xf32>
  }
  return %result : tensor<?xf32>
}

// CHECK-LABEL: func @merge_with_unranked_shape
func @merge_with_unranked_shape(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> tensor<*xf32> {
  %result = tf_executor.graph {

// CHECK: tf_executor.Merge{{.*}}(tensor<2xf32>, tensor<3xf32>) -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
    %value, %idx, %ctlMerge = "tf_executor.Merge"(%arg0, %arg1) : (tensor<2xf32>, tensor<3xf32>) -> (tensor<*xf32>, tensor<i32>, !tf_executor.control)
    tf_executor.fetch %value : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter(%{{.*}}: tensor<*xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @enter(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %{{.*}} frame "some/fra\22me" : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/fra\"me" : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_broadcast(%{{.*}}: tensor<8xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @enter_broadcast(%arg0: tensor<8xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %{{.*}} frame "some/fra\22me" : (tensor<8xf32>) -> (tensor<*xf32>, !tf_executor.control)
    %res:2 = tf_executor.Enter %arg0 frame "some/fra\"me" : (tensor<8xf32>) -> (tensor<*xf32>, !tf_executor.control)
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_parallel_iterations(%{{.*}}: tensor<*xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @enter_parallel_iterations(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %{{.*}} frame "some/frame" parallel_iterations 42 : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" parallel_iterations 42 : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_constant(%{{.*}}: tensor<*xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @enter_constant(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %{{.*}} frame "some/frame" constant : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" constant : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_parallel_iterations_constant(%{{.*}}: tensor<*xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @enter_parallel_iterations_constant(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %{{.*}} frame "some/frame" parallel_iterations 42 constant : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" parallel_iterations 42 constant : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_with_attributes(%{{.*}}: tensor<*xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @enter_with_attributes(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %result = tf_executor.graph {
// CHECK: tf_executor.Enter %{{.*}} frame "some/frame" : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %res:2 = tf_executor.Enter %arg0 frame "some/frame" : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %result : tensor<*xf32>
}

// CHECK-LABEL: func @enter_control(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @enter_control(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
// CHECK: tf_executor.Enter %{{.*}}, %{{.*}}, %{{.*}} frame "some/frame" : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0, %1#2, %1#2 frame "some/frame" : tensor<*xf32>
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @enter_control_longform(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @enter_control_longform(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
// CHECK: tf_executor.Enter %{{.*}}, %{{.*}}, %{{.*}} frame "some/frame" : tensor<*xf32>
    %res:2 = tf_executor.Enter %arg0, %1#2, %1#2 frame "some/frame" : (tensor<*xf32>, !tf_executor.control, !tf_executor.control) -> (tensor<*xf32>, !tf_executor.control)
    tf_executor.fetch %res#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @nextiteration(%{{.*}}: tensor<*xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @nextiteration(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : tensor<*xf32>
    tf_executor.NextIteration.Sink[%1#1] %1#0 : tensor<*xf32>
// CHECK: tf_executor.NextIteration.Source : tensor<*xf32>
// CHECK: tf_executor.NextIteration.Sink[%{{.*}}] %{{.*}} : tensor<*xf32>
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @nextiteration_with_attributes(%{{.*}}: tensor<*xf32>, %{{.*}}: i1) -> tensor<*xf32> {
func @nextiteration_with_attributes(%arg0: tensor<*xf32>, %arg1: i1) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.NextIteration.Source : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    tf_executor.NextIteration.Sink[%1#1] %1#0 : tensor<*xf32> {attr4 = 42 : i64, tf_executor.attr_push = "other_value"}
// CHECK: tf_executor.NextIteration.Source : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
// CHECK: tf_executor.NextIteration.Sink[%{{.*}}] %{{.*}} : tensor<*xf32> {attr4 = 42 : i64, tf_executor.attr_push = "other_value"}
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @nextiteration_control(%{{.*}}: tensor<*xf32>, %{{.*}}: tensor<i1>) -> tensor<*xf32> {
func @nextiteration_control(%arg0: tensor<*xf32>, %arg1: tensor<i1>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %arg0, %arg1 : tensor<*xf32>
    %2:2 = tf_executor.Enter %arg0, %1#2, %1#2 frame "some/frame" : tensor<*xf32>
    %3:3 = tf_executor.NextIteration.Source : tensor<*xf32>
    tf_executor.NextIteration.Sink[%3#1] %3#0, %1#2 : tensor<*xf32>
// CHECK: tf_executor.NextIteration.Source : tensor<*xf32>
// CHECK: tf_executor.NextIteration.Sink[%{{.*}}] %{{.*}}, %{{.*}} : tensor<*xf32>
    tf_executor.fetch %3#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @exit(%{{.*}}: tensor<*xf32>) -> tensor<*xf32> {
func @exit(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
// CHECK: tf_executor.Exit %{{.*}} : tensor<*xf32>
    %1:2 = tf_executor.Exit %arg0 : tensor<*xf32>
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @exit_with_attributes(%{{.*}}: tensor<*xf32>) -> tensor<*xf32> {
func @exit_with_attributes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
// CHECK: tf_executor.Exit %{{.*}} : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %1:2 = tf_executor.Exit %arg0 : tensor<*xf32> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    tf_executor.fetch %1#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @exit_with_control(%{{.*}}: tensor<*xf32>, %{{.*}}: !tf_executor.control) -> tensor<*xf32> {
func @exit_with_control(%arg0: tensor<*xf32>, %arg1: !tf_executor.control) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.Exit %arg0, %arg1 : tensor<*xf32>
    %2:2 = tf_executor.Exit %arg0, %1#1 : tensor<*xf32>
    tf_executor.fetch %2#0 : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @control_trigger(%{{.*}}: !tf_executor.control, %{{.*}}: !tf_executor.control) {
func @control_trigger(%arg0: !tf_executor.control, %arg1: !tf_executor.control) {
  tf_executor.graph {
// CHECK: tf_executor.ControlTrigger %{{.*}}, %{{.*}}
    %0 = tf_executor.ControlTrigger %arg0, %arg1
  }
  return
}

// CHECK-LABEL: func @control_trigger_with_attributes(%{{.*}}: !tf_executor.control, %{{.*}}: !tf_executor.control) {
func @control_trigger_with_attributes(%arg0: !tf_executor.control, %arg1: !tf_executor.control) {
  tf_executor.graph {
// CHECK: tf_executor.ControlTrigger %{{.*}}, %{{.*}} {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %0 = tf_executor.ControlTrigger %arg0, %arg1 {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
  }
  return
}

// CHECK-LABEL: func @loop_cond(%{{.*}}: tensor<i1>, %{{.*}}: !tf_executor.control) -> tensor<i1> {
func @loop_cond(%arg0: tensor<i1>, %arg1: !tf_executor.control) -> tensor<i1> {
  %0 = tf_executor.graph {
// CHECK: tf_executor.LoopCond %{{.*}} : tensor<i1>
    %1:2 = tf_executor.LoopCond %arg0 : tensor<i1>
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK-LABEL: func @loop_cond_with_attributes(%{{.*}}: tensor<i1>, %{{.*}}: !tf_executor.control) -> tensor<i1> {
func @loop_cond_with_attributes(%arg0: tensor<i1>, %arg1: !tf_executor.control) -> tensor<i1> {
  %0 = tf_executor.graph {
// CHECK: tf_executor.LoopCond %{{.*}} : tensor<i1> {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    %1:2 = tf_executor.LoopCond %arg0 : tensor<i1>  {attr3 = 32 : i64, tf_executor.attr_fetch = "some_value"}
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK-LABEL: func @loop_cond_with_control(%{{.*}}: tensor<i1>, %{{.*}}: !tf_executor.control) -> tensor<i1> {
func @loop_cond_with_control(%arg0: tensor<i1>, %arg1: !tf_executor.control) -> tensor<i1> {
  %0 = tf_executor.graph {
// CHECK: tf_executor.LoopCond %{{.*}}, %{{.*}} : tensor<i1>
    %1:2 = tf_executor.LoopCond %arg0, %arg1 : tensor<i1>
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK-LABEL: func @loop_cond_with_control_broadcast(%{{.*}}: tensor<i1>, %{{.*}}: !tf_executor.control) -> tensor<*xi1> {
func @loop_cond_with_control_broadcast(%arg0: tensor<i1>, %arg1: !tf_executor.control) -> tensor<*xi1> {
  %0 = tf_executor.graph {
// CHECK: tf_executor.LoopCond %{{.*}}, %{{.*}} : (tensor<i1>, !tf_executor.control) -> (tensor<*xi1>, !tf_executor.control)
    %1:2 = tf_executor.LoopCond %arg0, %arg1 : (tensor<i1>, !tf_executor.control) -> (tensor<*xi1>, !tf_executor.control)
    tf_executor.fetch %1#0 : tensor<*xi1>
  }
  return %0 : tensor<*xi1>
}
