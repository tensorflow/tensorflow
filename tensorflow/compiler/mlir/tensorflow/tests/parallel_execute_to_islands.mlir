// RUN: tf-opt %s -tf-parallel-execute-to-islands | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @check_regions_to_islands
func @check_regions_to_islands() {
  tf_executor.graph {
    tf_executor.island() {
      "tf_device.parallel_execute"() ({
        tf_device.return
      },
      {
        tf_device.return
      }) {} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      %[[ISLAND_INPUT_CTL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:   tf_executor.yield
// CHECK:      %[[ISLAND_1_CTL:[a-z_0-9]*]] = tf_executor.island(%[[ISLAND_INPUT_CTL]]) {
// CHECK:        tf_executor.yield
// CHECK:      %[[ISLAND_2_CTL:[a-z_0-9]*]] = tf_executor.island(%[[ISLAND_INPUT_CTL]]) {
// CHECK:        tf_executor.yield
// CHECK:      %{{.*}} = tf_executor.island(%[[ISLAND_1_CTL]], %[[ISLAND_2_CTL]]) {
// CHECK-NEXT:   tf_executor.yield


// CHECK-LABEL: func @check_regions_to_islands_with_inputs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @check_regions_to_islands_with_inputs(%arg0 : tensor<i1>) {
  tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %2 : tensor<i1>
    }
    tf_executor.island() {
      "tf_device.parallel_execute"() ({
        %3 = "tf.opB"(%1#0) : (tensor<i1>) -> tensor<i1>
        tf_device.return %3 : tensor<i1>
      },
      {
        %5 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
        tf_device.return %5 : tensor<i32>
      }) {} : () -> (tensor<i1>, tensor<i32>)
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:       %[[INPUT_A:[a-z_0-9]*]], %{{.*}} = tf_executor.island {
// CHECK-NEXT:    %[[OP_A_OUTPUT:[a-z_0-9]*]] = "tf.opA"(%[[ARG_0]]) : (tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    tf_executor.yield %[[OP_A_OUTPUT]] : tensor<i1>
// CHECK:       %[[INPUT_0:[a-z_0-9]*]], %[[INPUT_CONTROL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:    tf_executor.yield %[[INPUT_A]] : tensor<i1>
// CHECK:       %[[ISLAND_1_OUTPUT:[a-z_0-9]*]], %[[ISLAND_1_CTL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:    %[[OP_B_OUTPUT:[a-z_0-9]*]] = "tf.opB"(%[[INPUT_0]]) : (tensor<i1>) -> tensor<i1>
// CHECK:         tf_executor.yield %[[OP_B_OUTPUT]] : tensor<i1>
// CHECK:      %[[ISLAND_2_OUTPUT:[a-z_0-9]*]], %[[ISLAND_2_CTL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:   %[[OP_C_OUTPUT:[a-z_0-9]*]] = "tf.opC"(%outputs_0) : (tensor<i1>) -> tensor<i32>
// CHECK:        tf_executor.yield %[[OP_C_OUTPUT]] : tensor<i32>
// CHECK:      %{{.*}} = tf_executor.island(%[[ISLAND_1_CTL]], %[[ISLAND_2_CTL]]) {
// CHECK-NEXT:   tf_executor.yield


// CHECK-LABEL: func @check_input_sink_island_forwards_control_inputs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @check_input_sink_island_forwards_control_inputs(%arg0 : tensor<i1>) {
  tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %2 : tensor<i1>
    }
    %7 = tf_executor.ControlTrigger {}
    %8 = tf_executor.ControlTrigger {}
    tf_executor.island(%7, %8) {
      "tf_device.parallel_execute"() ({
        %3 = "tf.opB"(%1#0) : (tensor<i1>) -> tensor<i1>
        tf_device.return %3 : tensor<i1>
      },
      {
        %5 = "tf.opC"() : () -> tensor<i32>
        tf_device.return %5 : tensor<i32>
      }) {} : () -> (tensor<i1>, tensor<i32>)
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:       %[[INPUT_A:[a-z_0-9]*]], %{{.*}} = tf_executor.island {
// CHECK-NEXT:    %[[OP_A_OUTPUT:[a-z_0-9]*]] = "tf.opA"(%[[ARG_0]]) : (tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    tf_executor.yield %[[OP_A_OUTPUT]] : tensor<i1>
// CHECK: %[[CT_0:[0-9]*]] = tf_executor.ControlTrigger
// CHECK: %[[CT_1:[0-9]*]] = tf_executor.ControlTrigger
// CHECK:       %[[INPUT_0:[a-z_0-9]*]], %[[INPUT_CONTROL:[a-z_0-9]*]] = tf_executor.island(%[[CT_0]], %[[CT_1]]) {
// CHECK-NEXT:    tf_executor.yield %[[INPUT_A]] : tensor<i1>
// CHECK:       %[[ISLAND_1_OUTPUT:[a-z_0-9]*]], %[[ISLAND_1_CTL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:    %[[OP_B_OUTPUT:[a-z_0-9]*]] = "tf.opB"(%[[INPUT_0]]) : (tensor<i1>) -> tensor<i1>
// CHECK:         tf_executor.yield %[[OP_B_OUTPUT]] : tensor<i1>
// CHECK:      %[[ISLAND_2_OUTPUT:[a-z_0-9]*]], %[[ISLAND_2_CTL:[a-z_0-9]*]] = tf_executor.island(%[[INPUT_CONTROL]]) {
// CHECK-NEXT:   %[[OP_C_OUTPUT:[a-z_0-9]*]] = "tf.opC"() : () -> tensor<i32>
// CHECK:        tf_executor.yield %[[OP_C_OUTPUT]] : tensor<i32>
// CHECK:      %{{.*}} = tf_executor.island(%[[ISLAND_1_CTL]], %[[ISLAND_2_CTL]]) {
// CHECK-NEXT:   tf_executor.yield


// CHECK-LABEL: func @check_control_dep_added_when_region_does_not_have_inputs
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @check_control_dep_added_when_region_does_not_have_inputs(%arg0 : tensor<i1>) {
  tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %2 : tensor<i1>
    }
    %7:3 = tf_executor.island() {
      %8:2 = "tf_device.parallel_execute"() (
      {
        %3 = "tf.opB"() : () -> tensor<i1>
        tf_device.return %3 : tensor<i1>
      },
      {
        %5 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
        tf_device.return %5 : tensor<i32>
       }
       ) {} : () -> (tensor<i1>, tensor<i32>)

      tf_executor.yield %8#0, %8#1 : tensor<i1>, tensor<i32>
    }

    tf_executor.island {
      "tf.opD"(%7#0, %7#1) : (tensor<i1>, tensor<i32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:       %[[INPUT_A:[a-z_0-9]*]], %{{.*}} = tf_executor.island {
// CHECK-NEXT:    %[[OP_A_OUTPUT:[a-z_0-9]*]] = "tf.opA"(%[[ARG_0]]) : (tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    tf_executor.yield %[[OP_A_OUTPUT]] : tensor<i1>
// CHECK:      %[[INPUT_0:[a-z_0-9]*]], %[[INPUT_CTL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:   tf_executor.yield %[[INPUT_A]] : tensor<i1>
// CHECK:      %[[ISLAND_1_OUTPUT:[a-z_0-9]*]], %{{.*}} = tf_executor.island(%[[INPUT_CTL]]) {
// CHECK-NEXT:   %[[OP_B_OUTPUT:[a-z_0-9]*]] = "tf.opB"() : () -> tensor<i1>
// CHECK:        tf_executor.yield %[[OP_B_OUTPUT]] : tensor<i1>
// CHECK:      %[[ISLAND_2_OUTPUT:[a-z_0-9]*]], %{{.*}} = tf_executor.island {
// CHECK-NEXT:   %[[OP_C_OUTPUT:[a-z_0-9]*]] = "tf.opC"(%outputs_0) : (tensor<i1>) -> tensor<i32>
// CHECK:        tf_executor.yield %[[OP_C_OUTPUT]] : tensor<i32>
// CHECK:      %{{.*}} = tf_executor.island {
// CHECK-NEXT:   tf_executor.yield %[[ISLAND_1_OUTPUT]], %[[ISLAND_2_OUTPUT]]


// CHECK-LABEL: func @check_output_barrier_correctly_forwards_outputs
func @check_output_barrier_correctly_forwards_outputs(%arg0 : tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %2 : tensor<i1>
    }
    %8:3 = tf_executor.island() {
      %7:2 = "tf_device.parallel_execute"() ({
        %3 = "tf.opB"() : () -> tensor<i1>
        tf_device.return %3 : tensor<i1>
      },
      {
        %5 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
        tf_device.return %5 : tensor<i32>
      }) {} : () -> (tensor<i1>, tensor<i32>)
      tf_executor.yield %7#0, %7#1 : tensor<i1>, tensor<i32>
    }
    tf_executor.fetch %8#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK:       %[[INPUT_A:[a-z_0-9]*]], %{{.*}} = tf_executor.island {
// CHECK-NEXT:    %[[OP_A_OUTPUT:[a-z_0-9]*]] = "tf.opA"(%[[ARG_0]]) : (tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    tf_executor.yield %[[OP_A_OUTPUT]] : tensor<i1>
// CHECK:       %[[INPUT_0:[a-z_0-9]*]], %[[INPUT_CTL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:    tf_executor.yield %[[INPUT_A]] : tensor<i1>
// CHECK:       %[[ISLAND_1_OUTPUT:[a-z_0-9]*]], %{{.*}} = tf_executor.island(%[[INPUT_CTL]]) {
// CHECK-NEXT:    %[[OP_B_OUTPUT:[a-z_0-9]*]] = "tf.opB"() : () -> tensor<i1>
// CHECK:         tf_executor.yield %[[OP_B_OUTPUT]] : tensor<i1>
// CHECK:       %[[ISLAND_2_OUTPUT:[a-z_0-9]*]], %{{.*}} = tf_executor.island {
// CHECK-NEXT:    %[[OP_C_OUTPUT:[a-z_0-9]*]] = "tf.opC"(%[[INPUT_0]]) : (tensor<i1>) -> tensor<i32>
// CHECK:         tf_executor.yield %[[OP_C_OUTPUT]] : tensor<i32>
// CHECK:       %[[OUTPUT_SINK_OUTPUT:[a-z_0-9]*]]:2, %[[OUTPUT_SINK_CTL:[a-z_0-9]*]] = tf_executor.island {
// CHECK-NEXT:    tf_executor.yield %[[ISLAND_1_OUTPUT]], %[[ISLAND_2_OUTPUT]] : tensor<i1>, tensor<i32>
