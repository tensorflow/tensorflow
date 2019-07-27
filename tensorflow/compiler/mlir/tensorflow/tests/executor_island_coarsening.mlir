// RUN: tf-opt %s -tf-executor-island-coarsening | FileCheck %s --dump-input=fail


// Test that islands linked by a control dependency are merged.
// CHECK-LABEL: func @control_input
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<i1>)
func @control_input(%arg0 : tensor<i1>) -> tensor<f32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %3 = "tf.Identity"(%arg0) {T = "tfdtype$DT_BOOL", device = "", name = "Identity"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island(%1#1) {
      %cst = "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "Const", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
      tf_executor.yield %cst : tensor<f32>
    }
    tf_executor.fetch %2#0 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// CHECK:        %[[ISLAND:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     "tf.Identity"(%[[ARG0]])
// CHECK-NEXT:     %[[CONST:.*]] = "tf.Const"
// CHECK-NEXT:     tf_executor.yield %[[CONST]] : tensor<f32>
// CHECK:        tf_executor.fetch %[[ISLAND]]#0 : tensor<f32>


// Test that islands linked by a data dependency are merged.
// CHECK-LABEL: func @data_input
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<i1>)
func @data_input(%arg0 : tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %3 = "tf.Identity"(%arg0) {T = "tfdtype$DT_BOOL", device = "", name = "Identity_0"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island {
      %4 = "tf.Identity"(%1#0) {T = "tfdtype$DT_BOOL", device = "", name = "Identity_1"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %4 : tensor<i1>
    }
    tf_executor.fetch %2#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK:        %[[ISLAND:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[IDENTITY0:[0-9]*]] = "tf.Identity"(%[[ARG0]])
// CHECK-NEXT:     %[[IDENTITY1:[0-9]*]] = "tf.Identity"(%[[IDENTITY0]])
// CHECK-NEXT:     tf_executor.yield %[[IDENTITY1]] : tensor<i1>
// CHECK:        tf_executor.fetch %[[ISLAND]]#0 : tensor<i1>


// Test empty/trivial islands are merged.
// CHECK-LABEL: func @empty_islands
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<i1>, %[[ARG1:[a-z0-9]*]]: tensor<i1>)
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
    %5:3 = tf_executor.island {
      %10:2 = "tf.IdentityN"(%3#0, %4#0) {T = ["tfdtype$DT_BOOL", "tfdtype$DT_BOOL"], device = "", name = "out"} : (tensor<i1>, tensor<i1>) -> (tensor<i1>, tensor<i1>)
      tf_executor.yield %10#0, %10#1 : tensor<i1>, tensor<i1>
    }
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

// CHECK:        %[[ISLAND:[0-9]*]]:3 = tf_executor.island {
// CHECK-NEXT:     %[[IDENTITYN:[0-9]*]]:2 = "tf.IdentityN"(%[[ARG1]], %[[ARG0]])
// CHECK-NEXT:     tf_executor.yield %[[IDENTITYN]]#0, %[[IDENTITYN]]#1 : tensor<i1>, tensor<i1>
// CHECK:        tf_executor.fetch %[[ISLAND]]#0, %[[ISLAND]]#1 : tensor<i1>, tensor<i1>


// Test merging islands handle merging results.
// CHECK-LABEL: func @multiple_outputs
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<i1>, %[[ARG1:[a-z0-9]*]]: tensor<i1>)
func @multiple_outputs(%arg0 : tensor<i1>, %arg1 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:2 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %3 = "tf.Identity"(%arg0) {T = "tfdtype$DT_BOOL", device = "", name = "Identity_0"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island(%1#1) {
      %4 = "tf.Identity"(%arg1) {T = "tfdtype$DT_BOOL", device = "", name = "Identity_1"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %4 : tensor<i1>
    }
    tf_executor.fetch %1#0, %2#0 : tensor<i1>, tensor<i1>
  }
  return %0#0, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK:        %[[ISLAND:[0-9]*]]:3 = tf_executor.island {
// CHECK-NEXT:     %[[IDENTITY0:[0-9]*]] = "tf.Identity"(%[[ARG0]])
// CHECK-NEXT:     %[[IDENTITY1:[0-9]*]] = "tf.Identity"(%[[ARG1]])
// CHECK-NEXT:     tf_executor.yield %[[IDENTITY0]], %[[IDENTITY1]] : tensor<i1>, tensor<i1>
// CHECK:        tf_executor.fetch %[[ISLAND]]#0, %[[ISLAND]]#1 : tensor<i1>, tensor<i1>


// Test merging islands with multiple inner ops.
// CHECK-LABEL: func @multi_op_regions
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<i32>, %[[ARG1:[a-z0-9]*]]: tensor<i32>)
func @multi_op_regions(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.Add"(%arg0, %arg0) {T = "tfdtype$DT_INT32", device = "", name = "Add_0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %3 = "tf.Add"(%2, %arg0) {T = "tfdtype$DT_INT32", device = "", name = "Add_1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %3 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %5 = "tf.Add"(%1#0, %arg1) {T = "tfdtype$DT_INT32", device = "", name = "Add_2"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %6 = "tf.Add"(%5, %arg0) {T = "tfdtype$DT_INT32", device = "", name = "Add_3"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %6 : tensor<i32>
    }
    tf_executor.fetch %4#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK:        %[[ISLAND:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[ADD0:[0-9]*]] = "tf.Add"(%[[ARG0]], %[[ARG0]])
// CHECK-NEXT:     %[[ADD1:[0-9]*]] = "tf.Add"(%[[ADD0]], %[[ARG0]])
// CHECK-NEXT:     %[[ADD2:[0-9]*]] = "tf.Add"(%[[ADD1]], %[[ARG1]])
// CHECK-NEXT:     %[[ADD3:[0-9]*]] = "tf.Add"(%[[ADD2]], %[[ARG0]])
// CHECK-NEXT:     tf_executor.yield %[[ADD3]] : tensor<i32>
// CHECK:        tf_executor.fetch %[[ISLAND]]#0 : tensor<i32>


// Test merging multiple islands with multiple inner ops preserves order.
// CHECK-LABEL: func @transitive_preserve_order
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<i32>, %[[ARG1:[a-z0-9]*]]: tensor<i32>)
func @transitive_preserve_order(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %2 = "tf.Add"(%arg0, %arg0) {T = "tfdtype$DT_INT32", device = "", name = "Add_0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %3 = "tf.Add"(%2, %arg0) {T = "tfdtype$DT_INT32", device = "", name = "Add_1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %3 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %5 = "tf.Add"(%1#0, %arg1) {T = "tfdtype$DT_INT32", device = "", name = "Add_2"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %6 = "tf.Add"(%5, %arg0) {T = "tfdtype$DT_INT32", device = "", name = "Add_3"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %6 : tensor<i32>
    }
    %7:2 = tf_executor.island {
      %8 = "tf.Add"(%4#0, %1#0) {T = "tfdtype$DT_INT32", device = "", name = "Add_4"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %9 = "tf.Add"(%8, %8) {T = "tfdtype$DT_INT32", device = "", name = "Add_5"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %9 : tensor<i32>
    }
    tf_executor.fetch %7#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK:        %[[ISLAND:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[ADD0:[0-9]*]] = "tf.Add"(%[[ARG0]], %[[ARG0]])
// CHECK-NEXT:     %[[ADD1:[0-9]*]] = "tf.Add"(%[[ADD0]], %[[ARG0]])
// CHECK-NEXT:     %[[ADD2:[0-9]*]] = "tf.Add"(%[[ADD1]], %[[ARG1]])
// CHECK-NEXT:     %[[ADD3:[0-9]*]] = "tf.Add"(%[[ADD2]], %[[ARG0]])
// CHECK-NEXT:     %[[ADD4:[0-9]*]] = "tf.Add"(%[[ADD3]], %[[ADD1]])
// CHECK-NEXT:     %[[ADD5:[0-9]*]] = "tf.Add"(%[[ADD4]], %[[ADD4]])
// CHECK-NEXT:     tf_executor.yield %[[ADD5]] : tensor<i32>
// CHECK:        tf_executor.fetch %[[ISLAND]]#0 : tensor<i32>


// Test if islands can be merged when non dependent islands are interleaved.
// CHECK-LABEL: func @islands_interleaved
// CHECK-SAME: (%[[ARG0:[a-z0-9]*]]: tensor<i32>, %[[ARG1:[a-z0-9]*]]: tensor<i32>)
func @islands_interleaved(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %7 = "tf.Identity"(%arg0) {T = "tfdtype$DT_INT32", device = "", name = "Identity_0"} : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %7 : tensor<i32>
    }
    %2:2 = tf_executor.island {
      %8 = "tf.Identity"(%arg1) {T = "tfdtype$DT_INT32", device = "", name = "Identity_1"} : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %8 : tensor<i32>
    }
    %3:2 = tf_executor.island {
      %9 = "tf.Identity"(%1#0) {T = "tfdtype$DT_INT32", device = "", name = "Identity_2"} : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %9 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %10 = "tf.Identity"(%2#0) {T = "tfdtype$DT_INT32", device = "", name = "Identity_3"} : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %10 : tensor<i32>
    }
    %5:2 = tf_executor.island(%3#1) {
      %11 = "tf.Identity"(%arg0) {T = "tfdtype$DT_INT32", device = "", name = "Identity_4"} : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %11 : tensor<i32>
    }
    %6:2 = tf_executor.island {
      %12 = "tf.Identity"(%arg1) {T = "tfdtype$DT_INT32", device = "", name = "Identity_5"} : (tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    tf_executor.fetch %4#0, %3#0 : tensor<i32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// CHECK:        %[[ISLAND0:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[IDENTITY0:[0-9]*]] = "tf.Identity"(%[[ARG0]])
// CHECK-NEXT:     %[[IDENTITY2:[0-9]*]] = "tf.Identity"(%[[IDENTITY0]])
// CHECK-NEXT:     %{{[0-9]*}} = "tf.Identity"(%[[ARG0]])
// CHECK-NEXT:     tf_executor.yield %[[IDENTITY2]] : tensor<i32>
// CHECK:        %[[ISLAND1:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[IDENTITY1:[0-9]*]] = "tf.Identity"(%[[ARG1]])
// CHECK-NEXT:     %[[IDENTITY3:[0-9]*]] = "tf.Identity"(%[[IDENTITY1]])
// CHECK-NEXT:     tf_executor.yield %[[IDENTITY3]] : tensor<i32>
// CHECK:        %{{[0-9]*}}:2 = tf_executor.island {
// CHECK-NEXT:     %[[IDENTITY5:[0-9]*]] = "tf.Identity"(%[[ARG1]])
// CHECK-NEXT:     tf_executor.yield %[[IDENTITY5]] : tensor<i32>
// CHECK:        tf_executor.fetch %[[ISLAND1]]#0, %[[ISLAND0]]#0 : tensor<i32>, tensor<i32>


// Test only islands are merged when other tf_executor ops are interleaved.
// CHECK-LABEL: func @merge_islands_only
func @merge_islands_only() {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      %cst = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %cst : tensor<i32>
    }
    %1:2 = tf_executor.Enter %0#0 frame "while/while_context" : (tensor<i32>) -> (tensor<*xi32>, !tf_executor.control) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Enter"}
    %2 = tf_executor.island {
      "tf.NoOp"() {device = "", name = "cluster/pivot"} : () -> ()
      tf_executor.yield
    }
    %3:3 = tf_executor.NextIteration.Source : tensor<*xi32> {T =  "tfdtype$DT_INT32", device =  "", id =  0 : i64, name =  "while/NextIteration"}
    %4:3 = tf_executor.Merge %3#0, %1#0 : tensor<*xi32> {N = 2 : i64, T =  "tfdtype$DT_INT32", device =  "", name =  "while/Merge"}
    %5:2 = tf_executor.island(%4#2) {
      %cst = "tf.Const"() {device =  "", dtype =  "tfdtype$DT_INT32", name =  "while/Less/y", value =  dense<2> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %cst : tensor<i32>
    }
    %6:2 = tf_executor.island {
      %14 = "tf.Less"(%4#0, %5#0) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Less"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
      tf_executor.yield %14 : tensor<*xi1>
    }
    %7:2 = tf_executor.LoopCond %6#0 : (tensor<*xi1>) -> (tensor<i1>, !tf_executor.control) {device =  "", name =  "while/LoopCond"}
    %8:3 = tf_executor.Switch %4#0, %7#0 : tensor<*xi32> {T =  "tfdtype$DT_INT32", _class =  ["loc = @while/Merge"], device =  "", name =  "while/Switch"}
    %9:2 = tf_executor.Exit %8#0 : tensor<*xi32> {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Exit"}
    %10:2 = tf_executor.island {
      %15 = "tf.Identity"(%8#1) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Identity"} : (tensor<*xi32>) -> tensor<*xi32>
      tf_executor.yield %15 : tensor<*xi32>
    }
    %11:2 = tf_executor.island(%10#1) {
      %cst = "tf.Const"() {device =  "", dtype =  "tfdtype$DT_INT32", name =  "while/Add/y", value = dense<3> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield %cst : tensor<i32>
    }
    %12:2 = tf_executor.island {
      %16 = "tf.Add"(%10#0, %11#0) {T =  "tfdtype$DT_INT32", device =  "", name =  "while/Add"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tf_executor.yield %16 : tensor<*xi32>
    }
    %13 = tf_executor.ControlTrigger %2, %12#1, %9#1 {_tpu_replicate = "cluster", device = "", name = "gradients/while/mul_2_Da30D05wlPU_grad/SymbolicGradient/b_sync"}
    tf_executor.NextIteration.Sink [%3#1] %12#0, %13 : tensor<*xi32> {T =  "tfdtype$DT_INT32", device =  "", id = 0 : i64, name =  "while/NextIteration"}
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[CONST:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[CONST0:.*]] = "tf.Const"
// CHECK-NEXT:     tf_executor.yield %[[CONST0]] : tensor<i32>
// CHECK:        %[[ENTER:[0-9]*]]:2 = tf_executor.Enter %[[CONST]]#0
// CHECK-NEXT:   %[[NOOP:[0-9]*]] = tf_executor.island {
// CHECK-NEXT:     "tf.NoOp"()
// CHECK-NEXT:     tf_executor.yield
// CHECK:        %[[NEXTIT_SRC:[0-9]*]]:3 = tf_executor.NextIteration.Source
// CHECK-NEXT:   %[[MERGE:[0-9]*]]:3 = tf_executor.Merge %[[NEXTIT_SRC]]#0, %[[ENTER]]#0
// CHECK-NEXT:   %[[LESS:[0-9]*]]:2 = tf_executor.island(%[[MERGE]]#2) {
// CHECK-NEXT:     %[[CONST_LESS:.*]] = "tf.Const"
// CHECK-NEXT:     %[[LESS0:[0-9]*]] = "tf.Less"(%[[MERGE]]#0, %[[CONST_LESS]])
// CHECK-NEXT:     tf_executor.yield %[[LESS0]] : tensor<*xi1>
// CHECK:        %[[COND:[0-9]*]]:2 = tf_executor.LoopCond %[[LESS:[0-9]*]]#0
// CHECK-NEXT:   %[[SWITCH:[0-9]*]]:3 = tf_executor.Switch %[[MERGE]]#0, %[[COND]]#0
// CHECK-NEXT:   %[[EXIT:[0-9]*]]:2 = tf_executor.Exit %[[SWITCH]]#0
// CHECK-NEXT:   %[[ADD:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[IDENTITY:[0-9]*]] = "tf.Identity"(%[[SWITCH]]#1)
// CHECK-NEXT:     %[[CONST_ADD:.*]] = "tf.Const"
// CHECK-NEXT:     %[[ADD0:[0-9]*]] = "tf.Add"(%[[IDENTITY]], %[[CONST_ADD]])
// CHECK-NEXT:     tf_executor.yield %[[ADD0]] : tensor<*xi32>
// CHECK:        %[[CT:[0-9]*]] = tf_executor.ControlTrigger %[[NOOP]], %[[ADD]]#1, %[[EXIT]]#1
// CHECK-NEXT:   tf_executor.NextIteration.Sink [%[[NEXTIT_SRC]]#1] %[[ADD]]#0, %[[CT]]
// CHECK-NEXT:   tf_executor.fetch


// Test no merging took place as cycle would be formed otherwise.
// CHECK-LABEL: func @simple_potential_cycle
func @simple_potential_cycle() {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      %3 = "tf.Placeholder"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "a", shape = "tfshape$dim {\0A  size: 1\0A}\0A"} : () -> tensor<1xf32>
      tf_executor.yield %3 : tensor<1xf32>
    }
    %1 = tf_executor.ControlTrigger %0#1 {_tpu_replicate = "cluster", device = "", name = "b"}
    %2:3 = tf_executor.island(%1) {
      %4 = "tf.Placeholder"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "c", shape = "tfshape$dim {\0A  size: 1\0A}\0A"} : () -> tensor<1xf32>
      tf_executor.yield %0#0, %4 : tensor<1xf32>, tensor<1xf32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK:        %[[ISLAND0:[0-9]*]]:2 = tf_executor.island {
// CHECK-NEXT:     %[[PLACEHOLDER0:[0-9]*]] = "tf.Placeholder"
// CHECK-NEXT:     tf_executor.yield %[[PLACEHOLDER0]] : tensor<1xf32>
// CHECK:        %[[CONTROL_TRIGGER:[0-9]*]] = tf_executor.ControlTrigger %[[ISLAND0]]#1
// CHECK-NEXT:   %{{[0-9]*}}:3 = tf_executor.island(%[[CONTROL_TRIGGER]]) {
// CHECK-NEXT:     %[[PLACEHOLDER1:[0-9]*]] = "tf.Placeholder"
// CHECK-NEXT:     tf_executor.yield %[[ISLAND0]]#0, %[[PLACEHOLDER1]] : tensor<1xf32>, tensor<1xf32>
// CHECK:        tf_executor.fetch
