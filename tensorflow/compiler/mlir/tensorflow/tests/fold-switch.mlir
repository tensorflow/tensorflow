// RUN: tf-opt -tf-switch-fold %s | FileCheck %s

// CHECK-LABEL: test_single_branch_direct_f
// CHECK-NOT: Switch
// CHECK-NOT: tf.AddV2
func @test_single_branch_direct_f() -> tensor<i32> {
  %cst = constant dense<false> : tensor<i1>
  %cst_0 = constant dense<10> : tensor<i32>
  %cst_1 = constant dense<1> : tensor<i32>
  %0 = tf_executor.graph {
    %7:3 = tf_executor.Switch %cst_0, %cst : tensor<i32>
    %8:2 = tf_executor.island {
      %12 = "tf.AddV2"(%7#1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    %11:3 = tf_executor.Merge %7#0, %8#0 : tensor<i32> {N = 2 : i64}
    tf_executor.fetch %11#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK-LABEL: test_single_branch_direct_t
// CHECK-NOT: Switch
// CHECK: tf.AddV2
func @test_single_branch_direct_t() -> tensor<i32> {
  %cst = constant dense<true> : tensor<i1>
  %cst_0 = constant dense<10> : tensor<i32>
  %cst_1 = constant dense<1> : tensor<i32>
  %0 = tf_executor.graph {
    %7:3 = tf_executor.Switch %cst_0, %cst : tensor<i32>
    %8:2 = tf_executor.island {
      %12 = "tf.AddV2"(%7#1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    %11:3 = tf_executor.Merge %7#0, %8#0 : tensor<i32> {N = 2 : i64}
    tf_executor.fetch %11#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// CHECK-LABEL: test_single_branch_direct_arg_f
// CHECK: Switch
// CHECK: tf.AddV2
func @test_single_branch_direct_arg_f(%pred : tensor<i1>) -> tensor<i32> {
  %cst_0 = constant dense<10> : tensor<i32>
  %cst_1 = constant dense<1> : tensor<i32>
  %0 = tf_executor.graph {
    %7:3 = tf_executor.Switch %cst_0, %pred : tensor<i32>
    %8:2 = tf_executor.island {
      %12 = "tf.AddV2"(%7#1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    %11:3 = tf_executor.Merge %7#0, %8#0 : tensor<i32> {N = 2 : i64}
    tf_executor.fetch %11#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// pred ? x + 1 : x - 1
// CHECK-LABEL: ControlFlowTest.testCond_1f
// CHECK-NOT: Switch
// CHECK-NOT: tf.AddV2
// CHECK: tf.Sub
func @ControlFlowTest.testCond_1f() -> tensor<i32> {
  %cst = constant dense<false> : tensor<i1>
  %cst_0 = constant dense<10> : tensor<i32>
  %cst_1 = constant dense<1> : tensor<i32>
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %cst, %cst : tensor<i1> {T = "tfdtype$DT_BOOL"}
    %2:2 = tf_executor.island {
      %12 = "tf.Identity"(%1#0) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %12 : tensor<i1>
    }
    %3:2 = tf_executor.island(%2#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %12 = "tf.Identity"(%1#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %12 : tensor<i1>
    }
    %5:2 = tf_executor.island(%4#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %6:2 = tf_executor.island {
      %12 = "tf.Identity"(%cst) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %12 : tensor<i1>
    }
    %7:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %8:2 = tf_executor.island {
      %12 = "tf.AddV2"(%7#1, %5#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    %9:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %10:2 = tf_executor.island {
      %12 = "tf.Sub"(%9#0, %3#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    %11:3 = tf_executor.Merge %10#0, %8#0 : tensor<i32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    tf_executor.fetch %11#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// pred ? x + 1 : x - 1
// CHECK-LABEL: ControlFlowTest.testCond_1t
// CHECK-NOT: Switch
// CHECK: tf.AddV2
// CHECK-NOT: tf.Sub
func @ControlFlowTest.testCond_1t() -> tensor<i32> {
  %cst = constant dense<true> : tensor<i1>
  %cst_0 = constant dense<10> : tensor<i32>
  %cst_1 = constant dense<1> : tensor<i32>
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %cst, %cst : tensor<i1> {T = "tfdtype$DT_BOOL"}
    %2:2 = tf_executor.island {
      %12 = "tf.Identity"(%1#0) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %12 : tensor<i1>
    }
    %3:2 = tf_executor.island(%2#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %12 = "tf.Identity"(%1#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %12 : tensor<i1>
    }
    %5:2 = tf_executor.island(%4#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %6:2 = tf_executor.island {
      %12 = "tf.Identity"(%cst) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %12 : tensor<i1>
    }
    %7:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %8:2 = tf_executor.island {
      %12 = "tf.AddV2"(%7#1, %5#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    %9:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %10:2 = tf_executor.island {
      %12 = "tf.Sub"(%9#0, %3#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %12 : tensor<i32>
    }
    %11:3 = tf_executor.Merge %10#0, %8#0 : tensor<i32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    tf_executor.fetch %11#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// if (pred)
//   return pred ? x + 1 : x - 1
// else
//   return x - 1
// CHECK-LABEL: ControlFlowTest.testCond_3f
// CHECK-NOT: Switch
// CHECK-NOT: tf.AddV2
// CHECK: tf.Sub
func @ControlFlowTest.testCond_3f() -> tensor<i32> {
  %cst = constant dense<false> : tensor<i1>
  %cst_0 = constant dense<10> : tensor<i32>
  %cst_1 = constant dense<1> : tensor<i32>
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %cst, %cst : tensor<i1> {T = "tfdtype$DT_BOOL"}
    %2:2 = tf_executor.island {
      %24 = "tf.Identity"(%1#0) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %3:2 = tf_executor.island(%2#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %24 = "tf.Identity"(%1#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %5:2 = tf_executor.island(%4#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %6:2 = tf_executor.island {
      %24 = "tf.Identity"(%cst) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %7:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %8:2 = tf_executor.island {
      %24 = "tf.Sub"(%7#0, %3#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %9:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %10:3 = tf_executor.Switch %cst, %6#0 : tensor<i1> {T = "tfdtype$DT_BOOL", _class = ["loc:@Less"]}
    %11:3 = tf_executor.Switch %10#1, %10#1 : tensor<i1> {T = "tfdtype$DT_BOOL"}
    %12:2 = tf_executor.island {
      %24 = "tf.Identity"(%11#0) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %13:2 = tf_executor.island(%12#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %14:2 = tf_executor.island {
      %24 = "tf.Identity"(%11#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %15:2 = tf_executor.island(%14#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %16:2 = tf_executor.island {
      %24 = "tf.Identity"(%10#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %17:3 = tf_executor.Switch %9#1, %16#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %18:2 = tf_executor.island {
      %24 = "tf.AddV2"(%17#1, %15#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %19:3 = tf_executor.Switch %9#1, %16#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %20:2 = tf_executor.island {
      %24 = "tf.Sub"(%19#0, %13#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %21:3 = tf_executor.Merge %20#0, %18#0 : tensor<i32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    %22:2 = tf_executor.island {
      %24 = "tf.AddV2"(%21#0, %5#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %23:3 = tf_executor.Merge %8#0, %22#0 : tensor<i32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    tf_executor.fetch %23#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// if (pred)
//   return pred ? x + 1 : x - 1
// else
//   return x - 1
// CHECK-LABEL: ControlFlowTest.testCond_3t
// CHECK-NOT: Switch
// CHECK: tf.AddV2
// CHECK-NOT: tf.Sub
// CHECK: tf.AddV2
func @ControlFlowTest.testCond_3t() -> tensor<i32> {
  %cst = constant dense<true> : tensor<i1>
  %cst_0 = constant dense<10> : tensor<i32>
  %cst_1 = constant dense<1> : tensor<i32>
  %0 = tf_executor.graph {
    %1:3 = tf_executor.Switch %cst, %cst : tensor<i1> {T = "tfdtype$DT_BOOL"}
    %2:2 = tf_executor.island {
      %24 = "tf.Identity"(%1#0) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %3:2 = tf_executor.island(%2#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %4:2 = tf_executor.island {
      %24 = "tf.Identity"(%1#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %5:2 = tf_executor.island(%4#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %6:2 = tf_executor.island {
      %24 = "tf.Identity"(%cst) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %7:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %8:2 = tf_executor.island {
      %24 = "tf.Sub"(%7#0, %3#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %9:3 = tf_executor.Switch %cst_0, %6#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %10:3 = tf_executor.Switch %cst, %6#0 : tensor<i1> {T = "tfdtype$DT_BOOL", _class = ["loc:@Less"]}
    %11:3 = tf_executor.Switch %10#1, %10#1 : tensor<i1> {T = "tfdtype$DT_BOOL"}
    %12:2 = tf_executor.island {
      %24 = "tf.Identity"(%11#0) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %13:2 = tf_executor.island(%12#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %14:2 = tf_executor.island {
      %24 = "tf.Identity"(%11#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %15:2 = tf_executor.island(%14#1) {
      tf_executor.yield %cst_1 : tensor<i32>
    }
    %16:2 = tf_executor.island {
      %24 = "tf.Identity"(%10#1) {T = "tfdtype$DT_BOOL"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %24 : tensor<i1>
    }
    %17:3 = tf_executor.Switch %9#1, %16#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %18:2 = tf_executor.island {
      %24 = "tf.AddV2"(%17#1, %15#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %19:3 = tf_executor.Switch %9#1, %16#0 : tensor<i32> {T = "tfdtype$DT_INT32", _class = ["loc:@Const"]}
    %20:2 = tf_executor.island {
      %24 = "tf.Sub"(%19#0, %13#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %21:3 = tf_executor.Merge %20#0, %18#0 : tensor<i32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    %22:2 = tf_executor.island {
      %24 = "tf.AddV2"(%21#0, %5#0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.yield %24 : tensor<i32>
    }
    %23:3 = tf_executor.Merge %8#0, %22#0 : tensor<i32> {N = 2 : i64, T = "tfdtype$DT_INT32"}
    tf_executor.fetch %23#0 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// TODO(jpienaar): This needs to be updated post changing send/recv to executor.
// CHECK-LABEL: switch_with_send_recv
// CHECK: Switch
func @switch_with_send_recv() {
  %cst = constant dense<true> : tensor<i1>
  tf_executor.graph {
    %1 = tf_executor.island {
      "tf._Send"(%cst#0) {T = "tfdtype$DT_BOOL", client_terminated = false, device = "/job:localhost/replica:0/task:0/device:CPU:0", name = "Const/_0", recv_device = "/job:localhost/replica:0/task:0/device:CPU:0", send_device = "/job:localhost/replica:0/task:0/device:CPU:0", send_device_incarnation = 1 : i64, tensor_name = "edge_3_Const"} : (tensor<i1>) -> ()
      tf_executor.yield
    }
    %2:2 = tf_executor.island(%1) {
      %11 = "tf._Recv"() {client_terminated = false, device = "/job:localhost/replica:0/task:0/device:CPU:0", name = "Const/_1", recv_device = "/job:localhost/replica:0/task:0/device:CPU:0", send_device = "/job:localhost/replica:0/task:0/device:CPU:0", send_device_incarnation = 1 : i64, tensor_name = "edge_3_Const", tensor_type = "tfdtype$DT_BOOL"} : () -> tensor<*xi1>
      tf_executor.yield %11 : tensor<*xi1>
    }
    %3:3 = tf_executor.Switch %2#0, %cst#0 : tensor<*xi1> {T = "tfdtype$DT_BOOL", device = "/job:localhost/replica:0/task:0/device:CPU:0", name = "cond/Switch"}
    %4:2 = tf_executor.island {
      %11 = "tf.Identity"(%3#0) {T = "tfdtype$DT_BOOL", _class = ["loc:@cond/control_dependency_1"], device = "/job:localhost/replica:0/task:0/device:CPU:0", name = "cond/switch_f"} : (tensor<*xi1>) -> tensor<*xi1>
      tf_executor.yield %11 : tensor<*xi1>
    }
    %5:2 = tf_executor.island(%4#1) {
      %11 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = "tfdtype$DT_BOOL", name = "cond/Assert/Assert/condition", value = dense<false> : tensor<i1>} : () -> tensor<i1>
      tf_executor.yield %11 : tensor<i1>
    }
    %6:2 = tf_executor.island(%4#1) {
      %11 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = "tfdtype$DT_STRING", name = "cond/Assert/Assert/data_0", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C30313757726F6E67206272616E636821212122"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      tf_executor.yield %11 : tensor<!tf.string>
    }
    %7 = tf_executor.island {
      "tf.Assert"(%5#0, %6#0) {T = ["tfdtype$DT_STRING"], device = "/job:localhost/replica:0/task:0/device:CPU:0", name = "cond/Assert/Assert", summarize = 3 : i64} : (tensor<i1>, tensor<!tf.string>) -> ()
      tf_executor.yield
    }
    %8:2 = tf_executor.island(%7) {
      %11 = "tf.Identity"(%4#0) {T = "tfdtype$DT_BOOL", device = "/job:localhost/replica:0/task:0/device:CPU:0", name = "cond/control_dependency_1"} : (tensor<*xi1>) -> tensor<*xi1>
      tf_executor.yield %11 : tensor<*xi1>
    }
    %9:3 = tf_executor.Merge %8#0, %cst#0 : (tensor<*xi1>, tensor<i1>) -> (tensor<*xi1>, tensor<i32>, !tf_executor.control) {N = 2 : i64, T = "tfdtype$DT_BOOL", device = "/job:localhost/replica:0/task:0/device:CPU:0", name = "cond/Merge"}
    %10 = tf_executor.island {
      "tf._Retval"(%9#0) {T = "tfdtype$DT_BOOL", device = "/job:localhost/replica:0/task:0/device:CPU:0", index = 0 : i64, name = "_retval_cond/Merge_0_0"} : (tensor<*xi1>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}
