// RUN: litert-opt %s -split-input-file -tfl-analyze-variables-pass --cse | FileCheck %s

// CHECK: module attributes {tfl._legalize_tfl_variables = true}
module {
  func.func @f() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
    func.return %2 : tensor<*xi32>
  }
}

// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = true}
module {
  func.func @main() -> tensor<*xi32> {
    %0 = "tf.PartitionedCall"() {f = @f, config = "", config_proto = "", executor_type = ""}
      : () -> tensor<*xi32>
    func.return %0 : tensor<*xi32>
  }
  func.func @f() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>
    %1 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
    func.return %1 : tensor<*xi32>
  }
}


// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = false}
module {
  func.func @main() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>
    %1 = "tf.PartitionedCall"(%0) {f = @f, config = "", config_proto = "", executor_type = ""}
      : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
    func.return %1 : tensor<*xi32>
  }
  func.func @f(%arg0 : tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32> {
    %0 = "tf.ReadVariableOp"(%arg0) {dtype = i32} : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
    func.return %0 : tensor<*xi32>
  }
}

// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = false}
module {
  func.func @main() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>
    %cst = arith.constant dense<2> : tensor<4xi32>
    "tf.AssignAddVariableOp"(%0, %cst) {} : (tensor<*x!tf_type.resource<tensor<*xi32>>>, tensor<4xi32>) -> ()
    %1 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
    func.return %1 : tensor<*xi32>
  }
}

// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = true}
module {
  func.func @main() -> tensor<i32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>
    %cst = arith.constant dense<1> : tensor<i32>
    %1:2 = "tfl.while"(%cst, %0) ({
    ^bb0(%arg1: tensor<*xi32>, %arg2: tensor<*x!tf_type.resource<tensor<*xi32>>>):
      %2 = "tf.ReadVariableOp"(%arg2) {dtype = i32} : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
      %3 = "tfl.greater"(%arg1, %2) : (tensor<*xi32>, tensor<*xi32>) -> tensor<i1>
      "tfl.yield"(%3) : (tensor<i1>) -> ()
  },  {
    ^bb0(%arg3: tensor<*xi32>, %arg4: tensor<i32>):
      %4 = "tfl.sub"(%arg3, %arg4) {fused_activation_function = "NONE"} :
        (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      "tfl.yield"(%4) : (tensor<*xi32>) -> ()
  }) : (tensor<i32>, tensor<*x!tf_type.resource<tensor<*xi32>>>) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<*xi32>>>)
    func.return %1#0 : tensor<i32>
  }
}

// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = false}
module {
  func.func @main() -> tensor<i32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf_type.resource<tensor<*xi32>>>
    %cst = arith.constant dense<1> : tensor<i32>
    %1:2 = "tfl.while"(%cst, %0) ({
    ^bb0(%arg1: tensor<*xi32>, %arg2: tensor<*x!tf_type.resource<tensor<*xi32>>>):
      %2 = "tf.ReadVariableOp"(%arg2) {dtype = i32} : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
      %3 = "tfl.greater"(%arg1, %2) : (tensor<*xi32>, tensor<*xi32>) -> tensor<i1>
      "tfl.yield"(%3) : (tensor<i1>) -> ()
  },  {
    ^bb0(%arg3: tensor<*xi32>, %arg4: tensor<*x!tf_type.resource<tensor<*xi32>>>):
      %cst1 = arith.constant dense<2> : tensor<4xi32>
      "tf.AssignAddVariableOp"(%arg4, %cst1) {} : (tensor<*x!tf_type.resource<tensor<*xi32>>>, tensor<4xi32>) -> ()
      %4 = "tf.ReadVariableOp"(%arg4) {dtype = i32} : (tensor<*x!tf_type.resource<tensor<*xi32>>>) -> tensor<*xi32>
      "tfl.yield"(%4) : (tensor<*xi32>) -> ()
  }) : (tensor<i32>, tensor<*x!tf_type.resource<tensor<*xi32>>>) -> (tensor<i32>, tensor<*x!tf_type.resource<tensor<*xi32>>>)
    func.return %1#0 : tensor<i32>
  }
}

// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = true}
module {
  func.func @main(%arg0 : tensor<!tf_type.resource<tensor<4096xf32>>>,
      %arg1 : tensor<*x!tf_type.variant>) {
    %cst_0 = arith.constant dense<2> : tensor<i64>
    %cst_1 = arith.constant dense<0> : tensor<i32>
    %0 = "tf.RepeatDataset"(%arg1, %cst_0) {device = "",
      output_shapes = [#tf_type.shape<?>],
      output_types = [!tf_type.string]} : (tensor<*x!tf_type.variant>, tensor<i64>) -> tensor<!tf_type.variant>

    %1 = "tf.ReduceDataset"(%0, %cst_1, %arg0) {
      Targuments = [!tf_type.resource],
      Tstate = [i32], device = "",
      f = @__reduce_func, f._tf_data_function = true,
      output_shapes = [#tf_type.shape<>],
      output_types = [i32], use_inter_op_parallelism = true} : (tensor<!tf_type.variant>, tensor<i32>, tensor<!tf_type.resource<tensor<4096xf32>>>) -> (tensor<*xi32>)
    func.return
  }

  func.func private @__reduce_func(%arg0: tensor<i32> {tf._user_specified_name = "args_0"}) -> (tensor<i32>) attributes {tf._tf_data_function = true, tf.signature.is_stateful} {
    %0 = "tf.JustPretend"() : () -> (tensor<i32>)
    func.return %0: tensor<i32>
  }
}
