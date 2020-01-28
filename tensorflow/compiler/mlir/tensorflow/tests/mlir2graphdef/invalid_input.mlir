// RUN: not tf-mlir-translate -split-input-file -mlir-to-graphdef %s -o - 2>&1 | FileCheck %s --dump-input=fail

// Tests invalid tf_executor.graph args.

func @main(%arg0: tensor<i32>) {
  tf_executor.graph {
    %0:3 = tf_executor.Merge %arg0, %arg0 : tensor<i32> {device = "", N = 2, T = "tfdtype$DT_INT32"} loc("while/Merge")
    tf_executor.fetch
  }
  return
}

// CHECK: Arg in 'main' should only have one user.

// -----

func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  tf_executor.graph {
    %0:3 = tf_executor.Merge %arg0, %arg1 : tensor<i32> {device = "", N = 2, T = "tfdtype$DT_INT32"} loc("while/Merge")
    tf_executor.fetch
  }
  return
}

// CHECK: User of arg in 'main' must be in an inner op of a tf_executor.island.

// -----

func @main(%arg0: tensor<i32>) {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Identity"(%arg0) {T = "tfdtype$DT_INT32"} : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch %0#1 : !tf_executor.control
  }
  return
}

// CHECK: tf_executor.island of user of arg in 'main' must have no control output users.

// -----

// Tests function with multiple blocks.

func @main() {
  ^bb:
    br ^bb1
  ^bb1:
    return
}

// CHECK: Functions must be of a single Graph with single op Islands: only single block functions are supported.

// -----

// Tests invalid functions for exporting to Graph/GraphDef.

func @main() {
  return
}

// CHECK: Functions must be of a single Graph with single op Islands: first op in function is not a tf_executor.graph.

// -----

func @main() {
  tf_executor.graph {
    tf_executor.fetch
  }
  tf_executor.graph {
    tf_executor.fetch
  }
  return
}

// CHECK: Functions must be of a single Graph with single op Islands: function does not only contain a single tf_executor.graph.

// -----

func @main() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: Functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op.

// -----

func @main() {
  tf_executor.graph {
    %0 = tf_executor.island {
      %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: Functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op.

// -----

func @main() {
  tf_executor.graph {
    %0 = tf_executor.island {
      %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: Functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op.

// -----

func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
      tf_executor.yield %1#1, %1#0 : tensor<i32>, tensor<i32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK: Functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op.
