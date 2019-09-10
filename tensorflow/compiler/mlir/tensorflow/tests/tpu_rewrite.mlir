// RUN: tf-opt %s -split-input-file -tf-tpu-rewrite | FileCheck %s

// Tests simple case of `tf_device.launch_func` on TPU with single input and
// single output.

module {
  // CHECK-LABEL: func @single_tpu_launch_func
  func @single_tpu_launch_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests that launch_func without _tpu_replicate attribute is ignored.

module {
  // CHECK-LABEL: func @single_gpu_launch_func
  func @single_gpu_launch_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    %1 = "tf_device.launch_func"(%0) {device = "gpu0", func = @gpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: tf_device.launch_func
    // CHECK-SAME: {device = "gpu0", func = @gpu0_func}

    return %1 : tensor<?xi32>
  }

  func @gpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests of `tf_device.launch_func` on TPU with nested function calls.

module {
  // CHECK-LABEL: func @with_nested_func
  func @with_nested_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func @nested_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = call @nested_func(%0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @nested_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests of `tf_device.launch_func` on TPU with referenced function that's not
// via a standard call op.

module {
  // CHECK-LABEL: func @with_referenced_func
  func @with_referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func @referenced_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func} : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests rewriting `tf_device.launch_func` on TPU with a chain of referenced
// functions.

module {
  // CHECK-LABEL: func @with_referenced_func_chain
  func @with_referenced_func_chain(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: @referenced_func1
    // CHECK-SAME: tf.D
    // CHECK-SAME: @referenced_func2
    // CHECK-SAME: tf.E
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func1} : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @referenced_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = call @referenced_func2(%0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func2(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.E"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests rewriting `tf_device.launch_func` on TPU with multiple calls to same
// function.

module {
  // CHECK-LABEL: func @with_multiple_call_same_referenced_func
  func @with_multiple_call_same_referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-COUNT-2: call @referenced_func
    // CHECK-COUNT-1: func @referenced_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func1} : (tensor<?xi32>) -> tensor<?xi32>
    %1 = call @referenced_func(%0) : (tensor<?xi32>) -> tensor<?xi32>
    %2 = call @referenced_func(%1) : (tensor<?xi32>) -> tensor<?xi32>
    return %2 : tensor<?xi32>
  }

  func @referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}

// -----

// Tests multiple `tf_device.launch_func` on TPU with different computation.

module {
  // CHECK-LABEL: func @multiple_launch_different_func
  func @multiple_launch_different_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func0} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE0_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func0
    // CHECK: %[[EXECUTE0_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE0_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf_device.launch_func"(%1) {_tpu_replicate = "cluster1", device = "tpu0", func = @tpu0_func1} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[EXECUTE0_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[EXECUTE0_OUTPUT]])
    // CHECK: %[[COMPILE1_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[EXECUTE0_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func1
    // CHECK: %[[EXECUTE1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[EXECUTE0_OUTPUT]], %[[COMPILE1_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE1_OUTPUT]])

    return %3 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func0(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @tpu0_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests multiple `tf_device.launch_func` on TPU with same computation.

module {
  // CHECK-LABEL: func @multiple_launch_same_func
  func @multiple_launch_same_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE0_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE0_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE0_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf_device.launch_func"(%1) {_tpu_replicate = "cluster1", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[EXECUTE0_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[EXECUTE0_OUTPUT]])
    // CHECK: %[[COMPILE1_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[EXECUTE0_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[EXECUTE0_OUTPUT]], %[[COMPILE1_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE1_OUTPUT]])

    return %3 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests Functions referenced by TPU function via SymbolRefAttr nested in
// ArrayAttr and DictionaryAttr.

module {
  // CHECK-LABEL: func @single_tpu_launch_func
  func @single_tpu_launch_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func @referenced_func2
    // CHECK-SAME: tf.H
    // CHECK-SAME: func @referenced_func3
    // CHECK-SAME: tf.I
    // CHECK-SAME: func @referenced_func0
    // CHECK-SAME: tf.F
    // CHECK-SAME: func @referenced_func1
    // CHECK-SAME: tf.G
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)
    // CHECK-SAME: Targs = [tensor<?xi32>]
    // CHECK-SAME: Tresults = [tensor<?xi32>]

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.D"(%0) {array_attr_funcs = [@referenced_func0, @referenced_func1]} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.E"(%1) {dictionary_attr_funcs = {fn1 = @referenced_func2, fn2 = @referenced_func3}} : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @referenced_func0(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.F"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.G"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func2(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.H"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func3(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.I"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}


// -----


// Tests that TPUCompilationResult operations are properly rewritten

// CHECK-LABEL: func @tpu_compilation_result
func @tpu_compilation_result(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<!tf.string>, tensor<!tf.string>) {

  // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"
  // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"
  %1 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "tpu0", func = @tpu0_func} : (tensor<?xi32>) -> tensor<?xi32>

  %compile_result = "tf.TPUCompilationResult"() {_tpu_replicate = "cluster0"} : () -> tensor<!tf.string>
  %compile_result2 = "tf.TPUCompilationResult"() {_tpu_replicate = "cluster0"} : () -> tensor<!tf.string>

  // CHECK: return %[[EXECUTE_OUTPUT]], %[[COMPILE_OUTPUT]]#0, %[[COMPILE_OUTPUT]]#0
  return %1, %compile_result, %compile_result2 : tensor<?xi32>, tensor<!tf.string>, tensor<!tf.string>
}

func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}


// -----

// Tests that TPUReplicatedInput and TPUReplicatedOutput operations are properly rewritten

func @main(%arg0 : tensor<0xf32>, %arg1 : tensor<0xf32>) -> tensor<0xf32> {
  // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%arg0, %arg1
  %0 = "tf.TPUReplicatedInput"(%arg0) {N = 1 : i64} : (tensor<0xf32>) -> tensor<0xf32>
  %1 = "tf.TPUReplicatedInput"(%arg1) {N = 1 : i64} : (tensor<0xf32>) -> tensor<0xf32>
  %2 = "tf_device.launch_func"(%0, %1) {device = "", _tpu_replicate = "cluster", func = @_func} : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
  %3 = "tf.TPUReplicatedOutput"(%2) {num_replicas = 1 : i64} : (tensor<0xf32>) -> tensor<0xf32>
  return %3 : tensor<0xf32>
}
func @_func(%arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  %0 = "tf.Const"() {value = dense<3.000000e+00> : tensor<0xf32>} : () -> tensor<0xf32>
  return %0 : tensor<0xf32>
}
