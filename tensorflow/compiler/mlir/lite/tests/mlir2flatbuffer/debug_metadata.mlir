// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --serialize-debug-metadata=true %s -o - | flatbuffer_to_string - | FileCheck %s

module @jit_relu attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, tfl._legalize_tfl_variables = true} {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1> loc(#loc)
  %1 = "tf.If"(%0, %arg0, %arg1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc(#loc)
  func.return %1 : tensor<1xf32> loc(#loc)
  }

  func.func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#loc4)
    func.return %0 : tensor<*xf32> loc(#loc)
  }

  func.func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#loc5)
    func.return %0 : tensor<*xf32> loc(#loc)
  }
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("<ipython-input-7-340b9abeb7a8>":1:4)
#loc2 = loc("third_party/py/IPython/v3_2_3/core/interactiveshell.py":3066:16)
#loc3 = loc(callsite(#loc1 at #loc2))
#loc4 = loc("jit(relu)/jit(main)/max"(#loc3))
#loc5 = loc(fused<"">[#loc1, #loc2])

// CHECK: operators: [ {
// CHECK: name: "main",
// CHECK: debug_metadata_index:
// CHECK: name: "cond_true",
// CHECK: debug_metadata_index:
// CHECK: name: "cond_false",
// CHECK: debug_metadata_index:
// CHECK: metadata: [ {
// CHECK: name: "debug_metadata",