// Test to verify translation & export work as intended with runtime.

// RUN: tf-opt --mlir-print-debuginfo --canonicalize --tfl-while-loop-outline %s | mlir-tflite-runner --dump-interpreter-state 2>&1 | FileCheck %s

// Verify value computed:
// ----------------------
// CHECK: result: Tensor<type: FLOAT32, shape: 1, values: 96>
// CHECK: pconst: Tensor<type: INT32, shape: , values: 1>

// Verify tensors in interpreter state:
// ------------------------------------
// CHECK: Tensor 0 val kTfLiteFloat32 kTfLiteMmapRo 4 / 0.00 [1] [{{.*}})
// CHECK-NEXT: Tensor 1 N kTfLiteInt32 kTfLiteMmapRo 4 / 0.00 (null) [{{.*}})
// CHECK-NEXT: Tensor 2 pconst kTfLiteInt32 kTfLiteMmapRo 4 / 0.00 (null) [{{.*}})
// CHECK-NEXT: Tensor 3 tfl.while kTfLiteInt32 kTfLiteArenaRw 4 / 0.00 (null) [{{.*}})
// CHECK-NEXT: Tensor 4 result kTfLiteFloat32 kTfLiteArenaRw 4 / 0.00 [1] [{{.*}})

// Verify while was not folded away:
// ------------------------------------
// CHECK: Operator Builtin Code {{[0-9]*}} WHILE

func @main() -> (tensor<1xf32>, tensor<i32>)
    attributes {tf.entry_function = {outputs = "result,pconst"}} {
  %cst = arith.constant dense<1> : tensor<i32> loc("dec")
  %arg0 = arith.constant dense<5> : tensor<i32> loc("N")
  %arg1 = arith.constant dense<3.0> : tensor<1xf32> loc("val")
  %0:3 = "tfl.while"(%arg0, %arg1, %cst) ( {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>, %arg4: tensor<i32>):
      %cst_0 = arith.constant dense<0> : tensor<i32>
      %1 = "tfl.greater"(%arg2, %cst_0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>, %arg4: tensor<i32>):
      %1 = "tfl.sub"(%arg2, %arg4) {fused_activation_function = "NONE"} :
        (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = tfl.add %arg3, %arg3 {fused_activation_function = "NONE"} : tensor<*xf32>
      "tfl.yield"(%1, %2, %arg4) : (tensor<*xi32>, tensor<*xf32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>, tensor<i32>) -> (tensor<i32>, tensor<1xf32>, tensor<i32>)
  return %0#1, %0#2 : tensor<1xf32>, tensor<i32>
}

