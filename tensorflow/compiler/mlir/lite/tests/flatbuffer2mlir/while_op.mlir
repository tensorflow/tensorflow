// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// Check to see if nested regions in while loops are preserved
// CHECK:     %{{.*}}:2 = "tfl.while"(%{{.*}}, %{{.*}}) ( {
// CHECK:     ^bb0(%{{.*}}: tensor<*xi32>, %{{.*}}: tensor<*xf32>):
// CHECK:       "tfl.yield"(%{{.*}}) : (tensor<*xi1>) -> ()
// CHECK:     },  {
// CHECK:     ^bb0(%{{.*}}: tensor<*xi32>, %{{.*}}: tensor<*xf32>):
// CHECK:       "tfl.yield"(%{{.*}}, %{{.*}}) : (tensor<*xi32>, tensor<*xf32>) -> ()
// CHECK:     }) : (tensor<i32>, tensor<1xf32>) -> (tensor<*xi32>, tensor<1xf32>)

func @main(%arg0: tensor<i32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // While %arg0 is greater than zero, element wise add %arg1 with itself.
  %0:2 = "tfl.while"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):  // no predecessors
    %1 = call @cond(%arg2, %arg3) : (tensor<*xi32>, tensor<*xf32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):  // no predecessors
    %1:2 = call @body(%arg2, %arg3) : (tensor<*xi32>, tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>)
    "tfl.yield"(%1#0, %1#1) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>)
  return %0#1 : tensor<1xf32>
}

func @cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<i1> {
  %cst = constant dense<0> : tensor<i32> loc("Const")
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>
}

func @body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %cst = constant dense<1> : tensor<i32> loc("Const")
  %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %0, %1 : tensor<*xi32>, tensor<*xf32>
}
