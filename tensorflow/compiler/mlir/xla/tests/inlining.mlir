// RUN: xla-opt %s -inline | FileCheck %s --dump-input=fail

// Test case: Basic test of inlining into xla_hlo.while.

// CHECK-LABEL: func @caller
// CHECK:   "xla_hlo.while"{{.*}}( {
// CHECK:   },  {
// CHECK:     "xla_hlo.exponential"
// CHECK:   })
// CHECK-LABEL: func @callee

func @caller(%arg0: tensor<f32>, %pred: tensor<i1>) -> tensor<f32> {
  %0 = "xla_hlo.while"(%arg0) ( {
  ^entry(%unused: tensor<f32>):
    "xla_hlo.return"(%pred) : (tensor<i1>) -> ()
  }, {
  ^entry(%0: tensor<f32>):
    %1 = call @callee(%0) : (tensor<f32>) -> (tensor<f32>)
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  } ) : (tensor<f32>) -> (tensor<f32>)
  return %0 : tensor<f32>
}


func @callee(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "xla_hlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
