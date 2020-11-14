// RUN: mlir-hlo-opt %s -inline | FileCheck %s

// Test case: Basic test of inlining into mhlo.while.

// CHECK-LABEL: func @caller
// CHECK:   "mhlo.while"{{.*}}( {
// CHECK:   },  {
// CHECK:     "mhlo.exponential"
// CHECK:   })
// CHECK-LABEL: func @callee

func @caller(%arg0: tensor<f32>, %pred: tensor<i1>) -> tensor<f32> {
  %0 = "mhlo.while"(%arg0) ( {
  ^entry(%unused: tensor<f32>):
    "mhlo.return"(%pred) : (tensor<i1>) -> ()
  }, {
  ^entry(%0: tensor<f32>):
    %1 = call @callee(%0) : (tensor<f32>) -> (tensor<f32>)
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  } ) : (tensor<f32>) -> (tensor<f32>)
  return %0 : tensor<f32>
}


func @callee(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
