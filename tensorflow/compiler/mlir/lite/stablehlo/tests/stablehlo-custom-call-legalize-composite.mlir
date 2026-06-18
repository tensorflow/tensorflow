// RUN: odml-to-stablehlo-opt %s -stablehlo-custom-call-legalize-composite | FileCheck %s

// CHECK-LABEL: module
module {
  // CHECK-LABEL: @main
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) {
    // CHECK: stablehlo.custom_call @foo
    stablehlo.custom_call @foo() : () -> ()
    // CHECK-NOT: stablehlo.custom_call
    // CHECK: stablehlo.composite "odml.foo" %arg0, %arg1 {composite_attributes = {bar = 500 : i64}, decomposition = @foo.impl} : (tensor<1xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<1xf32>)
    %1:2 = stablehlo.custom_call @stablehlo.composite(%arg0, %arg1) {called_computations = [@foo.impl], composite.backend_config = {attributes = {bar = 500 : i64}, name = "odml.foo"}} : (tensor<1xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<1xf32>)
    return
  }
  // CHECK-LABEL: func private @foo.impl
  func.func private @foo.impl(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<1xf32>) {
    return %arg1, %arg0 : tensor<2xf32>, tensor<1xf32>
  }
}
