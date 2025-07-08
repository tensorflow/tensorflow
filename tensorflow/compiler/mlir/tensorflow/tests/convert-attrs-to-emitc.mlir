// RUN: tf-opt %s --tf-attrs-to-emitc-pipeline | FileCheck %s

module  {
  func.func @main(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["another_feature"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["some_feature"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes { } {
    %0 = tosa.add %arg1, %arg0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}


// CHECK: module {
// CHECK-NEXT:  func.func @main(
// CHECK-SAME:  %arg0: tensor<1xf32> {emitc.field_ref = "another_feature"}, 
// CHECK-SAME:  %arg1: tensor<1xf32> {emitc.field_ref = "some_feature"}) 
// CHECK-SAME:  -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) {
// CHECK-NEXT:     %0 = tosa.add %arg1, %arg0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK-NEXT:     return %0 : tensor<1xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

