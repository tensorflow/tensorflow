// RUN: odml_to_stablehlo %s -skip-resize -smuggle-disallowed-ops -o - | FileCheck %s

// CHECK-LABEL: @main
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 975 : i32}, tf_saved_model.semantics}  {
  func.func @serving_default(%arg0: tensor<1x32x32x128xf32> {tf_saved_model.index_path = ["a"]}) -> (tensor<1x64x64x128xf32> {tf_saved_model.index_path = ["b"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "c:0", outputs = "d:0"}, tf_saved_model.exported_names = ["serving_default"]} {
      %0  = "tf.Const"() {value = dense<[56, 904]> : tensor<2xi32>} : () -> tensor<2xi32>
      // CHECK: %1 = stablehlo.custom_call @tf.ResizeBilinear(%arg0, %0) {align_corners = false, device = "", half_pixel_centers = true} : (tensor<1x32x32x128xf32>, tensor<2xi32>) -> tensor<1x64x64x128xf32>
      %1 = "tf.ResizeBilinear"(%arg0, %0) {
        align_corners = false, device = "", half_pixel_centers = true
      } : (tensor<1x32x32x128xf32>, tensor<2xi32>) -> tensor<1x64x64x128xf32>
      func.return %1 : tensor<1x64x64x128xf32>
  }
}
