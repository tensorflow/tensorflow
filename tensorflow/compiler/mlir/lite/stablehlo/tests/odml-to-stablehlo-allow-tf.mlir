// RUN: odml_to_stablehlo %s --allow-tf=false -o /tmp/temp.mlir; [ -f /tmp/temp.mlir ]; [ -f /tmp/debug_stablehlo.mlir ]
// RUN: odml_to_stablehlo %s --allow-tf=true -o /tmp/temp2.mlir; [ -f /tmp/temp2.mlir ]

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 975 : i32}, tf_saved_model.semantics}  {
  func.func @serving_default(%arg0: tensor<1x20x20x28xf32> {tf_saved_model.index_path = ["a"]}) -> (tensor<1x40x40x28xf32> {tf_saved_model.index_path = ["b"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "c:0", outputs = "d:0"}, tf_saved_model.exported_names = ["serving_default"]} {
      %0 = stablehlo.constant dense<40> : tensor<2xi32>
      %1 = "tf.UnconvertedOp"(%arg0, %0) {align_corners = false, half_pixel_centers = false} : (tensor<1x20x20x28xf32>, tensor<2xi32>) -> tensor<1x40x40x28xf32>
      func.return %1 : tensor<1x40x40x28xf32>
  }
}
