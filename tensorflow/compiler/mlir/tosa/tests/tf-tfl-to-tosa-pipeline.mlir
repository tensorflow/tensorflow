// RUN: tf-tosa-opt --split-input-file --tf-tfl-to-tosa-pipeline  --verify-each %s | FileCheck %s


// These tests focus on TensorFlow and TensorFlow Lite hybrid lowering and focus
// on tfl.custom operations that are Flex ops.

// This test demonstrates how legalize and shape inference can be performed
// using the flex op legalization, and tensorflow-to-tosa legalization. The
// resulting cast is to maintain the function contract.
//
// To add a Flex op test you will need to compile a TF model with a flex op,
// then import the model. The opaque data is a serialize TF Node proto so not
// feasible to hand edit.
// CHECK-LABEL: func @test_flex_shape
// CHECK-SAME: tensor<?x2x64xf32>
func.func @test_flex_shape(%arg0: tensor<?x2x64xf32>, %arg1: tensor<1x1x64xf32>) -> tensor<*xf32> {
  // CHECK: %[[ADD:.+]] = tosa.add %arg0, %arg1 : (tensor<?x2x64xf32>, tensor<1x1x64xf32>) -> tensor<?x2x64xf32>
  // CHECK: return %[[ADD]]
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexAddV2", custom_option = #tfl<const_bytes : "0x054164645632002E120541646456321A001A002A070A01541202300132180A16726573696475616C5F626C6F636B5F2E5F302F616464000237311414042801">} : (tensor<?x2x64xf32>, tensor<1x1x64xf32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>
}
