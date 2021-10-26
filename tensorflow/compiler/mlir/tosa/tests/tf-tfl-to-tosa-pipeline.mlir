// RUN: tf-opt --split-input-file --tf-tfl-to-tosa-pipeline  --verify-each %s | FileCheck %s

// These tests focus on TensorFlow and TensorFlow Lite hybrid lowering. These
// tests focus on tfl.custom operations that are Flex ops.

// CHECK-LABEL: func @test_flex
func @test_flex(%arg0: tensor<1x1x64xf32>) -> tensor<*xf32> {
  %0 = "tfl.custom"(%arg0) {custom_code = "FlexIdentity", custom_option = opaque<"tfl", "0x084964656E74697479005212084964656E746974791A002A070A015412023001323B0A3966697273745F636F6E766F6C7574696F6E2F5446436F6E7631643178312F457870616E6444696D735F312F526561645661726961626C654F7000025E551414042801"> : tensor<102xi8>} : (tensor<1x1x64xf32>) -> tensor<*xf32>
  // CHECK: %[[CAST:.+]] = tensor.cast %arg0
  // CHECK: return %[[CAST]]
  return %0 : tensor<*xf32>
}

// ----

// CHECK-LABEL: func @test_flex_shape
func @test_flex_shape(%arg0: tensor<?x2x64xf32>, %arg1: tensor<1x1x64xf32>) -> tensor<*xf32> {
  // CHECK: %[[ADD:.+]] = "tosa.add"(%arg0, %arg1) : (tensor<?x2x64xf32>, tensor<1x1x64xf32>) -> tensor<?x2x64xf32>
  // CHECK: %[[CAST:.+]] = tensor.cast %[[ADD]]
  // CHECK: return %[[CAST]]
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexAddV2", custom_option = opaque<"tfl", "0x054164645632002E120541646456321A001A002A070A01541202300132180A16726573696475616C5F626C6F636B5F2E5F302F616464000237311414042801"> : tensor<63xi8>} : (tensor<?x2x64xf32>, tensor<1x1x64xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32>
}