// RUN: tf_tfl_translate --enable-hlo-to-tf-conversion --input-mlir %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s --check-prefix=CHECK-ROUNDTRIP


module {
  func.func public @main( %arg0: tensor<i64>) ->  tensor<i64> {
    %0 = func.call @test_add_roundtrip(%arg0) : (tensor<i64>) -> tensor<i64>

    return %0 : tensor<i64>
  }


  // CHECK-LABEL: func.func private @test_add_roundtrip
  func.func private @test_add_roundtrip(%arg0: tensor<i64>) -> tensor<i64> {
    // CHECK-ROUNDTRIP:  %0 = stablehlo.composite "stablehlo.add_n" %arg0 {composite_attributes = {test_bool = false, test_int = 2 : i64, test_string = "test"}, decomposition = @add_n.impl} : (tensor<i64>) -> tensor<i64>
    %0 = stablehlo.composite "stablehlo.add_n" %arg0 { composite_attributes = { test_int = 2 : i64, test_bool = 0 : i1, test_string = "test"}, decomposition = @add_n.impl } : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @add_n.impl(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.constant dense<2> : tensor<i64>
    %1 = stablehlo.add %arg0, %0 : tensor<i64>
    return %1 : tensor<i64>
  }




}