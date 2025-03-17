// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Ensure constants roundtrip exactly

func.func @f32() -> tensor<4xf32> {
  // CHECK-LABEL: @f32
  // CHECK: value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %0 = "tfl.pseudo_const"() { value = dense_resource<dense_elements_f32> : tensor<4xf32> } : () -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

func.func @i8() -> tensor<4xi8> {
  // CHECK-LABEL: @i8
  // CHECK: value = dense<[1, 2, 3, 4]> : tensor<4xi8>
  %0 = "tfl.pseudo_const" () { value = dense_resource<dense_elements_i8> : tensor<4xi8> } : () -> tensor<4xi8>
  func.return %0 : tensor<4xi8>
}

func.func @i16() -> tensor<4xi16> {
  // CHECK-LABEL: @i16
  // CHECK: value = dense<[1, 2, 3, 258]> : tensor<4xi16>
  %0 = "tfl.pseudo_const" () { value = dense_resource<dense_elements_i16> : tensor<4xi16> } : () -> tensor<4xi16>
  func.return %0 : tensor<4xi16>
}

func.func @i32() -> tensor<4xi32> {
  // CHECK-LABEL: @i32
  // CHECK: value = dense<[1, 2, 3, 16909060]> : tensor<4xi32>
  // Check bytes come back in the right order
  %0 = "tfl.pseudo_const" () { value = dense_resource<dense_elements_i32> : tensor<4xi32> } : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

func.func @uint8() -> tensor<4xui8> {
  // CHECK-LABEL: @uint8
  // CHECK: value = dense<[222, 173, 190, 239]> : tensor<4xui8>
  %0 = "tfl.pseudo_const"() {value = dense_resource<dense_elements_i8_1> : tensor<4xui8>} : () -> tensor<4xui8>
  func.return %0 : tensor<4xui8>
}

// Identity function to make the exporter happy
func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  func.return %arg0 : tensor<4xi8>
}

{-#
  dialect_resources: {
    builtin: {
      dense_elements_f32: "0x400000000000803F000000400000404000008040",
      dense_elements_i16: "0x400000000100020003000201",
      dense_elements_i32: "0x4000000001000000020000000300000004030201",
      dense_elements_i8: "0x4000000001020304",
      dense_elements_i8_1: "0x40000000DEADBEEF"
    }
  }
#-}
