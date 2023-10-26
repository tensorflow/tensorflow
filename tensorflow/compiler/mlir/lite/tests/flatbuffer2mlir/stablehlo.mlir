// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// test stablehlo roundtrip

// Identity function to make the exporter happy
func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  func.return %arg0 : tensor<4xi8>
}

//CHECK:func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> attributes {tf.entry_function = {inputs = "arg0", outputs = "arg0"}} {
//CHECK: return %arg0 : tensor<4xi8>
//CHECK:}

func.func @logistic(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

// CHECK:func.func private @logistic(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
// CHECK: %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
// CHECK: return %0 : tensor<1x1x1x96xf32>
// CHECK:}

func.func @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.add %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @multiply(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @multiply(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.multiply %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @divide(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @divide(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.divide %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @maximum(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @maximum(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.maximum %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @reshape(%arg0 : tensor<1x128xi32>) -> tensor<4x32x1xi32>{
  %0 = stablehlo.reshape %arg0 : (tensor<1x128xi32>) -> tensor<4x32x1xi32>
  func.return %0 : tensor<4x32x1xi32>
}

//CHECK:func.func private @reshape(%arg0: tensor<1x128xi32>) -> tensor<4x32x1xi32> {
//CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<1x128xi32>) -> tensor<4x32x1xi32>
//CHECK-NEXT: return %0 : tensor<4x32x1xi32>
//CHECK-NEXT:}

func.func @clamp(%arg0: tensor<f32>, %arg1: tensor<1x256x256x24xf32>, %arg2: tensor<f32>) -> tensor<1x256x256x24xf32>{
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<1x256x256x24xf32>, tensor<f32>) -> tensor<1x256x256x24xf32>
  return %0 : tensor<1x256x256x24xf32>
}

//CHECK:func.func private @clamp(%arg0: tensor<f32>, %arg1: tensor<1x256x256x24xf32>, %arg2: tensor<f32>) -> tensor<1x256x256x24xf32> {
//CHECK-NEXT: %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<1x256x256x24xf32>, tensor<f32>) -> tensor<1x256x256x24xf32>
//CHECK-NEXT: return %0 : tensor<1x256x256x24xf32>
//CHECK-NEXT:}

func.func @concat(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x2xi32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x2xi32>
  func.return %0 : tensor<1x30x2xi32>
}

//CHECK:func.func private @concat(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x2xi32> {
//CHECK-NEXT: %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x2xi32>
//CHECK-NEXT: return %0 : tensor<1x30x2xi32>
//CHECK-NEXT:}

func.func @broadcast_in_dim(%arg0: tensor<1x32x256xf32>) -> tensor<4x32x256xf32>{
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x32x256xf32>) -> tensor<4x32x256xf32>
  return %0 : tensor<4x32x256xf32>
}

//CHECK:func.func private @broadcast_in_dim(%arg0: tensor<1x32x256xf32>) -> tensor<4x32x256xf32> {
//CHECK-NEXT: %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x32x256xf32>) -> tensor<4x32x256xf32>
//CHECK-NEXT: return %0 : tensor<4x32x256xf32>
//CHECK-NEXT:}

func.func @slice(%arg0: tensor<160x20x1xf32>) -> tensor<1x1x1xf32> {
  %0 = stablehlo.slice %arg0 [0:1:1, 0:1:1, 0:1:1] : (tensor<160x20x1xf32>) -> tensor<1x1x1xf32>
  return %0 : tensor<1x1x1xf32>
}

//CHECK:func.func private @slice(%arg0: tensor<160x20x1xf32>) -> tensor<1x1x1xf32> {
//CHECK-NEXT: %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:1] : (tensor<160x20x1xf32>) -> tensor<1x1x1xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1xf32>
//CHECK-NEXT:}

func.func @convolution(%arg0: tensor<1x1x1600x32xf32>, %arg1: tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [6, 6]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 32 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x1x1600x32xf32>, tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32>
  return %0 : tensor<1x1x1600x32xf32>
}

//CHECK:func.func private @convolution(%arg0: tensor<1x1x1600x32xf32>, %arg1: tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32> {
//CHECK-NEXT{LITERAL}: %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [6, 6]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 32 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x1x1600x32xf32>, tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1600x32xf32>
//CHECK-NEXT:}

func.func @reduce(%arg0: tensor<1x16x16x320xf32>, %arg3 : tensor<f32>) -> tensor<1x320xf32> {
  %0 = stablehlo.reduce(%arg0 init: %arg3) across dimensions = [1, 2] : (tensor<1x16x16x320xf32>, tensor<f32>) -> tensor<1x320xf32>
   reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
    %421 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %421 : tensor<f32>
   }
  return %0 : tensor<1x320xf32>
}

//CHECK:func.func private @reduce(%arg0: tensor<1x16x16x320xf32>, %arg1: tensor<f32>) -> tensor<1x320xf32> {
//CHECK-NEXT: %0 = stablehlo.reduce(%arg0 init: %arg1) across dimensions = [1, 2] : (tensor<1x16x16x320xf32>, tensor<f32>) -> tensor<1x320xf32>
//CHECK-NEXT: reducer(%arg2: tensor<f32>, %arg3: tensor<f32>) {
//CHECK-NEXT:  %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
//CHECK-NEXT:  return %1 : tensor<f32>
//CHECK-NEXT: }
//CHECK-NEXT: return %0 : tensor<1x320xf32>
//CHECK-NEXT:}

func.func @abs(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.abs %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @abs(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.abs %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @and(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = stablehlo.and %arg0, %arg1 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @and(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = stablehlo.and %arg0, %arg1 : tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @cos(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.cosine %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @cos(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.cosine %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @exp(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.exponential %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @exp(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.exponential %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @floor(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
 %0 = stablehlo.floor %arg0 : tensor<1x1x1x96xf32>
 func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @floor(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.floor %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @log(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.log %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @log(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.log %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @min(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = stablehlo.minimum %arg0, %arg1 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @min(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = stablehlo.minimum %arg0, %arg1 : tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @neg(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.negate %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @neg(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.negate %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @or(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = stablehlo.or %arg0, %arg1 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @or(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = stablehlo.or %arg0, %arg1 : tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @power(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = stablehlo.power %arg0, %arg1 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @power(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = stablehlo.power %arg0, %arg1 : tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @remainder(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
 %0 = stablehlo.remainder %arg0, %arg1 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
 func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @remainder(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = stablehlo.remainder %arg0, %arg1 : tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @rsqrt(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
 %0 = stablehlo.rsqrt %arg0 : tensor<1x1x1x96xf32>
 func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @rsqrt(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.rsqrt %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @select(%arg0: tensor<1x30x1xi1>, %arg1: tensor<1x30x1xi32>, %arg2: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
 %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<1x30x1xi1>, tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
 func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @select(%arg0: tensor<1x30x1xi1>, %arg1: tensor<1x30x1xi32>, %arg2: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x30x1xi1>, tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @sub(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
 %0 = stablehlo.subtract %arg0, %arg1 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
 func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @sub(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = stablehlo.subtract %arg0, %arg1 : tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @tanh(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
 %0 = stablehlo.tanh %arg0 : tensor<1x1x1x96xf32>
 func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @tanh(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = stablehlo.tanh %arg0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @iota() -> tensor<3x4xf32> {
 %0 = stablehlo.iota dim = 0 : tensor<3x4xf32>
 return %0 : tensor<3x4xf32>
}

//CHECK:func.func private @iota() -> tensor<3x4xf32> {
//CHECK-NEXT: %0 = stablehlo.iota dim = 0 : tensor<3x4xf32>
//CHECK-NEXT: return %0 : tensor<3x4xf32>
//CHECK-NEXT:}

func.func @compare(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i1> {
 %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
 func.return %0 : tensor<i1>
}

//CHECK:func.func private @compare(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i1> {
//CHECK-NEXT: %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
//CHECK-NEXT: return %0 : tensor<i1>
//CHECK-NEXT:}

func.func @dynamic_update_slice(%arg0: tensor<4x4xi64>, %arg1: tensor<2x3xi64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<4x4xi64> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
  return %0 : tensor<4x4xi64>
}

//CHECK:func.func private @dynamic_update_slice(%arg0: tensor<4x4xi64>, %arg1: tensor<2x3xi64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<4x4xi64> {
//CHECK-NEXT: %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
//CHECK-NEXT: return %0 : tensor<4x4xi64>
//CHECK-NEXT:}

func.func @dyanmic_slice(%arg0: tensor<3x3xi64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<3x3xi64> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {
    slice_sizes = dense<[3, 3]> : tensor<2xi64>
  } : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
  return %0 : tensor<3x3xi64>
}

//CHECK:func.func private @dyanmic_slice(%arg0: tensor<3x3xi64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<3x3xi64> {
//CHECK-NEXT: %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [3, 3] : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64> 
//CHECK-NEXT: return %0 : tensor<3x3xi64>
//CHECK-NEXT:}

func.func @pad(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x161x1xf32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x161x1xf32>
  return %0 : tensor<1x161x1xf32>
}

//CHECK:func.func private @pad(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x161x1xf32> {
//CHECK-NEXT: %0 = stablehlo.pad %arg0, %arg1, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x161x1xf32>
//CHECK-NEXT: return %0 : tensor<1x161x1xf32>
//CHECK-NEXT:}

func.func @convert(%arg0: tensor<2xf64>) -> tensor<2xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

//CHECK:func.func private @convert(%arg0: tensor<2xf64>) -> tensor<2xf32> {
//CHECK-NEXT: %0 = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
//CHECK-NEXT: return %0 : tensor<2xf32>
//CHECK-NEXT:}

func.func @reduce_window(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x160x1xf32> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg23: tensor<f32>, %arg24: tensor<f32>):
      %1112 = stablehlo.add %arg23, %arg24 : tensor<f32>
      stablehlo.return %1112 : tensor<f32>
    }) {padding = dense<[[0, 0], [159, 0], [0, 0]]> : tensor<3x2xi64>, window_dimensions = dense<[1, 160, 1]> : tensor<3xi64>, window_strides = dense<1> : tensor<3xi64>} : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x160x1xf32>
  return %0 : tensor<1x160x1xf32>
}

//CHECK:func.func private @reduce_window(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x160x1xf32> {
//CHECK-NEXT: %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
//CHECK-NEXT:  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
//CHECK-NEXT:   %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
//CHECK-NEXT:   stablehlo.return %1 : tensor<f32>
//CHECK-NEXT{LITERAL}:  }) {padding = dense<[[0, 0], [159, 0], [0, 0]]> : tensor<3x2xi64>, window_dimensions = dense<[1, 160, 1]> : tensor<3xi64>, window_strides = dense<1> : tensor<3xi64>} : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x160x1xf32>
//CHECK-NEXT: return %0 : tensor<1x160x1xf32>
//CHECK-NEXT:}

func.func @dot_general(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %0 : tensor<1x1x64xf32>
}

//CHECK:func.func private @dot_general(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
//CHECK-NEXT: %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
//CHECK-NEXT: return %0 : tensor<1x1x64xf32>
//CHECK-NEXT:}

func.func @sort(%arg0: tensor<448xf32>, %arg1: tensor<448xi32>) -> tensor<448xf32> {
  %0, %1 = "stablehlo.sort"(%arg0, %arg1) ({
    ^bb0(%arg23: tensor<f32>, %arg24: tensor<f32>, %arg25: tensor<i32>, %arg26: tensor<i32>):
      %1112 = stablehlo.compare  GT, %arg23, %arg24,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1112 : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<448xf32>, tensor<448xi32>) -> (tensor<448xf32>, tensor<448xi32>)
  return %0 : tensor<448xf32>
}

//CHECK:func.func private @sort(%arg0: tensor<448xf32>, %arg1: tensor<448xi32>) -> tensor<448xf32> {
//CHECK-NEXT: %0:2 = "stablehlo.sort"(%arg0, %arg1) ({
//CHECK-NEXT:  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
//CHECK-NEXT:   %1 = stablehlo.compare  GT, %arg2, %arg3,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
//CHECK-NEXT:   stablehlo.return %1 : tensor<i1>
//CHECK-NEXT:  }) {dimension = 0 : i64, is_stable = true} : (tensor<448xf32>, tensor<448xi32>) -> (tensor<448xf32>, tensor<448xi32>)
//CHECK-NEXT: return %0#0 : tensor<448xf32>
//CHECK-NEXT:}


func.func @while(%init_i: tensor<i64>, %init_sum: tensor<i64>) -> tensor<i64>{
  %0, %1 = "stablehlo.while"(%init_i, %init_sum) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %cond = "stablehlo.compare"(%arg0, %arg1) {
        comparison_direction = #stablehlo<comparison_direction LT>
      } : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %new_sum = stablehlo.add %arg1, %arg1 : tensor<i64>
      %new_i = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
  }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
  return %0 : tensor<i64>
}

//CHECK:func.func private @while(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
//CHECK-NEXT: %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %arg1) : tensor<i64>, tensor<i64>
//CHECK-NEXT:  cond {
//CHECK-NEXT:   %1 = stablehlo.compare LT, %iterArg, %iterArg_0, NOTYPE : (tensor<i64>, tensor<i64>) -> tensor<i1>
//CHECK-NEXT:   stablehlo.return %1 : tensor<i1>
//CHECK-NEXT:  } do {
//CHECK-NEXT:   %1 = stablehlo.add %iterArg_0, %iterArg_0 : tensor<i64>
//CHECK-NEXT:   %2 = stablehlo.add %iterArg, %iterArg_0 : tensor<i64>
//CHECK-NEXT:   stablehlo.return %2, %1 : tensor<i64>, tensor<i64>
//CHECK-NEXT: }
//CHECK-NEXT: return %0#0 : tensor<i64>
//CHECK-NEXT:}

func.func @scatter(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    "stablehlo.return"(%lhs) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// CHECK-LABEL: func.func private @scatter(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
// CHECK-NEXT:  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f32>
// CHECK-NEXT:  }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
// CHECK-NEXT:  return %0 : tensor<200x100x300xf32>
// CHECK-NEXT: }

func.func @scatter_multiple_ops(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %res = stablehlo.add %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%res) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  %1 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %res = stablehlo.multiply %lhs, %rhs : tensor<f32>
    "stablehlo.return"(%res) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %1 : tensor<200x100x300xf32>
}

// CHECK-LABEL: func.func private @scatter_multiple_ops(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
// CHECK-NEXT:   %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:     %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:   }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
// CHECK-NEXT:   %1 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:   ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:     %2 = stablehlo.multiply %arg3, %arg4 : tensor<f32>
// CHECK-NEXT:     stablehlo.return %2 : tensor<f32>
// CHECK-NEXT:   }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
// CHECK-NEXT:   return %1 : tensor<200x100x300xf32>
// CHECK-NEXT: }

func.func @gather(%operand: tensor<3x4x2xi32>, %start_indices: tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>{
  %result = "stablehlo.gather"(%operand, %start_indices) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [2, 3],
    collapsed_slice_dims = [0],
    start_index_map = [1, 0],
    index_vector_dim = 2>,
  slice_sizes = dense<[1, 2, 2]> : tensor<3xi64>,
  indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>
  return %result : tensor<2x3x2x2xi32>
}


// CHECK: func.func private @gather(%arg0: tensor<3x4x2xi32>, %arg1: tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32> {
// CHECK-NEXT: %0 = "stablehlo.gather"(%arg0, %arg1) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 2]> : tensor<3xi64>} : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>
// CHECK-NEXT: return %0 : tensor<2x3x2x2xi32>
// CHECK-NEXT:}

func.func @transpose(%arg0: tensor<2x3x2xi32>) -> tensor<2x3x2xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = dense<[2, 1, 0]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  return %0 : tensor<2x3x2xi32>
}

// CHECK:func.func private @transpose(%arg0: tensor<2x3x2xi32>) -> tensor<2x3x2xi32> {
// CHECK-NEXT:  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// CHECK-NEXT:  return %0 : tensor<2x3x2xi32>
// CHECK-NEXT:}

func.func @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm DEFAULT>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>)
  func.return %output_state, %output : tensor<2xui64>, tensor<10x12xui32>
}

// CHECK:func.func private @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
// CHECK-NEXT:  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>)
// CHECK-NEXT:  return %output_state, %output : tensor<2xui64>, tensor<10x12xui32>
// CHECK-NEXT:}

func.func @rng_bit_generator_threefry(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui64>) {
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui64>)
  func.return %output_state, %output : tensor<2xui64>, tensor<10x12xui64>
}

// CHECK:func.func private @rng_bit_generator_threefry(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui64>) {
// CHECK-NEXT:  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui64>)
// CHECK-NEXT:  return %output_state, %output : tensor<2xui64>, tensor<10x12xui64>
// CHECK-NEXT:}


func.func @rng_bit_generator_philox(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xi32>) {
  %output_state, %output = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm PHILOX>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xi32>)
  func.return %output_state, %output : tensor<2xui64>, tensor<10x12xi32>
}

// CHECK:func.func private @rng_bit_generator_philox(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xi32>) {
// CHECK-NEXT:  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = PHILOX : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xi32>)
// CHECK-NEXT:  return %output_state, %output : tensor<2xui64>, tensor<10x12xi32>
// CHECK-NEXT:}