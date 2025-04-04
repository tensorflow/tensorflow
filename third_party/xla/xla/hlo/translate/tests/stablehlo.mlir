// RUN: hlo-translate -mlir-to-hlo -split-input-file %s | FileCheck %s
// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false -split-input-file %s | FileCheck %s --check-prefix CHECK-DIRECT

// Tests for all stablehlo ops to validate stablehlo -> hlo conversion.


// CHECK-LABEL: HloModule

// CHECK: %[[ARG0:.*]] = f32[4] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[4] parameter(1)
// CHECK: ROOT %add.3 = f32[4] add(%[[ARG0]], %[[ARG1]])
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>  func.return %0 : tensor<4xf32>
}
// CHECK-DIRECT: stablehlo.add

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[1,10]{1,0}, s32[1,10]{1,0}, f32[], s32[])->(f32[1]{0}, s32[1]{0})}

// CHECK:       %[[$region_0_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_6:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_2_8:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  %[[maximum_10:[^ ]+]] = f32[] maximum(%[[Arg_0_6]], %[[Arg_2_8]]),
// CHECK-NEXT:  %[[Arg_1_7:[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:  %[[Arg_3_9:[^ ]+]] = s32[] parameter(3)
// CHECK-NEXT:  %[[maximum_11:[^ ]+]] = s32[] maximum(%[[Arg_1_7]], %[[Arg_3_9]]),
// CHECK-NEXT:  ROOT %[[tuple_12:[^ ]+]] = (f32[], s32[]) tuple(%[[maximum_10]], %[[maximum_11]])

// CHECK:  ENTRY %[[$main_17:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[1,10] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[1,10] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = s32[] parameter(3)
// CHECK-NEXT:  %[[reduce_13:[^ ]+]] = (f32[1], s32[1]) reduce(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]], %[[Arg_3_4]]), dimensions={1}, to_apply=%[[$region_0_5]],
// CHECK-NEXT:  %[[get_tuple_element_14:[^ ]+]] = f32[1] get-tuple-element(%[[reduce_13]]), index=0,
// CHECK-NEXT:  %[[get_tuple_element_15:[^ ]+]] = s32[1] get-tuple-element(%[[reduce_13]]), index=1,
// CHECK-NEXT:  ROOT %[[tuple_16:[^ ]+]] = (f32[1], s32[1]) tuple(%[[get_tuple_element_14]], %[[get_tuple_element_15]])
func.func @main(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0:2 = stablehlo.reduce(%arg0 init: %arg2), (%arg1 init: %arg3) across dimensions = [1] : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
    reducer(%arg4: tensor<f32>, %arg6: tensor<f32>) (%arg5: tensor<i32>, %arg7: tensor<i32>)  {
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<1xf32>, tensor<1xi32>
}

// CHECK-DIRECT: stablehlo.reduce
