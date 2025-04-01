// RUN: stablehlo-quant-opt %s -split-input-file -tf-xla-call-module-serialization -stablehlo-convert-xla-call-module-op-to-bfloat16 -tf-xla-call-module-deserialization | FileCheck %s

// ConvertXlaCallModuleOpToBfloat16Pass works on XlaCallModuleOps with
// serialized modules. Which makes verification difficult. Therefore we add
// (de)serialization passes so that the input and output are deserializated
// StableHLO functions.

// CHECK-LABEL: module
module {
  // CHECK-LABEL: func @main
  // CHECK-SAME: %[[ARG_0:.*]]: tensor<10xf32>, %[[ARG_1:.*]]: tensor<10xf32>, %[[ARG_2:.*]]: tensor<6xi32>
  func.func @main(
      %arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<6xi32>
    ) -> (tensor<10xf32>, tensor<6xi32>) {
    // CHECK: %[[CAST_0:.*]] = "tf.Cast"(%[[ARG_0]]) <{Truncate = false}> : (tensor<10xf32>) -> tensor<10xbf16>
    // CHECK: %[[CAST_1:.*]] = "tf.Cast"(%[[ARG_1]]) <{Truncate = false}> : (tensor<10xf32>) -> tensor<10xbf16>
    // CHECK: %[[RESULT:.*]]:2 = "tf.XlaCallModule"(%[[CAST_0]], %[[CAST_1]], %[[ARG_2]])
    // CHECK-SAME: _stablehlo_version = "1.0.0"
    // CHECK-SAME: (tensor<10xbf16>, tensor<10xbf16>, tensor<6xi32>) -> (tensor<10xbf16>, tensor<6xi32>)
    // CHECK: %[[RESULT_CAST:.*]] = "tf.Cast"(%[[RESULT]]#0) <{Truncate = false}> : (tensor<10xbf16>) -> tensor<10xf32>
    %0:2 = "tf.XlaCallModule"(%arg0, %arg1, %arg2) {
      Sout = [#tf_type.shape<10>], dim_args_spec = [],
      _entry_function = @main_0,
      _stablehlo_version = "1.0.0",
      _stablehlo_module_attrs = { mhlo.num_partitions = 1 }, module = "",
      platforms = [], version = 5 : i64
    } : (tensor<10xf32>, tensor<10xf32>, tensor<6xi32>) -> (tensor<10xf32>, tensor<6xi32>)
    // CHECK: return %[[RESULT_CAST]], %[[RESULT]]#1 : tensor<10xf32>, tensor<6xi32>
    func.return %0#0, %0#1 : tensor<10xf32>, tensor<6xi32>
  }

  // CHECK-LABEL: func private @main_0
  // CHECK-SAME: %[[ARG_0:.*]]: tensor<10xbf16>, %[[ARG_1:.*]]: tensor<10xbf16>, %[[ARG_2:.*]]: tensor<6xi32>
  func.func private @main_0(
      %arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<6xi32>
    ) -> (tensor<10xf32>, tensor<6xi32>) attributes {_from_xla_call_module} {
    // CHECK: %[[ADD:.*]] = stablehlo.add %[[ARG_0]], %[[ARG_1]] : tensor<10xbf16>
    %0 = stablehlo.add %arg0, %arg1 : tensor<10xf32>
    // CHECK: return %[[ADD]], %[[ARG_2]] : tensor<10xbf16>, tensor<6xi32>
    return %0, %arg2 : tensor<10xf32>, tensor<6xi32>
  }
}
