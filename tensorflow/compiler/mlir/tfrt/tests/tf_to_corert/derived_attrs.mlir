// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @derived_attrs
func.func @derived_attrs(
  %serialized: tensor<?x!tf_type.string>,
  %names: tensor<0x!tf_type.string>,
  %sparse_keys: tensor<0x!tf_type.string>,
  %dense_keys: tensor<1x!tf_type.string>,
  %ragged_keys: tensor<0x!tf_type.string>,
  %dense_default: tensor<0xi64>) -> tensor<?xi64> {

  %dense_value =
    "tf.ParseExampleV2"(%serialized, %names, %sparse_keys, %dense_keys, %ragged_keys, %dense_default)
    // CHECK: Tdense = [i64]
    // CHECK-SAME: dense_shapes = [#corert.shape<>]
    { device = "/device:CPU:0", num_sparse = 0 : i64, dense_shapes = [#tf_type.shape<>], resultSegmentSizes = array<i32: 0, 0, 0, 1, 0, 0>}
      : (tensor<?x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0x!tf_type.string>, tensor<1x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0xi64>)
      -> tensor<?xi64>

  func.return %dense_value : tensor<?xi64>
}
