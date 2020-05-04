// RUN: tf-opt -tf-to-corert %s | FileCheck %s

// CHECK-LABEL: func @derived_attrs
func @derived_attrs(
  %serialized: tensor<?x!tf.string>,
  %names: tensor<0x!tf.string>,
  %sparse_keys: tensor<0x!tf.string>,
  %dense_keys: tensor<1x!tf.string>,
  %ragged_keys: tensor<0x!tf.string>,
  %dense_default: tensor<0xi64>) -> tensor<?xi64> {

  %dense_value =
    "tf.ParseExampleV2"(%serialized, %names, %sparse_keys, %dense_keys, %ragged_keys, %dense_default)
    // CHECK: Tdense = [i64]
    // CHECK-SAME: dense_shapes = [#corert.shape<>]
    { device = "cpu", num_sparse = 0 : i64, dense_shapes = [#tf.shape<>], result_segment_sizes = dense<[0, 0, 0, 1, 0, 0]> : vector<6xi32>}
      : (tensor<?x!tf.string>, tensor<0x!tf.string>, tensor<0x!tf.string>, tensor<1x!tf.string>, tensor<0x!tf.string>, tensor<0xi64>)
      -> tensor<?xi64>

  return %dense_value : tensor<?xi64>
}
