// RUN: tf-opt %s -xla-legalize-tf -split-input-file -verify-diagnostics

// expected-error@below{{The following operations cannot be legalized: tf.OpA (count: 1). These legalization failure(s) may be due to missing TF to HLO lowerings and/or unsupported attributes, etc.}}
func.func @tf_unknown_op(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error@+1 {{'tf.OpA' op is not legalizable}}
  %0 = "tf.OpA"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// -----

func.func @tf_known_op(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "tf.AddV2"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// -----

// expected-error@below{{The following operations cannot be legalized: tf.OpA (count: 1); tf.OpB (count: 2). These legalization failure(s) may be due to missing TF to HLO lowerings and/or unsupported attributes, etc.}}
// expected-error@below{{Emitting more detail about one op that failed to legalize...}}
func.func @tf_unknown_known_mix(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error@+1 {{'tf.OpA' op is not legalizable}}
  %0 = "tf.OpA"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = "tf.OpB"(%0, %0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %2 = "tf.AddV2"(%1, %1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %3 = "tf.OpB"(%2, %2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %2: tensor<2xi32>
}
