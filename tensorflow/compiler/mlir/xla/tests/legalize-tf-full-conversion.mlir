// RUN: tf-opt %s -xla-legalize-tf -split-input-file -verify-diagnostics

// expected-error@below{{The following operations cannot be legalized: tf.NoOp (count: 1); tf_executor.fetch (count: 1); tf_executor.graph (count: 1); tf_executor.island (count: 1); tf_executor.yield (count: 1). These legalization failure(s) may be due to missing TF to HLO lowerings and/or unsupported attributes, etc.}}
// expected-error@below{{Emitting more detail about one op that failed to legalize...}}
func @tf_executor_graph_op() {
    tf_executor.graph {
      %0 = tf_executor.island {
        // expected-error@+1 {{'tf.NoOp' op is not legalizable}}
        "tf.NoOp"() {} : () -> ()
        tf_executor.yield
      }
      tf_executor.fetch
    }
    return
}

// -----

// expected-error@below{{The following operations cannot be legalized: tf.OpA (count: 1). These legalization failure(s) may be due to missing TF to HLO lowerings and/or unsupported attributes, etc.}}
func @tf_unknown_op(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error@+1 {{'tf.OpA' op is not legalizable}}
  %0 = "tf.OpA"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----

func @tf_known_op(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "tf.AddV2"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----

// expected-error@below{{The following operations cannot be legalized: tf.OpA (count: 1); tf.OpB (count: 2). These legalization failure(s) may be due to missing TF to HLO lowerings and/or unsupported attributes, etc.}}
// expected-error@below{{Emitting more detail about one op that failed to legalize...}}
func @tf_unknown_known_mix(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error@+1 {{'tf.OpA' op is not legalizable}}
  %0 = "tf.OpA"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = "tf.OpB"(%0, %0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %2 = "tf.AddV2"(%1, %1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %3 = "tf.OpB"(%2, %2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %2: tensor<2xi32>
}
